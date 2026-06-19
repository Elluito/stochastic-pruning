import os
import pickle
import typing
import pandas as pd
import wandb
import optuna
import omegaconf
import copy
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.nn.utils.prune as prune
from functools import partial
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.metrics as igm
import matplotlib
import matplotlib.pyplot as plt
import time
from torch.utils.data import random_split
import logging
import seaborn as sns
from sparse_ensemble_utils import get_layer_dict, is_prunable_module, \
    sparsity, test, restricted_fine_tune_measure_flops, \
    restricted_fine_tune_measure_flops_sto_and_deterministic, \
    get_random_batch, check_for_layers_collapse, get_mask, \
    apply_mask, \
    measure_gradient_flow_only
from itertools import cycle
from shrinkbench.metrics.flops import flops
from pathlib import Path

print("safe All imports")
plt.rcParams["mathtext.fontset"] = "cm"

# enable cuda devices

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"

sns.reset_orig()
sns.reset_defaults()
matplotlib.rc_file_defaults()
fs = 12
fig_size = (3, 3)
legend_multiplier = 0.6
labels_multiplier = 0.8
ticks_multiplier = 0.6
plt.rcParams.update({
    "axes.linewidth": 0.5,
    'axes.edgecolor': 'black',
    "grid.linewidth": 0.4,
    "lines.linewidth": 1,
    'xtick.bottom': True,
    'xtick.color': 'black',
    "xtick.direction": "out",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    'ytick.left': True,
    'ytick.color': 'black',
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    # "figure.figsize": [3.3, 2.5],
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{bm} \usepackage{amsmath}",
})


def strip_prefix(net_state_dict: dict):
    new_dict = {}
    for k, v in net_state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict


def load_model(net, path):
    state_dict = torch.load(path,map_location=device)
    if "net" in state_dict.keys():
        net.load_state_dict(strip_prefix(state_dict["net"]))
    else:
        net.load_state_dict(torch.load(path))


    # Go trough all the above_cutoff_index


######################### Noise adding functions #################################################################
def add_geometric_gaussian_noise_to_weights(m, sigma=0.2):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            m.weight.multiply_(torch.normal(mean=torch.ones_like(m.weight), std=sigma).to(m.weight.device))


def add_gaussian_noise_to_weights(m, sigma=0.01, adaptive=False):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            if adaptive:
                sigma_adaptive = torch.quantile(torch.abs(m.weight), torch.tensor([0.50]))
                m.weight.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma_adaptive).to(m.weight.device))
            else:
                m.weight.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device))


def add_geogaussian_noise_to_layers(model: torch.nn.Module, sigma_per_layer: dict, exclude_layers: list = []):
    named_modules = model.named_modules()

    with torch.no_grad():
        for name, m in named_modules:
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m,
                                                                                     nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layers:
                sigma = sigma_per_layer[name]
                m.weight.multiply_(torch.normal(mean=torch.ones_like(m.weight), std=sigma).to(m.weight.device))


def add_gaussian_noise_to_layers(model: torch.nn.Module, sigma_per_layer: dict, iterative: bool = False, exclude_layers:
list =
[]):
    named_modules = model.named_modules()
    with torch.no_grad():
        for name, m in named_modules:
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m,
                                                                                     nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layers:
                sigma = sigma_per_layer[name]
                if "weight_mask" in dict(m.named_buffers()).keys() and iterative:
                    weight_mask = dict(m.named_buffers())["weight_mask"]
                    noise = torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device)
                    noise.mul_(weight_mask)
                    m.weight.data.add_(noise)
                else:
                    m.weight.data.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device))


def get_noisy_sample_sigma_per_layer(net: torch.nn.Module, cfg: omegaconf.DictConfig, sigma_per_layer, clone=True):
    current_model = None
    if clone:
        current_model = copy.deepcopy(net)
    else:
        current_model = net
    if cfg.noise == "gaussian":
        add_gaussian_noise_to_layers(current_model, sigma_per_layer=sigma_per_layer, exclude_layers=cfg.exclude_layers)
    elif cfg.noise == "geogaussian":
        add_geogaussian_noise_to_layers(current_model, sigma_per_layer=sigma_per_layer,
                                        exclude_layers=cfg.exclude_layers)
    return current_model


# def get_noisy_sample_pruned_net_work(net: torch.nn.Module, cfg: omegaconf.DictConfig, sigma_per_layer):
def get_noisy_sample(net: torch.nn.Module, cfg: omegaconf.DictConfig, noise_on_LAMP_scores: bool = False):
    current_model = copy.deepcopy(net)
    if cfg.noise == "gaussian":
        current_model.apply(partial(add_gaussian_noise_to_weights, sigma=cfg.sigma))

    elif cfg.noise == "geogaussian":
        current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=cfg.sigma))
    return current_model




def weights_to_prune(model: torch.nn.Module, exclude_layer_list=[], param_name="weight"):
    modules = []
    for name, m in model.named_modules():
        if hasattr(m, param_name) and type(m) != nn.BatchNorm1d and not isinstance(m,
                                                                                   nn.BatchNorm2d) and not isinstance(
            m, nn.BatchNorm3d) and name not in exclude_layer_list:
            modules.append((m, param_name))
            # print(name)

    return modules


def filter_modules(model: torch.nn.Module, exclude_layer_list=[]):
    modules = []
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            modules.append((name, m))
            # print(name)

    return modules


def remove_reparametrization(model, name_module="", exclude_layer_list: list = []):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            if name_module == "":
                prune.remove(m, "weight")
            if name == name_module:
                prune.remove(m, "weight")
                break


def objective_function(sto_performance, deter_performance, pruning_rate):
    if sto_performance > deter_performance:
        return ((sto_performance - deter_performance)) * pruning_rate
    if sto_performance <= deter_performance:
        return ((sto_performance - deter_performance))


def find_pr_sigma_MOO_for_dataset_architecture_fine_tuned_GMP(trial: optuna.trial.Trial, cfg, one_batch=True,
                                                              use_population=True, use_log_sigma=False, Fx=1):
    # in theory cfg is available everywhere because it is define on the if name ==__main__ section
    net = get_model(cfg)

    train_loader, val_loader, test_loader = get_datasets(cfg)

    # dense_performance = test(net, use_cuda=True, testloader=val_loader, verbose=0, one_batch=one_batch)
    if use_log_sigma:
        sample_sigma = trial.suggest_float("sigma", 0.0001, 0.01, log=True)
    else:
        sample_sigma = trial.suggest_float("sigma", 0.0001, 0.01)
    sample_pruning_rate = trial.suggest_float("pruning_rate", 0.01, 0.99)

    # def objective_function(stochastic_performance,deter_performance, pruning_rate):
    #     return ((stochastic_performance - deter_performance)) * pruning_rate

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [sample_sigma] * number_of_layers))
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.amount = sample_pruning_rate

    pruned_model = copy.deepcopy(net)
    prune_function(pruned_model, cfg_copy)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)

    models = []

    # Add small noise just to get tiny variations of the deterministic case
    det_performance = test(pruned_model, use_cuda=True, testloader=val_loader, verbose=0, one_batch=one_batch)
    print("Det performance: {}".format(det_performance))

    if use_population:
        performance_of_models = []
        for individual_index in range(5):
            ############### Here I ask for pr and for sigma ###################################

            models.append(current_model)
            # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
            prune_function(current_model, cfg_copy)

            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
            # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
            stochastic_performance = test(current_model, use_cuda=True, testloader=val_loader, verbose=0,
                                          one_batch=one_batch)
            # Dense stochastic performance
            performance_of_models.append(stochastic_performance)
        performance_of_models = np.array(performance_of_models)
        best_model = models[np.argmax(performance_of_models)]
        #
        # median = np.median(performance_of_models)
        # print("Median of population performance: {}".format(median))
        # average_difference_performance = det_performance - performance_of_models
        # fitness_function_median = objective_function(median, det_performance, sample_pruning_rate)
        # fitness_function_vector = np.array(list(map()))objective_function(performance_of_models, det_performance,sample_pruning_rate)
        # average_fitness_function = fitness_function_vector.mean()
        stochastic_fine_tuned_performance, diference = restricted_fine_tune_measure_flops_sto_and_deterministic(
            best_model, pruned_model, train_loader, test_loader, epochs=100, cfg=cfg)
        if Fx == 1:
            return median, fitness_function_median
        if Fx == 3:
            return stochastic_fine_tuned_performance, diference
        else:
            return median, sample_pruning_rate
    else:

        stochastic_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
        prune_function(stochastic_model, cfg_copy)

        remove_reparametrization(stochastic_model, exclude_layer_list=cfg.exclude_layers)
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        stochastic_performance = test(stochastic_model, use_cuda=True, testloader=val_loader, verbose=0,
                                      one_batch=one_batch)
        fitness_function_median = objective_function(stochastic_performance, det_performance, sample_pruning_rate)
        print("Stochastic performance: {}".format(stochastic_performance))
        if Fx == 1:
            return stochastic_performance, fitness_function_median
        else:
            return stochastic_performance, sample_pruning_rate


def find_pr_sigma_MOO_for_dataset_architecture_one_shot_GMP(trial: optuna.trial.Trial, cfg, one_batch=True,
                                                            use_population=True, use_log_sigma=False, Fx=1):
    # in theory cfg is available everywhere because it is define on the if name ==__main__ section
    net = get_model(cfg)
    train, val_loader, test_loader = get_datasets(cfg)

    # dense_performance = test(net, use_cuda=True, testloader=val_loader, verbose=0, one_batch=one_batch)
    if use_log_sigma:
        sample_sigma = trial.suggest_float("sigma", 0.0001, 0.01, log=True)
    else:
        sample_sigma = trial.suggest_float("sigma", 0.0001, 0.01)
    sample_pruning_rate = trial.suggest_float("pruning_rate", 0.01, 0.99)

    # def objective_function(stochastic_performance,deter_performance, pruning_rate):
    #     return ((stochastic_performance - deter_performance)) * pruning_rate

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [sample_sigma] * number_of_layers))
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.amount = sample_pruning_rate

    pruned_model = copy.deepcopy(net)
    prune_function(pruned_model, cfg_copy)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)

    # Add small noise just to get tiny variations of the deterministic case
    det_performance = test(pruned_model, use_cuda=True, testloader=val_loader, verbose=0, one_batch=one_batch)
    print("Det performance: {}".format(det_performance))

    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])
    # sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    if use_population:
        performance_of_models = []
        for individual_index in range(5):
            ############### Here I ask for pr and for sigma ###################################

            current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
            # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
            prune_function(current_model, cfg_copy)

            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
            # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
            stochastic_performance = test(current_model, use_cuda=True, testloader=val_loader, verbose=0,
                                          one_batch=one_batch)
            # Dense stochastic performance
            performance_of_models.append(stochastic_performance)
        performance_of_models = np.array(performance_of_models)
        median = np.median(performance_of_models)
        print("Median of population performance: {}".format(median))
        average_difference_performance = det_performance - performance_of_models
        fitness_function_median = objective_function(median, det_performance, sample_pruning_rate)
        # fitness_function_vector = np.array(list(map()))objective_function(performance_of_models, det_performance,sample_pruning_rate)
        # average_fitness_function = fitness_function_vector.mean()
        if Fx == 1:
            return median, fitness_function_median
        else:
            return median, sample_pruning_rate
    else:

        stochastic_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
        prune_function(stochastic_model, cfg_copy)

        remove_reparametrization(stochastic_model, exclude_layer_list=cfg.exclude_layers)
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        stochastic_performance = test(stochastic_model, use_cuda=True, testloader=val_loader, verbose=0,
                                      one_batch=one_batch)
        fitness_function_median = objective_function(stochastic_performance, det_performance, sample_pruning_rate)
        print("Stochastic performance: {}".format(stochastic_performance))
        if Fx == 1:
            return stochastic_performance, fitness_function_median
        else:
            return stochastic_performance, sample_pruning_rate


def test_pr_sigma_combination(cfg, pr, sigma, cal_val=False):
    net = get_model(cfg)
    train, val, testloader = get_datasets(cfg)

    pruned_model = copy.deepcopy(net)
    cfg.amount = pr
    prune_function(pruned_model, cfg)
    remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # Add small noise just to get tiny variations of the deterministic ase
    det_performance = test(pruned_model, use_cuda=True, testloader=testloader, verbose=0)
    if cal_val:
        det_performance_val = test(pruned_model, use_cuda=True, testloader=val, verbose=0)

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [sigma] * number_of_layers))
    # print("Deterministic performance on test set = {}".format(det_performance))
    if cal_val:
        print("Deterministic performance on val set = {}".format(det_performance_val))

    stochastic_performance = []
    performance_of_models = []
    GF_of_models = []
    if cal_val:
        performance_of_models_val = []
        GF_of_models_val = []

    det_test_GF_dict, det_val_GF_dict = measure_gradient_flow_only(pruned_model, val, testloader, cfg)

    for individual_index in range(11):
        ############### Here I ask for pr and for sigma ###################################

        current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
        prune_function(current_model, cfg)

        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        stochastic_performance = test(current_model, use_cuda=True, testloader=testloader, verbose=0)

        stochastic_test_GF_dict, stochastic_val_GF_dict = measure_gradient_flow_only(pruned_model, val, testloader, cfg)

        GF_of_models.append(stochastic_test_GF_dict["test_set_gradient_magnitude"][0])

        if cal_val:
            stochastic_performance_val = test(current_model, use_cuda=True, testloader=val, verbose=0)

            GF_of_models_val.append(stochastic_val_GF_dict["val_set_gradient_magnitude"][0])

        performance_of_models.append(stochastic_performance)
        if cal_val:
            performance_of_models_val.append(stochastic_performance_val)

    performance_of_models = np.array(performance_of_models)
    performance_of_models_median_index = np.argsort(performance_of_models)[len(performance_of_models) // 2]
    median = np.median(performance_of_models)
    GF_median_test = GF_of_models[performance_of_models_median_index]
    # print("Median accuracy of population: {}".format(median))
    fitness_function_median_test = objective_function(median, det_performance, pr)
    # print("Fintness function of the median on test set: {}".format(fitness_function_median))

    ##########  val functions###########################
    if cal_val:
        performance_of_models_val = np.array(performance_of_models_val)
        performance_of_models_val_median_index = np.argsort(performance_of_models_val)[
            len(performance_of_models_val) // 2]
        median_val = np.median(performance_of_models_val)
        GF_median_val = GF_of_models_val[performance_of_models_val_median_index]
        print("Median accuracy of population valset: {}".format(median_val))
        fitness_function_median_val = objective_function(median, det_performance_val, pr)
        print("Fintness function of the median on val set: {}".format(fitness_function_median_val))
        return median_val, median - det_performance, GF_median_val

    return fitness_function_median_test, median, median - det_performance, GF_median_test


def run_pr_sigma_fine_tuned_search_MOO_for_cfg(cfg, arg):
    # one_batch = False  # arg["one_batch"]
    one_batch = arg["one_batch"]
    one_batch_string = "whole_batch" if not one_batch else "one_batch"
    sampler = arg["sampler"]
    log_sigma = arg["log_sigma"]
    number_of_trials = arg["trials"]
    functions = arg["functions"]

    use_population = True if cfg["population"] > 1 else False
    function_string = "F1" if functions == 1 else "F2"
    if sampler == "nsga":
        # sampler = optuna.samplers.CmaEsSampler(restart_strategy="ipop",n_startup_trials=10,inc_popsize=2)
        sampler = optuna.samplers.NSGAIISampler()
    elif sampler == "tpe":
        sampler = optuna.samplers.TPESampler()
    else:
        raise Exception("Sampler {} is not suported for this experiment".format(sampler))

    # sampler = optuna.samplers.CmaEsSampler(n_startup_trials=10,popsize=4)
    # sampler = optuna.samplers.TPESamplerler()
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler,
                                study_name="stochastic-global-pr-and-sigma-optimisation-MOO-{}-{}-{}-{}".format(
                                    cfg.architecture,
                                    cfg.dataset, sampler, function_string),
                                storage="sqlite:///find_pr_sigma_fine_tuning_database_MOO_{}_{}_{}_{}_{}.dep".format(
                                    cfg.architecture,
                                    cfg.dataset,
                                    sampler,
                                    one_batch, function_string),
                                load_if_exists=True)

    study.optimize(
        lambda trial: find_pr_sigma_MOO_for_dataset_architecture_fine_tuned_GMP(trial, cfg, one_batch, use_population,
                                                                                use_log_sigma=log_sigma, Fx=functions),
        n_trials=arg["trials"])
    ##Save the sampler with pickle to be loaded later.
    with open("find_pr_sigma_database_pickle_MOO_fine_tuned_{}_{}_{}_{}_{}.pkl".format(
            cfg.architecture,
            cfg.dataset,
            sampler,
            one_batch, function_string), "wb") as fout:
        pickle.dump(study, fout)

    ################################
    ## Read the  MOO study from memory
    ################################

    with open("find_pr_sigma_database_pickle_MOO__fine_tuned_{}_{}_{}_{}_{}.pkl".format(
            cfg.architecture,
            cfg.dataset,
            sampler,
            one_batch, function_string), "rb") as fout:
        study = pickle.load(fout)
    print("MOO for : {} {} {} {} {}".format(
        cfg.architecture,
        cfg.dataset,
        sampler,
        one_batch, function_string))

    # print("Number of finished trials: {}".format(len(study.trials)))

    print("\n Best trial:")
    trials = study.best_trials
    # print("Size of the pareto front: {}".format(len(trials)))

    sigmas_list = []
    pruning_rate_list = []
    f1_list = []
    calculated_SP_performance_list = []
    f2_list = []
    difference_with_deterministic_list = []
    fitness_list = []
    GF_list = []

    # if functions == 1:
    #     trial_with_highest_difference = max(study.best_trials, key=lambda t: t.values[0])
    #     f1, f2 = trial_with_highest_difference.values
    #     print("  Values: {},{}".format(f1, f2))
    #     print("  Params: ")
    #     for key, value in trial_with_highest_difference.params.items():
    #         print("    {}: {}".format(key, value))
    #     fitness_function_on_test_se, t
    #     test_median_stochastic_performanci = test_pr_sigma_combination(cfg,
    #                                                                    trial_with_highest_difference.params[
    #                                                                        "pruning_rate"],
    #                                                                    trial_with_highest_difference.params[
    #                                                                        "sigma"])
    #     print(
    #         "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
    #             fitness_function_on_test_set, test_median_stochastic_performance,
    #             fitness_function_on_test_set / trial_with_highest_difference.params[
    #                 "pruning_rate"]))
    # else:
    #     trial_with_highest_difference = max(study.best_trials, key=lambda t: t.values[0])
    #
    #     f1, f2 = trial_with_highest_difference.values
    #     print("  Values: {},{}".format(f1, f2))
    #     print("  Params: ")
    #     for key, value in trial_with_highest_difference.params.items():
    #         print("    {}: {}".format(key, value))
    #     fitness_function_on_test_set, test_median_stochastic_performance = test_pr_sigma_combination(cfg,
    #                                                                                                  trial_with_highest_difference.params[
    #                                                                                                      "pruning_rate"],
    #                                                                                                  trial_with_highest_difference.params[
    #                                                                                                      "sigma"])
    #     print(
    #         "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
    #             fitness_function_on_test_set, test_median_stochastic_performance,
    #             fitness_function_on_test_set / trial_with_highest_difference.params[
    #                 "pruning_rate"]))

    for trial in trials:
        f1, f2 = trial.values
        pr, sigma = trial.params["pruning_rate"], trial.params["sigma"]
        f1_list.append(f1)
        f2_list.append(f2)
        print("  Values: {},{}".format(f1, f2))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        val_median_stochastic_performance, difference_in_test_set, GF_median_val = test_pr_sigma_combination(cfg,
                                                                                                             trial.params[
                                                                                                                 "pruning_rate"],
                                                                                                             trial.params[
                                                                                                                 "sigma"],
                                                                                                             cal_val=True)
        calculated_SP_performance_list.append(val_median_stochastic_performance)
        difference_with_deterministic_list.append(difference_in_test_set)
        # fitness_list.append(fitness_function_on_test_set)
        GF_list.append(GF_median_val)

        print(
            "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
                fitness_function_on_test_set, test_median_stochastic_performance,
                fitness_function_on_test_set / trial.params[
                    "pruning_rate"]))

        sigmas_list.append(sigma)
        pruning_rate_list.append(pr)

    # p = pd.read_csv("pareto_front_with_GF_{}_{}_{}_{}_{}.csv".format(cfg.architecture, cfg.dataset, sampler, function_string,
    #                                                          one_batch_string))

    # p["Sigma"] = sigmas_list

    p = pd.DataFrame({"Pruning rate": pruning_rate_list, "Stochastic Performance": f1_list,
                      "Stochastic Performance calculated": calculated_SP_performance_list,
                      "Sigma": sigmas_list, "Gradient Flow On Val post": GF_list,
                      "Difference with deterministic": difference_with_deterministic_list, "F2": f2_list})

    p.to_csv(
        "MOO_pareto_fronts/pareto_front_with_GF_fine_tuned_{}_{}_{}_{}_{}.csv".format(cfg.architecture, cfg.dataset,
                                                                                      sampler,
                                                                                      function_string,
                                                                                      one_batch_string), index=False)


def run_pr_sigma_search_MOO_for_cfg(cfg, arg):
    # one_batch = False  # arg["one_batch"]
    one_batch = arg["one_batch"]
    one_batch_string = "whole_batch" if not one_batch else "one_batch"
    sampler = arg["sampler"]
    log_sigma = arg["log_sigma"]
    number_of_trials = arg["trials"]
    functions = arg["functions"]

    use_population = True if cfg["population"] > 1 else False
    function_string = "F1" if functions == 1 else "F2"
    if sampler == "nsga":
        # sampler = optuna.samplers.CmaEsSampler(restart_strategy="ipop",n_startup_trials=10,inc_popsize=2)
        sampler = optuna.samplers.NSGAIISampler()
    elif sampler == "tpe":
        sampler = optuna.samplers.TPESampler()
    else:
        raise Exception("Sampler {} is not suported for this experiment".format(sampler))

    # sampler = optuna.samplers.CmaEsSampler(n_startup_trials=10,popsize=4)
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler,
                                study_name="stochastic-global-pr-and-sigma-optimisation-MOO-{}-{}-{}-{}".format(
                                    cfg.architecture,
                                    cfg.dataset, sampler, function_string),
                                storage="sqlite:///find_pr_sigma_database_MOO_{}_{}_{}_{}_{}.dep".format(
                                    cfg.architecture,
                                    cfg.dataset,
                                    sampler,
                                    one_batch, function_string),
                                load_if_exists=True)

    study.optimize(
        lambda trial: find_pr_sigma_MOO_for_dataset_architecture_one_shot_GMP(trial, cfg, one_batch, use_population,
                                                                              use_log_sigma=log_sigma, Fx=functions),
        n_trials=arg["trials"])
    # Save the sampler with pickle to be loaded later.
    with open("find_pr_sigma_database_pickle_MOO_{}_{}_{}_{}_{}.pkl".format(
            cfg.architecture,
            cfg.dataset,
            sampler,
            one_batch, function_string), "wb") as fout:
        pickle.dump(study, fout)

    #################################
    # Read the  MOO study from memory
    #################################

    # with open("find_pr_sigma_database_pickle_MOO_{}_{}_{}_{}_{}.pkl".format(
    #         cfg.architecture,
    #         cfg.dataset,
    #         sampler,
    #         one_batch, function_string), "rb") as fout:
    #     study = pickle.load(fout)
    # print("MOO for : {} {} {} {} {}".format(
    #     cfg.architecture,
    #     cfg.dataset,
    #     sampler,
    #     one_batch, function_string))
    #
    # # print("Number of finished trials: {}".format(len(study.trials)))
    #
    # print("\n Best trial:")
    # trials = study.best_trials
    # # print("Size of the pareto front: {}".format(len(trials)))
    #
    # sigmas_list = []
    # pruning_rate_list = []
    # f1_list = []
    # calculated_SP_performance_list=[]
    # f2_list = []
    # difference_with_deterministic_list = []
    # fitness_list = []
    # GF_list = []
    #
    # # if functions == 1:
    # #     trial_with_highest_difference = max(study.best_trials, key=lambda t: t.values[0])
    # #     f1, f2 = trial_with_highest_difference.values
    # #     print("  Values: {},{}".format(f1, f2))
    # #     print("  Params: ")
    # #     for key, value in trial_with_highest_difference.params.items():
    # #         print("    {}: {}".format(key, value))
    # #     fitness_function_on_test_se, t
    # #     test_median_stochastic_performanci = test_pr_sigma_combination(cfg,
    # #                                                                    trial_with_highest_difference.params[
    # #                                                                        "pruning_rate"],
    # #                                                                    trial_with_highest_difference.params[
    # #                                                                        "sigma"])
    # #     print(
    # #         "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
    # #             fitness_function_on_test_set, test_median_stochastic_performance,
    # #             fitness_function_on_test_set / trial_with_highest_difference.params[
    # #                 "pruning_rate"]))
    # # else:
    # #     trial_with_highest_difference = max(study.best_trials, key=lambda t: t.values[0])
    # #
    # #     f1, f2 = trial_with_highest_difference.values
    # #     print("  Values: {},{}".format(f1, f2))
    # #     print("  Params: ")
    # #     for key, value in trial_with_highest_difference.params.items():
    # #         print("    {}: {}".format(key, value))
    # #     fitness_function_on_test_set, test_median_stochastic_performance = test_pr_sigma_combination(cfg,
    # #                                                                                                  trial_with_highest_difference.params[
    # #                                                                                                      "pruning_rate"],
    # #                                                                                                  trial_with_highest_difference.params[
    # #                                                                                                      "sigma"])
    # #     print(
    # #         "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
    # #             fitness_function_on_test_set, test_median_stochastic_performance,
    # #             fitness_function_on_test_set / trial_with_highest_difference.params[
    # #                 "pruning_rate"]))
    #
    # for trial in trials:
    #     f1, f2 = trial.values
    #     pr, sigma = trial.params["pruning_rate"], trial.params["sigma"]
    #     f1_list.append(f1)
    #     f2_list.append(f2)
    #     print("  Values: {},{}".format(f1, f2))
    #
    #     print("  Params: ")
    #     for key, value in trial.params.items():
    #         print("    {}: {}".format(key, value))
    #     val_median_stochastic_performance, difference_in_test_set, GF_median_val = test_pr_sigma_combination(cfg,
    #                                                                                                          trial.params[
    #                                                                                                              "pruning_rate"],
    #                                                                                                          trial.params[
    #                                                                                                              "sigma"],
    #                                                                                                          cal_val=True)
    #     calculated_SP_performance_list.append(val_median_stochastic_performance)
    #     difference_with_deterministic_list.append(difference_in_test_set)
    #     # fitness_list.append(fitness_function_on_test_set)
    #     GF_list.append(GF_median_val)
    #
    #     # print(
    #     #     "Fitness function on Test {} , Median stochastic performance {} , Difference with deterministic {}".format(
    #     #         fitness_function_on_test_set, test_median_stochastic_performance,
    #     #         fitness_function_on_test_set / trial.params[
    #     #             "pruning_rate"]))
    #
    #     sigmas_list.append(sigma)
    #     pruning_rate_list.append(pr)
    #
    # # p = pd.read_csv("pareto_front_with_GF_{}_{}_{}_{}_{}.csv".format(cfg.architecture, cfg.dataset, sampler, function_string,
    # #                                                          one_batch_string))
    #
    # # p["Sigma"] = sigmas_list
    #
    # p = pd.DataFrame({"Pruning rate": pruning_rate_list, "Stochastic Performance": f1_list, "Stochastic Performance calculated":calculated_SP_performance_list,
    #                   "Sigma": sigmas_list, "Gradient Flow On Val post": GF_list,
    #                   "Difference with deterministic": difference_with_deterministic_list, "F2": f2_list})
    #
    # p.to_csv("MOO_pareto_fronts/pareto_front_with_GF_{}_{}_{}_{}_{}.csv".format(cfg.architecture, cfg.dataset, sampler,
    #                                                                             function_string,
    #                                                                             one_batch_string), index=False)

    #############################################################
    #                   plotting pareto front
    #############################################################

    # p = pd.read_csv(
    #     "MOO_pareto_fronts/pareto_front_with_GF_{}_{}_{}_{}_{}.csv".format(cfg.architecture, cfg.dataset, sampler,
    #                                                                        function_string,
    #                                                                        one_batch_string))
    # # p["Difference with deterministic"] = p["Fitness"]/p["Pruning rate"]
    # # plt.figure()
    # # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # # g = sns.scatterplot(data=p, x="Stochastic performance", y="Pruning rate", hue="Fitness",palette="deep")
    # fig, axs = plt.subplots(1, 1, figsize=fig_size, layout="compressed")
    # plt.title("{}".format(cfg.dataset.upper()))
    # cmap = plt.cm.get_cmap('magma')
    # # cmap = mpl.cm.viridis
    # # # cmap = (matplotlib.colors.ListedColormap(['royalblue', 'cyan', 'orange', 'red']))
    # # cmap = matplotlib.colors.ListedColormap(['royalblue', 'cyan', 'yellow', 'orange'])
    # # diff = p["Difference with deterministic"]
    # # min_val = diff.min()
    # # q25 = diff.quantile(q=0.25)
    # # q50 = diff.quantile(q=0.50)
    # # q75 = diff.quantile(q=0.75)
    # # max_val = diff.max()
    # # bounds =[min_val,q25,0,q50,q75]
    # # bounds.sort()
    #
    # # bounds = [p["Difference with deterministic"].min(), -0.5,0,0.1 ,p["Difference with deterministic"].max()]
    # # bounds = np.linspace(p["Difference with deterministic"].min(),p["Difference with deterministic"].max(),6)
    # # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    # # # fig.colorbar(
    # # #     mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    # # #     cax=ax,
    # # #     extend='both',
    # # #     ticks=bounds,
    # # #     spacing='proportional',
    # # #     orientation='horizontal',
    # # #     label='Discrete intervals, some other units',
    # # # )
    # #
    #
    # # sc = ax.scatter(xs=p["Stochastic performance"], ys=p["Pruning rate"], c=p["Difference with deterministic"], s=15, cmap=cmap,norm=norm)
    # p_lees_than_0 = p[p["Difference with deterministic"] < 0]
    # p_more_than_0 = p[p["Difference with deterministic"] >= 0]
    # sc_less_than_0 = axs.scatter(y=p_lees_than_0["Stochastic Performance"], x=p_lees_than_0["Pruning rate"],
    #                              facecolors='none', edgecolors='k', s=100)
    #
    # A = 15000
    # sizes = p_more_than_0["Sigma"] * A
    # color_values = p_more_than_0["Difference with deterministic"]
    #
    # sc = axs.scatter(y=p_more_than_0["Stochastic Performance"], x=p_more_than_0["Pruning rate"],
    #                  c=color_values, cmap=cmap,  # s=sizes*np.log(sizes),
    #                  norm=matplotlib.colors.PowerNorm(gamma=0.8), s=100)
    # # axs_y = axs.twinx()
    # # sc2 = axs_y.scatter(y=p_more_than_0["Gradient Flow On Val"], x=p_more_than_0["Pruning rate"],
    # #                  c=color_values, cmap=cmap,#s=sizes*np.log(sizes),
    # #                  norm=matplotlib.colors.PowerNorm(gamma=1.2))
    # # sc_less_than_0_2 = axs_y.scatter(y=p_lees_than_0["Gradient Flow On Val"], x=p_lees_than_0["Pruning rate"],
    # #                              facecolors='none', edgecolors='k', s=15)
    #
    # # sc = plt.scatter(y=p_more_than_0["Sigma"], x=p_more_than_0["Pruning rate"],
    # #                  c=color_values, cmap=cmap,#s=sizes*np.log(sizes),
    # #                  norm=matplotlib.colors.PowerNorm(gamma=1.2))
    #
    # # axins2 = inset_axes(axs, width="40%", height="40%", loc="lower left")
    #
    # axins = axs.inset_axes([0.13, 0.15, 0.35, 0.5])
    #
    # axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    #
    # # axins_cbar = axs.inset_axes([0.49, 0.12, 0.02, 0.5])
    #
    # scins = axins.scatter(y=p_more_than_0["Sigma"], x=p_more_than_0["Pruning rate"],
    #                       c=color_values, cmap=cmap,  # s=sizes*np.log(sizes),
    #                       norm=matplotlib.colors.PowerNorm(gamma=1))
    # scins = axins.scatter(y=p_lees_than_0["Sigma"], x=p_lees_than_0["Pruning rate"],
    #                       facecolors="none", edgecolors="k", )  # s=sizes*np.log(sizes),)
    # axins.tick_params(axis='both', which='major', labelsize=fs)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     axins.spines[axis].set_linewidth(1)
    #     axins.spines[axis].set_color('gray')
    #
    # axins.set_ylabel("$\sigma$", fontsize=20)
    # axins.set_xlabel("$\gamma$", fontsize=20)
    # # mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec='gray', lw=1)
    # # fig.colorbar(scins,cax=axins_cbar)
    # # norm=matplotlib.colors.LogNorm(vmin=color_values.min(),vmax=color_values.max()), s=100)
    #
    # plt.ylabel("Stochastic performance on Val set")
    #
    # cbar = plt.colorbar(sc, label="Difference with deterministic on test set")
    # cbar.ax.tick_params(labelsize=fs)
    # plt.tick_params(axis="y", labelsize=fs)
    # plt.tick_params(axis="x", labelsize=fs)
    # if cfg.dataset == "cifar10":
    #     plt.xlabel("")
    # else:
    #     plt.xlabel("Pruning rate", fontsize=fs)
    # # plt.ylabel("$\sigma$", fontsize=fs * 0.8)
    # # plt.legend()
    # plt.savefig(
    #     "figures/pareto_fronts/pareto_front_v3_{}_{}_{}_{}_{}.pdf".format(
    #         cfg.architecture, cfg.dataset, sampler, function_string,
    #         one_batch_string), bbox_inches="tight")
    # plt.close()
    #
    # fig, axs = plt.subplots(1, 1, figsize=fig_size, layout="compressed")
    # plt.title("{}".format(cfg.dataset.upper()))
    # cmap = plt.cm.get_cmap('magma')
    # p_lees_than_0 = p[p["Difference with deterministic"] < 0]
    # p_more_than_0 = p[p["Difference with deterministic"] > 0]
    #
    # sc_less_than_0 = axs.scatter(x=p_lees_than_0["Gradient Flow On Val post"],
    #                              y=p_lees_than_0["Stochastic Performance"],
    #                              facecolors='none', edgecolors='k', s=100)
    #
    # A = 15000
    # sizes = p_more_than_0["Sigma"] * A
    # color_values = p_more_than_0["Difference with deterministic"]
    #
    # sc = axs.scatter(x=p_more_than_0["Gradient Flow On Val post"], y=p_more_than_0["Stochastic Performance"],
    #                  c=color_values, cmap=cmap,  # s=sizes*np.log(sizes),
    #                  s=100,
    #                  norm=matplotlib.colors.PowerNorm(gamma=0.8))
    # # axs_y = axs.twinx()
    # # sc2 = axs_y.scatter(y=p_more_than_0["Gradient Flow On Val"], x=p_more_than_0["Stochastic performance"],
    # #                     c=color_values, cmap=cmap,#s=sizes*np.log(sizes),
    # #                     norm=matplotlib.colors.PowerNorm(gamma=1.2))
    # # sc_less_than_0_2 = axs_y.scatter(y=p_lees_than_0["Gradient Flow On Val"], x=p_lees_than_0["Stochastic performance"],
    # #                                  facecolors='none', edgecolors='k', s=15)
    #
    # # sc = plt.scatter(y=p_more_than_0["Sigma"], x=p_more_than_0["Pruning rate"],
    # #                  c=color_values, cmap=cmap,#s=sizes*np.log(sizes),
    # #                  norm=matplotlib.colors.PowerNorm(gamma=1.2))
    #
    # # axins2 = inset_axes(axs, width="40%", height="40%", loc="lower left")
    #
    # # axins = axs.inset_axes([0.13, 0.12, 0.35, 0.5])
    # #
    # # axins.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    # #
    # # axins_cbar = axs.inset_axes([0.49, 0.12, 0.02, 0.5])
    # #
    # # scins = axins.scatter(y=p_more_than_0["Sigma"], x=p_more_than_0["Pruning rate"],
    # #                       c=color_values, cmap=cmap,#s=sizes*np.log(sizes),
    # #                       norm=matplotlib.colors.PowerNorm(gamma=1.2))
    # # scins = axins.scatter(y=p_lees_than_0 ["Sigma"], x=p_lees_than_0["Pruning rate"],
    # #                       facecolors="none",edgocolor="k",) #s=sizes*np.log(sizes),)
    # # axins.tick_params(axis='both', which='major', labelsize=fs*0.8)
    # # for axis in ['top', 'bottom', 'left', 'right']:
    # #     axins.spines[axis].set_linewidth(1)
    # #     axins.spines[axis].set_color('gray')
    # #
    # # axins.set_ylabel("$\sigma$",fontsize=20)
    # # axins.set_xlabel("$\sigma$",fontsize=20)
    #
    # # mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec='gray', lw=1)
    # # fig.colorbar(scins,cax=axins_cbar)
    # # norm=matplotlib.colors.LogNorm(vmin=color_values.min(),vmax=color_values.max()), s=100)
    #
    # plt.ylabel("Stochastic performance on Val set")
    #
    # cbar = plt.colorbar(sc, label="Difference with deterministic on test set")
    #
    # cbar.ax.tick_params(labelsize=fs)
    #
    # plt.tick_params(axis="y", labelsize=fs)
    #
    # plt.tick_params(axis="x", labelsize=fs)
    #
    # if cfg.dataset == "cifar10":
    #     plt.xlabel("")
    # else:
    #     plt.xlabel("Gradient Flow on Val set", fontsize=fs)
    # # plt.ylabel("$\sigma$", fontsize=fs * 0.8)
    # # plt.legend()
    # plt.savefig(
    #     "figures/pareto_fronts/pareto_front_with_GF_{}_{}_{}_{}_{}.pdf".format(
    #         cfg.architecture, cfg.dataset, sampler, function_string,
    #         one_batch_string), bbox_inches="tight")
    # plt.close()

    #################### NOw we compare wit the deteminstic for the test set ##############################################

    # if functions == 1:
    #     g = optuna.visualization.plot_pareto_front(study, target_names=["Stochastic Performance", "Differce with Det."])
    #     g.update_layout(
    #         title=dict(text="{} {} {}".format(cfg.architecture, cfg.dataset,sampler), font=dict(size=20), automargin=True, yref='paper')
    #
    #     )
    #     g.show()
    # if functions == 2:
    #     g = optuna.visualization.plot_pareto_front(study, target_names=["Stochastic Performance", "Pruning rate"])
    #     g.update_layout(
    #         title=dict(text="{} {} {}".format(cfg.architecture, cfg.dataset,sampler), font=dict(size=20), automargin=True, yref='paper')
    #
    #     )
    #     g.show()
    #
    # net = get_model(cfg)
    # train, val, testloader = get_datasets(cfg)
    #
    # dense_performance = test(net, use_cuda=True, testloader=testloader, verbose=0)

    ######################### testing one net on the test set #########################################################

    # pruned_model = copy.deepcopy(net)
    # cfg.amount = best_pruning_rate
    # prune_function(pruned_model, cfg)
    # remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # # Add small noise just to get tiny variations of the deterministic ase
    # det_performance = test(pruned_model, use_cuda=True, testloader=testloader, verbose=0)
    # det_performance_val = test(pruned_model, use_cuda=True, testloader=val, verbose=0)
    #
    # names, weights = zip(*get_layer_dict(net))
    # number_of_layers = len(names)
    # sigma_per_layer = dict(zip(names, [best_sigma] * number_of_layers))
    # print("Deterministic performance on test set = {}".format(det_performance))
    # print("Deterministic performance on val set = {}".format(det_performance_val))
    # stochastic_performance = []
    #
    # performance_of_models = []
    # performance_of_models_val = []
    #
    # for individual_index in range(10):
    #     ############### Here I ask for pr and for sigma ###################################
    #
    #     current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
    #     # Here it needs to be the copy just in case the other trials make reference to the same object so it does not interfere
    #     prune_function(current_model, cfg)
    #
    #     remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
    #     stochastic_performance = test(current_model, use_cuda=True, testloader=testloader, verbose=0)
    #     stochastic_performance_val = test(current_model, use_cuda=True, testloader=val, verbose=0)
    #
    #     performance_of_models_val.append(stochastic_performance_val)
    #     # Dense stochastic performance
    #     performance_of_models.append(stochastic_performance)

    #
    # performance_of_models = np.array(performance_of_models)
    # median = np.median(performance_of_models)
    # print("Median accuracy of population: {}".format(median))
    # fitness_function_median = objective_function(median,det_performance, best_pruning_rate)
    # print("Fintness function of the median on test set: {}".format(fitness_function_median))
    # ##########  val functions###########################
    # performance_of_models_val= np.array(performance_of_models_val)
    # median = np.median(performance_of_models_val)
    # print("Median accuracy of population valset: {}".format(median))
    # fitness_function_median = objective_function(median,det_performance_val, best_pruning_rate)
    # print("Fintness function of the median on val set: {}".format(fitness_function_median))


    # Here is where I transfer the mask from the pruned stochastic model to the
    # original weights and put it in the ranking
    # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)


    # fig1 = optuna.visualization.plot_optimization_history(study)
    # fig2 = optuna.plot_intermediate_values(study)
    # fig3 = optuna.plot_param_importances(study)
    # fig4 = optuna.contour_plot(study, params=["sigma_add", "sigma_mul"])
    #
    # fig1.savefig("data/figures/opt_history.png")
    # fig2.savefig("data/figures/intermediate_values.png")
    # fig3.savefig("data/figures/para_importances.png")
    # fig4.savefig("data/figures/contour_plot.png")


def prune_function(net, cfg, pr_per_layer=None, dataloader=None):
    target_sparsity = cfg.amount
    if cfg.pruner == "global":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    if cfg.pruner == "manual":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner="manual", pr_per_layer=pr_per_layer)

        individual_prs_per_layer = prune_with_rate(net, target_sparsity,
                                                   exclude_layers=cfg.exclude_layers, type="layer-wise",
                                                   pruner="lamp", return_pr_per_layer=True)
        if cfg.use_wandb:
            log_dict = {}
            for name, elem in individual_prs_per_layer.items():
                log_dict["individual_{}_pr".format(name)] = elem
            wandb.log(log_dict)
    if cfg.pruner == "lamp":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)

    if cfg.pruner == "random":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="random",
                        pr_per_layer=pr_per_layer)
    if cfg.pruner == "grasp":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="graps", dataLoader=dataloader)


def prune_with_rate(net: torch.nn.Module, amount: typing.Union[int, float], pruner: str = "erk",
                    type: str = "global",
                    criterion:
                    str = "l1", exclude_layers: list = [], pr_per_layer: dict = {}, return_pr_per_layer: bool = False,
                    is_stochastic: bool = False, noise_type: str = "", noise_amplitude=0, dataLoader=None,
                    input_shape=None, num_classes=10):
    if type == "global":
        # print("Exclude layers in prun_with_rate:{}".format(exclude_layers))
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        # print("Length of weigths to prune:{}".format(len(weights))
        #       )
        if criterion == "l1":
            prune.global_unstructured(
                weights,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        if criterion == "l2":
            prune.global_unstructured(
                weights,
                pruning_method=prune.LnStructured,
                amount=amount,
                n=2
            )
    elif type == "layer-wise":
        from layer_adaptive_sparsity.tools.pruners import weight_pruner_loader
        if pruner == "lamp":
            pruner = weight_pruner_loader(pruner)
            if return_pr_per_layer:
                return pruner(model=net, amount=amount, exclude_layers=exclude_layers,
                              return_amounts=return_pr_per_layer)
            else:
                pruner(model=net, amount=amount, exclude_layers=exclude_layers, is_stochastic=is_stochastic,
                       noise_type=noise_type, noise_amplitude=noise_amplitude)
        if pruner == "erk":
            pruner = weight_pruner_loader(pruner)
            pruner(model=net, amount=amount, exclude_layers=exclude_layers)
            # _, amount_per_layer, _, _ = erdos_renyi_per_layer_pruning_rate(model=net, cfg=cfg)
            # names, weights = zip(*get_layer_dict(net))
            # for name, module in net.named_modules():
            #     if name in exclude_layers or name not in names:
            #         continue
            #     else:
            #         prune.l1_unstructured(module, name="weight", amount=float(amount_per_layer[name]))
        if pruner == "manual":
            for name, module in net.named_modules():
                with torch.no_grad():
                    if name in exclude_layers or not is_prunable_module(module):
                        continue
                    else:
                        prune.l1_unstructured(module, name="weight", amount=float(pr_per_layer[name]))
    elif type == "random":
        # weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        if criterion == "l1":
            #
            # prune.random_unstructured(
            #     weights,
            #     # pruning_method=prune.L1Unstructured,
            #     amount=amount
            # )
            for name, module in net.named_modules():
                with torch.no_grad():
                    if name in exclude_layers or not is_prunable_module(module):
                        continue
                    else:
                        prune.random_unstructured(module, name="weight", amount=float(pr_per_layer[name]))
    elif type == "grasp":
        from GRASP.pruner.GraSP import GraSP
        weight_selection_function = partial(weights_to_prune, exclude_layer_list=exclude_layers)
        module_filter_function = partial(filter_modules, exclude_layer_list=exclude_layers)
        mask: dict[torch.Tensor] = GraSP(net, amount, reinit=False, train_dataloader=dataLoader, device=device,
                                         weight_function=weight_selection_function,
                                         filter_function=module_filter_function, num_classes=num_classes)
        apply_mask(net, mask_dict=mask)
    elif type == "synflow":
        from synflow_snip_graps.pruning_method.Synflow import Synflow
        weight_selection_function = partial(weights_to_prune, exclude_layer_list=exclude_layers)
        # module_filter_function = partial(filter_modules,exclude_layer_list=exclude_layers)
        pruner = Synflow(net, torch.device(device), input_shape=[32, 32, 3], dataloader=dataLoader, criterion="l1",
                         weights_function=weight_selection_function)
        pruner.prune(amount)
        pass

    else:
        raise NotImplementedError("Not implemented for type {}".format(type))


######################################################################################################
def get_cifar_datasets(cfg: omegaconf.DictConfig):
    if cfg.dataset == "cifar10":
        data_path = cfg.get("data_folder") or "datasets"

        if cfg.pad:

            pad_to_use = cfg.input_resolution - 32

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=pad_to_use, padding_mode="edge"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([transforms.Pad(pad_to_use, padding_mode="edge"),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010)),
                                                 ])
        else:
            transform_train = transforms.Compose([transforms.Resize(cfg.input_resolution, antialias=True),
                                                  transforms.RandomCrop(cfg.input_resolution, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010)),
                                                  ])

            transform_test = transforms.Compose([transforms.Resize(cfg.input_resolution),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010)),
                                                 ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train)

        cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])
        trainloader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
                                                 num_workers=cfg.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=cfg.num_workers)
        return trainloader, val_loader, testloader
    if cfg.dataset == "cifar100":
        data_path = cfg.get("data_folder") or "datasets"

        # transform_train = transforms.Compose(
        #     [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        #      transforms.Normalize( (0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))])
        # transform_test = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize( (0.50707516, 0.48654887, 0.44091784),
        #                                                           (0.26733429, 0.25643846, 0.27615047))])

        if cfg.pad:

            pad_to_use = cfg.input_resolution - 32

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=pad_to_use, padding_mode="edge"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(),
            ])

            transform_test = transforms.Compose([transforms.Pad(pad_to_use, padding_mode="edge"),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.50707516, 0.48654887, 0.44091784),
                                                                      (0.26733429, 0.25643846, 0.27615047)),
                                                 ])
        else:
            transform_train = transforms.Compose([transforms.Resize(cfg.input_resolution, antialias=True),
                                                  transforms.RandomCrop(cfg.input_resolution, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.50707516, 0.48654887, 0.44091784),
                                                                       (0.26733429, 0.25643846, 0.27615047)),
                                                  ])

            transform_test = transforms.Compose([transforms.Resize(cfg.input_resolution),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.50707516, 0.48654887, 0.44091784),
                                                                      (0.26733429, 0.25643846, 0.27615047)),
                                                 ])

        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

        trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=cfg.batch_size, shuffle=True,
                                                  num_workers=cfg.num_workers)
        val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
                                                 num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)
        return trainloader, val_loader, testloader


def get_datasets(cfg: omegaconf.DictConfig):
    if "cifar" in cfg.dataset:
        return get_cifar_datasets(cfg)
    if "mnist" == cfg.dataset:
        data_path = cfg.get("data_folder") or "datasets"

        transfos = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.MNIST(root=data_path,
                                              train=True,
                                              transform=transfos,
                                              download=True)

        testset = torchvision.datasets.MNIST(root=data_path,
                                             train=False,
                                             transform=transfos
                                             )

        MNIST_train, MNIST_val = random_split(trainset, [len(trainset) - 5000, 5000])

        trainloader = torch.utils.data.DataLoader(
            MNIST_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        valloader = torch.utils.data.DataLoader(
            MNIST_val, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

        # testset = torchvision.datasets.CIFAR10(
        #     root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        return trainloader, valloader, testloader

    if 'imagenet' == cfg.dataset:
        # cfg.dataset="cifar10"
        # return get_cifar_datasets(cfg)
        # Excerpt take from https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126
        # Data loading code

        data_path = cfg.get("data_folder") or "datasets"
        traindir = data_path + '/imagenet/' + 'train'
        testdir = data_path + '/imagenet/' + 'val'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        whole_train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        print(f"Length of dataset: {len(whole_train_dataset)}")

        train_dataset, val_dataset = torch.utils.data.random_split(whole_train_dataset, [1231167, 50000])

        full_test_dataset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        big_test, small_test = torch.utils.data.random_split(full_test_dataset, [len(full_test_dataset) - 10000, 10000])

        # This code is to transform it into the "fast" format of ffcv

        # my_dataset = val_dataset
        # write_path = data_path + "imagenet/valSplit_dataset.beton"

        # For the validation set that I use to recover accuracy

        # # Pass a type for each data field
        # writer = DatasetWriter(write_path, {
        #     # Tune options to optimize dataset size, throughput at train-time
        #     'image': RGBImageField(
        #         max_resolution=256,
        #         jpeg_quality=90
        #     ),
        #     'label': IntField()
        # })
        # # Write dataset
        # writer.from_indexed_dataset(my_dataset)

        # For the validation set that I use to recover accuracy

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, sampler=None)
        if cfg.length_test == "small":
            test_loader = torch.utils.data.DataLoader(
                small_test,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)
        if cfg.length_test == "big":
            test_loader = torch.utils.data.DataLoader(
                big_test,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)
        if cfg.length_test == "whole":
            test_loader = torch.utils.data.DataLoader(
                full_test_dataset,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader

    if 'small_imagenet' == cfg.dataset:

        data_path = cfg.get("data_folder") or "datasets"
        from test_imagenet import load_small_imagenet
        trainloader, valloader, testloader = load_small_imagenet(
            {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
             "num_workers": cfg.num_workers, "batch_size": cfg.batch_size, "resolution": cfg.input_resolution})
        return trainloader, valloader, testloader

    if 'tiny_imagenet' == cfg.dataset:

        from test_imagenet import load_tiny_imagenet

        data_path = cfg.get("data_folder") or "datasets"
        traindir = data_path + '/tiny_imagenet_200/' + 'train'
        testdir = data_path + '/tiny_imagenet_200/' + 'val'
        cfg.traindir = traindir
        cfg.valdir = testdir
        return load_tiny_imagenet(dict(cfg))


def train(model: nn.Module, train_loader, val_loader, save_name, epochs, learning_rate, is_cyclic=False,
          cosine_schedule=False, lr_peak_epoch=5, weight_decay=1e-4, momentum=0.9):
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                                nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import ExponentialLR
    iters_per_epoch = len(train_loader)
    lr_schedule = np.interp(np.arange((epochs + 1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    if is_cyclic:
        # lr_scheduler = LambdaLR(optimizer, lr_schedule.__getitem__)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs,
                                                           steps_per_epoch=len(train_loader))
    else:
        lr_scheduler = ExponentialLR(optimizer, gamma=0.90)

    trainer = create_supervised_trainer(model, optimizer, criterion, device="cuda")
    val_metrics = {
        "accuracy": igm.Accuracy(),
        "nll": igm.Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device="cuda")

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    print("\nFine tuning has began\n")

    # Setup engine &  logger
    def setup_logger(logger):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, TerminateOnNan

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Store the best model
    def default_score_fn(engine):
        score = engine.state.metrics['accuracy']
        return score

    # Force filename to model.pt to ease the rerun of the notebook
    disk_saver = DiskSaver(dirname="trained_models", require_empty=False)
    best_model_handler = Checkpoint(to_save={f'{save_name}': model},
                                    save_handler=disk_saver,
                                    filename_pattern="{name}.{ext}",
                                    n_saved=1)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    # Add early stopping
    es_patience = 10
    es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)
    setup_logger(es_handler.logger)

    # Clear cuda cache between training/testing
    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())
    trainer.run(train_loader, max_epochs=epochs)


###############################  Channel inspection
# ############################################


############################### Experiments 25 of October # ############################################################


def get_model(cfg: omegaconf.DictConfig):
    net = None
    if cfg.architecture == "resnet18":
        if not cfg.solution:
            if "csgmcmc" == cfg.model_type:
                net = ResNet18()
                return net
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet18
                if cfg.dataset == "cifar10":
                    net = ResNet18()
                if cfg.dataset == "cifar100":
                    net = ResNet18(num_classes=100)
                if cfg.dataset == "mnist":
                    net = ResNet18()
                if cfg.dataset == "imagenet":
                    net = ResNet18(num_classes=1000)
                return net
            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    from torchvision import resnet18
                    net = resnet18()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 100)
                    net.load_state_dict(cfg, solution)
                if cfg.dataset == "cifar10":
                    from torchvision.models import resnet18
                    net = resnet18()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 10)

                    # temp_dict = torch.load(cfg.solution)["net"]
                    # real_dict = {}
                    # for k, item in temp_dict.items():
                    #     if k.startswith('module'):
                    #         new_key = k.replace("module.", "")
                    #         real_dict[new_key] = item
                    # net.load_state_dict(real_dict)

                if cfg.dataset == "imagenet":
                    from torchvision.models import resnet18

                    net = resnet18()
                    temp_dict = torch.load(cfg.solution)
                    net.load_state_dict(temp_dict)

                    # Using pretrained weights:
                    # net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    # net = resnet18(weights="IMAGENET1K_V1")
                return net
        else:
            if "csgmcmc" == cfg.model_type:
                net = ResNet18()
                load_model(net, cfg.solution)
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet18
                if cfg.dataset == "cifar10":
                    net = ResNet18()
                if cfg.dataset == "cifar100":
                    net = ResNet18(num_classes=100)
                if cfg.dataset == "mnist":
                    net = ResNet18()
                if cfg.dataset == "imagenet":
                    net = ResNet18(num_classes=1000)

                load_model(net, cfg.solution)

            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    from torchvision import resnet18
                    net = resnet18()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 100)
                    net.load_state_dict(cfg, solution)
                if cfg.dataset == "cifar10":
                    from torchvision.models import resnet18
                    net = resnet18()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 10)

                    temp_dict = torch.load(cfg.solution)["net"]
                    real_dict = {}
                    for k, item in temp_dict.items():
                        if k.startswith('module'):
                            new_key = k.replace("module.", "")
                            real_dict[new_key] = item
                    net.load_state_dict(real_dict)

                if cfg.dataset == "imagenet":
                    from torchvision.models import resnet18

                    net = resnet18()
                    temp_dict = torch.load(cfg.solution)
                    net.load_state_dict(temp_dict)

                    # Using pretrained weights:
                    # net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    # net = resnet18(weights="IMAGENET1K_V1")

            return net

    if cfg.architecture == "vgg19" or cfg.architecture == "VGG19":
        if not cfg.solution:
            if "csgmcmc" == cfg.model_type:
                net = VGG(cfg.architecture)
                return net
            if "alternative" == cfg.model_type:
                from alternate_models.vgg import VGG
                if cfg.dataset == "cifar100":
                    net = VGG("VGG19", num_classes=100)
                if cfg.dataset == "cifar10":
                    net = VGG("VGG19")
                if cfg.dataset == "imagenet":
                    net = VGG("VGG19", num_classes=1000)
                return net
        else:
            if "csgmcmc" == cfg.model_type:
                net = VGG("VGG19")
                load_model(net, cfg.solution)
            if "alternative" == cfg.model_type:
                from alternate_models.vgg import VGG
                if cfg.dataset == "cifar100":
                    net = VGG("VGG19", num_classes=100)
                if cfg.dataset == "cifar10":
                    net = VGG("VGG19")
                load_model(net, cfg.solution)
            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    net = torch.hub.load("chenyaofo/pytorch-cifar-models", cfg.solution, pretrained=True)
                if cfg.dataset == "cifar10":
                    net = torch.hub.load("chenyaofo/pytorch-cifar-models", cfg.solution, pretrained=True)

            return net

    if cfg.architecture == "resnet50":
        if not cfg.solution:
            if "csgmcmc" == cfg.model_type:
                net = ResNet50()
                return net
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet50
                if cfg.dataset == "cifar10":
                    net = ResNet50()
                if cfg.dataset == "cifar100":
                    net = ResNet50(num_classes=100)
                if cfg.dataset == "mnist":
                    net = ResNet50()
                if cfg.dataset == "imagenet":
                    net = ResNet50(num_classes=1000)
                return net
            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    from torchvision import resnet50
                    net = resnet50()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 100)
                    net.load_state_dict(cfg, solution)
                if cfg.dataset == "cifar10":
                    from torchvision.models import resnet50
                    net = resnet50()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 10)
                    #
                    # temp_dict = torch.load(cfg.solution)["net"]
                    # real_dict = {}
                    # for k, item in temp_dict.items():
                    #     if k.startswith('module'):
                    #         new_key = k.replace("module.", "")
                    #         real_dict[new_key] = item
                    # net.load_state_dict(real_dict)

                if cfg.dataset == "imagenet":
                    from torchvision.models import resnet50

                    net = resnet50()
                    # temp_dict = torch.load(cfg.solution)
                    # net.load_state_dict(temp_dict)

                    # Using pretrained weights:
                    # net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    # net = resnet18(weights="IMAGENET1K_V1")
                return net
        else:
            if "csgmcmc" == cfg.model_type:
                net = ResNet50()
                load_model(net, cfg.solution)
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet50
                if cfg.dataset == "cifar10":
                    net = ResNet50()
                if cfg.dataset == "cifar100":
                    net = ResNet50(num_classes=100)
                if cfg.dataset == "mnist":
                    net = ResNet50()
                load_model(net, cfg.solution)
            if "hub" == cfg.model_type:
                if cfg.dataset == "cifar100":
                    net = torch.hub.load("chenyaofo/pytorch-cifar-models", cfg.solution, pretrained=True)
                if cfg.dataset == "cifar10":
                    # net = torch.hub.load("chenyaofo/pytorch-cifar-models", cfg.solution, pretrained=True)
                    from torchvision.models import resnet50
                    net = resnet50()
                    in_features = net.fc.in_features
                    net.fc = nn.Linear(in_features, 10)

                    temp_dict = torch.load(cfg.solution)["net"]
                    real_dict = {}
                    for k, item in temp_dict.items():
                        if k.startswith('module'):
                            new_key = k.replace("module.", "")
                            real_dict[new_key] = item
                    net.load_state_dict(real_dict)

                if cfg.dataset == "imagenet":
                    from torchvision.models import resnet50
                    # Using pretrained weights:
                    net = resnet50(weights="IMAGENET1K_V1")

            return net

    else:
        raise NotImplementedError("Not implemented for architecture:{}".format(cfg.architecture))


######################################## Check functions ##########################################################
############################# Stochastic pruning with sigma optimization ###########################################
############################# Ablation experiments #####################################################################


def run_fine_tune_experiment(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    post_pruning_noise_string = "_post_training_noise" if bool(
        cfg.noise_after_pruning) * cfg.measure_gradient_flow else ""

    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"deterministic_fine_tune_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}{post_pruning_noise_string}",
            reinit=True,
        )
    pruned_model = get_model(cfg)
    dense_model = get_model(cfg)
    pruned_model.to(device)
    dense_model.to(device)
    print(f"CFG pruner {cfg.pruner}")
    if cfg.pruner == "global":
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
        remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    if cfg.pruner == "grasp":
        print("I entered to GRASP")
        num_classes = 10 if cfg.dataset == "cifar10" else 100
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type=cfg.pruner,
                        dataLoader=valloader, num_classes=num_classes)
    if cfg.pruner == "synflow":
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type=cfg.pruner,
                        dataLoader=valloader)
        remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # else:
    #     print("I entered to the ELSE ")
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
    #                     pruner=cfg.pruner)
    #     remove_reparametrization(pruned_model, exclude_layer_list=cfg.exclude_layers)
    # Add small noise just to get tiny variations of the deterministic case
    initial_performance = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
    print("Original version performance: {}".format(initial_performance))
    if cfg.noise_after_pruning and cfg.measure_gradient_flow:
        mask_dict = get_mask(pruned_model)
        names, weights = zip(*get_layer_dict(pruned_model))
        sigma_per_layer = dict(zip(names, [cfg.noise_after_pruning] * len(names)))
        p2_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        print("p2_model version 1 performance: {}".format(initial_performance))
        apply_mask(p2_model, mask_dict)
        initial_performance = test(p2_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
        print("p2_model version 2 performance: {} with sparsity {}".format(initial_performance, sparsity(p2_model)))
        pruned_model = p2_model

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
    # mask_dict = get_mask(pruned_model)
    # p2_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
    # print("p2_model version 1 performance: {}".format(initial_performance))
    # apply_mask(p2_model,mask_dict)
    # initial_performance = test(p2_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
    # print("p2_model version 2 performance: {}".format(initial_performance))
    # return
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": initial_performance, "initial_accuracy": initial_performance})
    filepath_GF_measure = ""
    if cfg.measure_gradient_flow:
        identifier = f"{time.time():14.2f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/{}/deterministic_LAMP{}/{}/{}/sigma0.0/pr{}/{}/".format(
                cfg.dataset, f"_{cfg.name}" if cfg.name else "",
                cfg.architecture,
                cfg.model_type,
                # cfg.sigma,
                cfg.amount,
                identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
            # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/{}/deterministic_GLOBAL{}/{}/{}/sigma0.0/pr{}/{}/".format(
                cfg.dataset, cfg.pruner.upper(), f"_{cfg.name}" if cfg.name else "",
                cfg.architecture,
                cfg.model_type,
                # cfg.sigma,
                cfg.amount,
                identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)

            # else:

            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        else:

            filepath_GF_measure += "gradient_flow_data/{}/deterministic_{}{}/{}/{}/sigma0.0/pr{}/{}/".format(
                cfg.dataset, cfg.pruner.upper(), f"_{cfg.name}" if cfg.name else "",
                cfg.architecture,
                cfg.model_type,
                # cfg.sigma,
                cfg.amount,
                identifier)
            path: Path = Path(filepath_GF_measure)

            if not path.is_dir():
                path.mkdir(parents=True)

            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"

    file_path = None
    weights_path = ""
    gradient_flow_file_prefix = filepath_GF_measure
    weights_file_path = ""
    if gradient_flow_file_prefix != "":
        weights_file_path = "GF_data/" + gradient_flow_file_prefix + "weights/"
        weights_path = Path(weights_file_path)
        weights_path.mkdir(parents=True)
        state_dict = dense_model.state_dict()
        temp_name = weights_path / "dense.pth"
        torch.save(state_dict, temp_name)

    restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                       use_wandb=cfg.use_wandb, epochs=cfg.epochs, exclude_layers=cfg.exclude_layers,
                                       fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                       fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                       cfg=cfg,
                                       gradient_flow_file_prefix=filepath_GF_measure)

    if cfg.use_wandb:
        wandb.join()


def fine_tune_after_stochastic_pruning_experiment(cfg: omegaconf.DictConfig, print_exclude_layers=True):
    trainloader, valloader, testloader = get_datasets(cfg)
    # batch_shape = next(itertrainloader).shape
    batch_shape = (32, 32, 3)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    ################################## WANDB configuration ############################################
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    one_batch_string = "_one_batch_per_generation" if cfg.one_batch else "_whole_dataset_per_generation"
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"fine_tune_base_stochastic_pruning_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}"
                 f"{non_zero_string}{one_batch_string}",
            notes="This run is to see if gradient clipping is hindering stochastic pruning",
            reinit=True,
        )
    ################################## Gradient flow measure###############test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)#############################
    filepath_GF_measure = ""
    if cfg.measure_gradient_flow:

        identifier = f"{time.time():14.5f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/{}/stochastic_LAMP{}/{}/{}/sigma{}/pr{}/{}/".format(cfg.dataset,
                                                                                                           f"_{cfg.name}" if cfg.name else "",
                                                                                                           cfg.architecture,
                                                                                                           cfg.model_type,
                                                                                                           cfg.sigma,
                                                                                                           cfg.amount,
                                                                                                           identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
            # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/{}/stochastic_GLOBAL{}/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, f"_{cfg.name}" if cfg.name else "",
                cfg.architecture,
                cfg.model_type,
                cfg.sigma,
                cfg.amount,
                identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        else:
            filepath_GF_measure += "gradient_flow_data/{}/stochastic_{}{}/{}/{}/sigma{}/pr{}/{}/".format(
                cfg.dataset, cfg.pruner.upper(), f"_{cfg.name}" if cfg.name else "",
                cfg.architecture,
                cfg.model_type,
                cfg.sigma,
                cfg.amount,
                identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"

    pruned_model = get_model(cfg)
    best_model = None
    best_dense_model = None
    best_accuracy = -1
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    evaluation_set = valloader
    if cfg.one_batch:
        evaluation_set = [(data, y)]
    names, weights = zip(*get_layer_dict(pruned_model))
    sigma_per_layer = dict(zip(names, [cfg.sigma] * len(names)))

    pr_per_layer = prune_with_rate(copy.deepcopy(pruned_model), target_sparsity, exclude_layers=cfg.exclude_layers,
                                   type="layer-wise",
                                   pruner="lamp", return_pr_per_layer=True)
    if cfg.use_wandb:
        log_dict = {}
        for name, elem in pr_per_layer.items():
            log_dict["deterministic_{}_pr".format(name)] = elem
        wandb.log(log_dict)

    # Go over the population t

    for n in range(cfg.population):
        # current_model = get_noisy_sample(pruned_model, cfg)
        current_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        copy_of_dense_model = copy.deepcopy(current_model)

        if cfg.pruner == "global":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        if cfg.pruner == "manual":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="manual", pr_per_layer=pr_per_layer)

            # individual_prs_per_layer = prune_with_rate(, target_sparsity,
            #                                            exclude_layers=cfg.exclude_layers, type="layer-wise",
            #                                            pruner="lamp", return_pr_per_layer=True)
            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
            # if cfg.use_wandb:
            #     log_dict = {}
            #     for name, elem in individual_prs_per_layer.items():
            #         log_dict["individual_{}_pr".format(name)] = elem
            #     wandb.log(log_dict)
        if cfg.pruner == "lamp":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers,
                            type="layer-wise",
                            pruner=cfg.pruner)
            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        if cfg.pruner == "grasp":
            num_classes = 10 if cfg.dataset == "cifar10" else 100
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type=cfg.pruner,
                            dataLoader=valloader, num_classes=num_classes)
        if cfg.pruner == "synflow":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="synflow",
                            dataLoader=valloader, input_shape=batch_shape)
            remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)

        # prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                 pruner=cfg.pruner)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)

        if first_iter:
            _, unit_sparse_flops = flops(current_model, data)
            first_iter = 0
        noisy_sample_performance, individual_sparse_flops = test(current_model, use_cuda, evaluation_set, verbose=0,
                                                                 count_flops=True, batch_flops=unit_sparse_flops)
        check_for_layers_collapse(current_model)
        initial_flops += individual_sparse_flops
        if noisy_sample_performance > best_accuracy:
            best_accuracy = noisy_sample_performance
            best_model = current_model
            best_dense_model = copy_of_dense_model

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)

    file_path = None
    weights_file_path = ""
    gradient_flow_file_prefix = filepath_GF_measure
    if gradient_flow_file_prefix != "":
        weights_file_path = "GF_data/" + gradient_flow_file_prefix + "weights/"
        weights_path = Path(weights_file_path)
        weights_path.mkdir(parents=True)
        state_dict = best_dense_model.state_dict()
        temp_name = weights_path / "dense.pth"
        torch.save(state_dict, temp_name)
    initial_performance = test(best_model, use_cuda=use_cuda, testloader=valloader, verbose=1)

    end = time.time()
    initial_test_performance = test(best_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    total = time.time() - end
    print("Time for testing: {} s".format(total))

    # torch.save({"model_state":best_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_{cfg.pruner}_s{cfg.sigma}_pr{cfg.amount}.pth")

    # if cfg.pruner == "global":
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    #
    # if cfg.pruner == "lamp":
    #     prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers,
    #                     type="layer-wise",
    #                     pruner=cfg.pruner)
    #
    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)

    # perfor = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    # torch.save({"model_state":pruned_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_deterministic_{cfg.pruner}_pr{cfg.amount}.pth")

    if cfg.use_wandb:
        wandb.log({"val_set_accuracy": initial_performance, "sparse_flops": initial_flops, "initial_test_performance":
            initial_test_performance})

    restricted_fine_tune_measure_flops(best_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                       use_wandb=cfg.use_wandb, epochs=cfg.epochs, exclude_layers=cfg.exclude_layers,
                                       initial_flops=initial_flops,
                                       fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                       fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                       gradient_flow_file_prefix=filepath_GF_measure,
                                       cfg=cfg)




def one_shot_static_sigma_stochastic_pruning(cfg, eval_set="test", print_exclude_layers=True):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers" if print_exclude_layers else ""
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"one_shot_stochastic_pruning_static_sigma_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}",
            notes="This experiment is to test if iterative global stochastic pruning, compares to one-shot stochastic pruning",
            reinit=True,
        )

    pruned_model = get_model(cfg)
    best_model = None
    best_accuracy = 0
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    evaluation_set = None
    if cfg.one_batch:
        evaluation_set = [data]
    else:
        if eval_set == "test":
            evaluation_set = testloader
        if eval_set == "val":
            evaluation_set = valloader

    for n in range(cfg.population):

        noisy_sample = get_noisy_sample(pruned_model, cfg)

        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        # Dense stochastic performance
        if cfg.pruner == "global" :
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
        elif cfg.pruner== "grasp":
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="grasp")
        else:
            # This is for lamp
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)
        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)
        if first_iter:
            _, unit_sparse_flops = flops(noisy_sample, data)
            first_iter = 0

        noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, evaluation_set, verbose=0,
                                                                 count_flops=True, batch_flops=unit_sparse_flops)

        initial_flops += individual_sparse_flops
        if cfg.use_wandb:
            test_accuracy = test(noisy_sample, use_cuda, [get_random_batch(testloader)], verbose=0)
            log_dict = {"val_set_accuracy": noisy_sample_performance, "individual": n,
                        "sparse_flops": initial_flops,
                        "test_set_accuracy": test_accuracy
                        }
            wandb.log(log_dict)
        if noisy_sample_performance > best_accuracy:
            best_accuracy = noisy_sample_performance
            best_model = noisy_sample
            test_accuracy = test(best_model, use_cuda, [get_random_batch(testloader)], verbose=0)
            if cfg.use_wandb:
                log_dict = {"best_val_set_accuracy": best_accuracy, "individual": n,
                            "sparsity": sparsity(best_model),
                            "sparse_flops": initial_flops,
                            "test_set_accuracy": test_accuracy
                            }
                wandb.log(log_dict)


def experiment_selector(cfg: omegaconf.DictConfig, args, number_experiment: int = 1):
    # Experiment 10
    if number_experiment == 1:
        one_shot_static_sigma_stochastic_pruning(cfg, eval_set="val")
    # Experiment 11
    if number_experiment == 2:
        fine_tune_after_stochastic_pruning_experiment(cfg)
    #     Experiment 19
    if number_experiment == 3:
        run_pr_sigma_search_MOO_for_cfg(cfg, args)
    #     Experiment 20
    if number_experiment == 4:
        plot_pr_sigma_search_MOO_for_cfg(cfg, args)
    #     Experiment 21
    if number_experiment == 5:
        run_pr_sigma_fine_tuned_search_MOO_for_cfg(cfg, args)
    if number_experiment == 6:
        run_fine_tune_experiment(cfg)
    # if number_experiment == 22:
    #     fine_tune_after_stochatic_pruning_experiment(cfg)
    # if number_experiment == 13:



def LeMain(args):
    solution = ""
    exclude_layers = None
    if args["dataset"] == "cifar100":
        if args["modeltype"] == "alternative":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar100/resnet18_cifar100_traditional_train.pth"
                exclude_layers = ["conv1", "linear"]
            if args["architecture"] == "vgg19":
                solution = "trained_models/cifar100/vgg19_cifar100_traditional_train.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/cifar100/resnet50_cifar100.pth"
                exclude_layers = ["conv1", "linear"]
    if args["dataset"] == "cifar10":
        if args["modeltype"] == "alternative":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth"
                # solution = "trained_models/cifar10/resnet18_cifar10_normal_seed_2.pth"
                # solution = "trained_models/cifar10/resnet18_cifar10_normal_seed_3.pth"
                exclude_layers = ["conv1", "linear"]
            if args["architecture"] == "vgg19":
                solution = "trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/cifar10/resnet50_cifar10.pth"
                exclude_layers = ["conv1", "linear"]
        if args["modeltype"] == "hub":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth"
                exclude_layers = ["conv1", "fc"]
    if args["dataset"] == "imagenet":

        if args["modeltype"] == "hub":

            if args["architecture"] == "resnet18":
                solution = "trained_models/imagenet/resnet18_imagenet.pth"
                exclude_layers = ["conv1", "fc"]
                # exclude_layers = []
            if args["architecture"] == "VGG19":
                raise NotImplementedError("Not implemented")
                solution = "trained_models/cifar100/vgg19_cifar100_traditional_train.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/imagenet/resnet50_imagenet.pth"
                exclude_layers = ["conv1", "fc"]

    # `exclude_layers` is purely structural (depends on architecture/modeltype), so it is always
    # derived from the lookup above. The checkpoint path itself ("solution") is a run parameter:
    # use the one passed on the command line if given, otherwise fall back to the paper's default
    # checkpoint for this dataset/modeltype/architecture combination.
    if args.get("solution"):
        solution = args["solution"]

    cfg = omegaconf.DictConfig({
        "population": args["population"],
        "generations": 10,
        "epochs": args["epochs"],
        "short_epochs": 10,
        "architecture": args["architecture"],
        "solution": solution,
        "noise": "gaussian",
        "pruner": args["pruner"],
        "model_type": args["modeltype"],
        "fine_tune_exclude_layers": False,
        "fine_tune_non_zero_weights": True,
        "sampler": "tpe",
        "flop_limit": 0,
        "one_batch": args["one_batch"],
        "measure_gradient_flow": True,
        "full_fine_tune": False,
        "use_stochastic": True,
        "sigma": args["sigma"],
        "noise_after_pruning": 0,
        "amount": args["pruning_rate"],
        "dataset": args["dataset"],
        "batch_size": args["batch_size"],
        "num_workers": args["num_workers"],
        "save_model_path": "stochastic_pruning_models/",
        "save_data_path": "stochastic_pruning_data/",
        "gradient_cliping": True,
        "pad": False,
        "input_resolution": 32,
        "resize": False,
        "use_wandb": False,
        "name": args["name"],
        "data_folder": args.get("data_folder")
    })

    cfg.exclude_layers = exclude_layers

    experiment_selector(cfg, args, args["experiment"])


    # cfg2 = omegaconf.DictConfig({
    #     "sigma":0.0,
    #     "amount":0.9,
    #     "architecture":"VGG19",
    #     "model_type": "alternative",
    #     "solution":"trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth",
    #     "dataset":"cifar10",
    #     "set":"test"
    #
    # })

    # weights_analysis_per_weight(cfg1,cfg2)


def run_le_Main_with_external_parameters():
    ######  Para fine-tuning the modelos en general
    parser = argparse.ArgumentParser(description='Stochastic pruning experiments')
    parser.add_argument('-exp', '--experiment', type=int, default=15, help='Experiment number', required=True)
    parser.add_argument('-pop', '--population', type=int, default=1, help='Population', required=False)
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Epochs for fine tuning', required=False)
    parser.add_argument('-sig', '--sigma', type=float, default=0.005, help='Noise amplitude', required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch size', required=True)
    parser.add_argument('-pr', '--pruner', type=str, default="global", help='Type of prune', required=True)
    parser.add_argument('-dt', '--dataset', type=str, default="cifar10", help='Dataset for experiments', required=True)
    parser.add_argument('-ar', '--architecture', type=str, default="resnet18", help='Type of architecture',
                        required=True)
    parser.add_argument('-mt', '--modeltype', type=str, default="alternative",
                        help='The type of model (which model definition/declaration) to use in the architecture',
                        required=True)
    parser.add_argument('-pru', '--pruning_rate', type=float, default=0.9, help='percentage of weights to prune',
                        required=False)
    parser.add_argument('--name', type=str, default="",
                        help='Name for the file', required=False)
    parser.add_argument('-nw', '--num_workers', type=int, default=8, help='Number of workers', required=False)
    parser.add_argument('-ob', '--one_batch', type=bool, default=False, help='One batch in sigma pr optim',
                        required=False)

    #   ############ additional parameters #################################
    parser.add_argument('-so', '--solution', type=str, default=None,
                        help='Path to the pretrained dense model checkpoint (the "solution") to prune. '
                             'If omitted, falls back to the default checkpoint path used in the paper for the '
                             'given dataset/modeltype/architecture combination.',
                        required=False)
    parser.add_argument('-df', '--data_folder', type=str, default=None,
                        help='Root directory where the dataset is/will be downloaded. '
                             'If omitted, falls back to a hardcoded per-machine default path.',
                        required=False)
    # parser.add_argument('-gen', '--generation', type=int, default=10, help='Generations', required=False)

    ############ parameters needed by the Optuna pr/sigma search experiments (18, 19, 21) #################
    parser.add_argument('-sa', '--sampler', type=str, default="tpe",
                        help='Optuna sampler for the pr/sigma search [tpe, nsga, cmaes]', required=False)
    parser.add_argument('-ls', '--log_sigma', type=bool, default=False,
                        help='Sample sigma on a log scale during the pr/sigma search', required=False)
    parser.add_argument('-tr', '--trials', type=int, default=300,
                        help='Number of Optuna trials for the pr/sigma search', required=False)
    parser.add_argument('-fnc', '--functions', type=int, default=1,
                        help='Which fitness function(s) to use in the multi-objective pr/sigma search [1, 2]',
                        required=False)
    args = vars(parser.parse_args())

    LeMain(args)

if __name__ == '__main__':
    run_le_Main_with_external_parameters()
