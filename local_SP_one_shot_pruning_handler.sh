#!/bin/bash
# Local (non-SLURM) handler/launcher for one-shot stochastic pruning experiments
# (experiment 1 in main.py). Runs each (model, dataset, sigma, pruning_rate) combination
# sequentially via local_SP_one_shot_pruning_run.sh instead of submitting a SLURM array job.

run_sp_one_shot_pruning() {
model=$1
dataset=$2
sigma=$3
pr=$4
pruner=$5
name=$6
solution=$7

# Repeated 5 times like the SLURM --array=1-5 job.
for rep in 1 2 3 4 5; do
  job="SP_oneshot_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_${rep}"
  echo "Running ${job}"
  bash local_SP_one_shot_pruning_run.sh 1 128 "${sigma}" "${pruner}" "${model}" "${dataset}" "${pr}" "alternative" 0 "${name}" "${solution}" \
    > "${job}.out" 2> "${job}.err"
done
}

# Table 1 parameters from the paper
model_list=("resnet18" "resnet18" "resnet50" "resnet50" "vgg19" "vgg19")
dataset_list=("cifar10" "cifar100" "cifar10" "cifar100" "cifar10" "cifar100")
sigma_list=("0.005" "0.003" "0.003" "0.001" "0.003" "0.001")
pruning_rate_list=("0.9" "0.9" "0.95" "0.85" "0.95" "0.8")

pruning_method="global"
max=${#model_list[@]}
for ((idxA=0; idxA<max; idxA++)); do
model="${model_list[$idxA]}"
dataset="${dataset_list[$idxA]}"
sigma="${sigma_list[$idxA]}"
pruning_rate="${pruning_rate_list[$idxA]}"

run_sp_one_shot_pruning "${model}" "${dataset}" "${sigma}" "${pruning_rate}" "${pruning_method}" "one_shot_table_1" ""
done
