#!/bin/bash
# Local (non-SLURM) handler/launcher for the Optuna multi-objective pr/sigma search
# (experiment 3 in main.py). Runs each (model, dataset) combination sequentially via
# local_SP_moo_search_run.sh instead of submitting SLURM jobs.

run_sp_moo_search() {
model=$1
dataset=$2
pruner=$3
sampler=$4
trials=$5
functions=$6
log_sigma=$7
name=$8

job="MOO_${name}_${model}_${dataset}_${pruner}_${sampler}_F${functions}"
echo "Running ${job}"
bash local_SP_moo_search_run.sh "${model}" "${dataset}" "${pruner}" "${sampler}" "${trials}" "${functions}" "${log_sigma}" "${name}" \
  > "${job}.out" 2> "${job}.err"
}

model_list=("resnet18" "resnet18" "resnet50" "resnet50" "vgg19" "vgg19")
dataset_list=("cifar10" "cifar100" "cifar10" "cifar100" "cifar10" "cifar100")

pruner="global"
sampler="nsga"
trials=300
functions=2
log_sigma=0

max=${#model_list[@]}
for ((idxA=0; idxA<max; idxA++)); do
model="${model_list[$idxA]}"
dataset="${dataset_list[$idxA]}"

run_sp_moo_search "${model}" "${dataset}" "${pruner}" "${sampler}" "${trials}" "${functions}" "${log_sigma}" "MOO_search"
done
