#!/bin/bash
# Handler/launcher for the Optuna multi-objective pr/sigma search (experiment 19 in
# main.py). Submits one SLURM job per (model, dataset) combination via
# slurm_SP_moo_search_run.sh. See the NOTE in that script: experiment 19's search loop is
# currently commented out in main.py and needs to be re-enabled before this produces results.

run_sp_moo_search() {
model=$1
dataset=$2
pruner=$3
sampler=$4
trials=$5
functions=$6
log_sigma=$7
name=$8

sbatch --nodes=1 --time=48:00:00 --partition=gpu --mail-type=all --mail-user=sclaam@leeds.ac.uk \
  --error="MOO_${name}_${model}_${dataset}_${pruner}_${sampler}_F${functions}.err" --gres=gpu:1 \
  --output="MOO_${name}_${model}_${dataset}_${pruner}_${sampler}_F${functions}.out" \
  --job-name="MOO_${name}_${model}_${dataset}_${pruner}_${sampler}_F${functions}" \
  slurm_SP_moo_search_run.sh "${model}" "${dataset}" "${pruner}" "${sampler}" "${trials}" "${functions}" "${log_sigma}" "${name}"
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
