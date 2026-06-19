#!/bin/bash
# Handler/launcher for the preliminary dense-model training step (train_CIFAR10.py).
# Submits one SLURM job per (model, dataset) combination via slurm_train_CIFAR10_run.sh.
# The resulting checkpoints are the "solution" files later consumed by main.py's
# stochastic pruning experiments (pass them via --solution, see slurm_SP_*_run.sh).

run_train_cifar10() {
model=$1
dataset=$2
epochs=$3
name=$4

sbatch --nodes=1 --time=24:00:00 --partition=gpu --mail-type=all --mail-user=you@example.com \
  --error="train_${name}_${model}_${dataset}.err" --gres=gpu:1 \
  --output="train_${name}_${model}_${dataset}.out" \
  --job-name="train_${name}_${model}_${dataset}" \
  slurm_train_CIFAR10_run.sh "${model}" "${dataset}" "${epochs}" 128 "0.1" 8 \
  "trained_models/${dataset}" "datasets" "${name}_${model}_${dataset}"
}

model_list=("resnet18" "resnet50" "vgg19")
dataset_list=("cifar10" "cifar100")

epochs=200
for model in "${model_list[@]}"; do
for dataset in "${dataset_list[@]}"; do
run_train_cifar10 "${model}" "${dataset}" "${epochs}" "dense_train"
done
done
