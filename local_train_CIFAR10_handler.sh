#!/bin/bash
# Local (non-SLURM) handler/launcher for the preliminary dense-model training step
# (train_CIFAR10.py). Runs each (model, dataset) combination sequentially via
# local_train_CIFAR10_run.sh instead of submitting SLURM jobs. The resulting checkpoints
# are the "solution" files later consumed by main.py's stochastic pruning experiments
# (pass them via --solution, see local_SP_*_run.sh).

run_train_cifar10() {
model=$1
dataset=$2
epochs=$3
name=$4

job="train_${name}_${model}_${dataset}"
echo "Running ${job}"
bash local_train_CIFAR10_run.sh "${model}" "${dataset}" "${epochs}" 128 "0.1" 8 \
  "trained_models/${dataset}" "datasets" "${name}_${model}_${dataset}" \
  > "${job}.out" 2> "${job}.err"
}

model_list=("resnet18" "resnet50" "vgg19")
dataset_list=("cifar10" "cifar100")

epochs=200
for model in "${model_list[@]}"; do
for dataset in "${dataset_list[@]}"; do
run_train_cifar10 "${model}" "${dataset}" "${epochs}" "dense_train"
done
done
