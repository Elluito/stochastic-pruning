#!/bin/bash
# Handler/launcher for fine-tuning after stochastic pruning (experiment 11 in main.py),
# and its deterministic-pruning baseline (experiment 6). Submits one SLURM array job per
# (model, dataset, sigma, pruning_rate) combination via slurm_SP_fine_tuning_run.sh.

run_sp_fine_tuning() {
model=$1
dataset=$2
sigma=$3
pr=$4
pruner=$5
epochs=$6
name=$7
solution=$8

# Stochastic pruning + fine-tuning (experiment 11)
sbatch --nodes=1 --time=12:00:00 --array=1-5 --partition=gpu --mail-type=all --mail-user=sclaam@leeds.ac.uk \
  --error="SP_FT_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_sto.err" --gres=gpu:1 \
  --output="SP_FT_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_sto.out" \
  --job-name="SP_FT_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_sto" \
  slurm_SP_fine_tuning_run.sh 11 "${sigma}" "${pruner}" "${model}" "${dataset}" "${pr}" "alternative" "${epochs}" "${name}" "${solution}"

# Deterministic pruning + fine-tuning baseline (experiment 6)
sbatch --nodes=1 --time=12:00:00 --partition=gpu --mail-type=all --mail-user=sclaam@leeds.ac.uk \
  --error="DP_FT_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_det.err" --gres=gpu:1 \
  --output="DP_FT_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_det.out" \
  --job-name="DP_FT_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_det" \
  slurm_SP_fine_tuning_run.sh 6 "${sigma}" "${pruner}" "${model}" "${dataset}" "${pr}" "alternative" "${epochs}" "${name}" "${solution}"
}

# Table 1 parameters from the paper
model_list=("resnet18" "resnet18" "resnet50" "resnet50" "vgg19" "vgg19")
dataset_list=("cifar10" "cifar100" "cifar10" "cifar100" "cifar10" "cifar100")
sigma_list=("0.005" "0.003" "0.003" "0.001" "0.003" "0.001")
pruning_rate_list=("0.9" "0.9" "0.95" "0.85" "0.95" "0.8")

pruning_method="global"
epochs=200
max=${#model_list[@]}
for ((idxA=0; idxA<max; idxA++)); do
model="${model_list[$idxA]}"
dataset="${dataset_list[$idxA]}"
sigma="${sigma_list[$idxA]}"
pruning_rate="${pruning_rate_list[$idxA]}"

run_sp_fine_tuning "${model}" "${dataset}" "${sigma}" "${pruning_rate}" "${pruning_method}" "${epochs}" "FT_comparison_table_1" ""
done
