#!/bin/bash
# Local (non-SLURM) handler/launcher for fine-tuning after stochastic pruning (experiment 2
# in main.py), and its deterministic-pruning baseline (experiment 6). Runs each
# (model, dataset, sigma, pruning_rate) combination sequentially via local_SP_fine_tuning_run.sh
# instead of submitting SLURM array jobs.
#
# Logs go to <job_name>.out / <job_name>.err in the current directory, mirroring the
# SLURM version's --output/--error naming.

run_sp_fine_tuning() {
model=$1
dataset=$2
sigma=$3
pr=$4
pruner=$5
epochs=$6
name=$7
solution=$8

# Stochastic pruning + fine-tuning (experiment 2), repeated 5 times like the SLURM
# --array=1-5 job.
for rep in 1 2 3 4 5; do
  job="SP_FT_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_sto_${rep}"
  echo "Running ${job}"
  bash local_SP_fine_tuning_run.sh 2 "${sigma}" "${pruner}" "${model}" "${dataset}" "${pr}" "alternative" "${epochs}" "${name}" "${solution}" \
    > "${job}.out" 2> "${job}.err"
done

# Deterministic pruning + fine-tuning baseline (experiment 6)
job="DP_FT_${name}_${model}_${dataset}_sig_${sigma}_pr_${pr}_${pruner}_det"
echo "Running ${job}"
bash local_SP_fine_tuning_run.sh 6 "${sigma}" "${pruner}" "${model}" "${dataset}" "${pr}" "alternative" "${epochs}" "${name}" "${solution}" \
  > "${job}.out" 2> "${job}.err"
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
