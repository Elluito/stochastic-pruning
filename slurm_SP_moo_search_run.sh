#!/bin/bash
# Worker script for SLURM (sbatch) jobs running experiment 19 from main.py:
# multi-objective Optuna search over (sigma, pruning_rate) for a fixed architecture/dataset/
# pruner config (run_pr_sigma_search_MOO_for_cfg). Results are stored in a sqlite study
# database (find_pr_sigma_database_MOO_*.dep) so the search can be resumed/extended.
#
# NOTE: as inherited from the original research code, the body of
# run_pr_sigma_search_MOO_for_cfg() in main.py has its optuna.create_study/study.optimize
# calls commented out (it currently only builds the sampler and returns). Uncomment that
# block in main.py before relying on this script for real results - see the sibling,
# fully-active experiment 21 (run_pr_sigma_fine_tuned_search_MOO_for_cfg) for the
# equivalent working pattern.
#
# Positional arguments:
#   $1  architecture [resnet18, resnet50, vgg19]
#   $2  dataset [cifar10, cifar100]
#   $3  pruner [global, lamp, erk, random, grasp, synflow, manual]
#   $4  sampler [tpe, nsga, cmaes]
#   $5  number of optuna trials
#   $6  functions: which fitness function(s) to use [1, 2]
#   $7  log_sigma: 1 to sample sigma on a log scale, 0 otherwise
#   $8  name (run identifier used in output file names)

# Activate your conda/venv environment before running this script (e.g. via
# `conda activate <env_name>`). Most clusters set LD_LIBRARY_PATH/PYTHONPATH
# automatically on activation; override them here only if your cluster requires it, e.g.:
# export LD_LIBRARY_PATH="/path/to/your/conda/envs/<env_name>/lib:$LD_LIBRARY_PATH"
# export PYTHONPATH="/path/to/your/conda/envs/<env_name>/lib/python3.9/site-packages"

# NOTE: argparse parses --log_sigma with type=bool, so ANY string (even "False") is truthy.
# The only reliable way to keep it False is to omit the flag entirely and rely on its default.
log_sigma_arg=""
if [ "$7" -eq 1 ]; then
  log_sigma_arg="--log_sigma True"
fi

python main.py -exp 19 -bs 128 --sigma 0.005 --pruner "$3" --architecture "$1" --dataset "$2" \
  --pruning_rate 0.9 --modeltype "alternative" --epochs 0 --name "$8" -pop 1 --num_workers 8 \
  --sampler "$4" --trials "$5" --functions "$6" ${log_sigma_arg}
