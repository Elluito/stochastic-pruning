#!/bin/bash
# Local (non-SLURM) worker script for running experiment 2 from main.py:
# fine-tuning after stochastic pruning (fine_tune_after_stochastic_pruning_experiment).
# Also works for experiment 6 (deterministic pruning + fine-tuning), since main.py just
# dispatches on the experiment number $1.
#
# Run this from a shell where your python environment (conda/venv) is already activated.
#
# Positional arguments:
#   $1  experiment number (2 = fine-tune after stochastic pruning, 6 = deterministic)
#   $2  sigma (noise amplitude)
#   $3  pruner [global, lamp, erk, random, grasp, synflow, manual]
#   $4  architecture [resnet18, resnet50, vgg19]
#   $5  dataset [cifar10, cifar100]
#   $6  pruning_rate
#   $7  modeltype [alternative, hub] - use "alternative" for the alternate_models implementation
#   $8  epochs of fine-tuning
#   $9  name (run identifier used in output file names)
#   $10 (optional) path to a specific dense "solution" checkpoint to prune; if omitted,
#       main.py falls back to the paper's default checkpoint for $4/$5/$7

set -e

solution_arg=""
if [ -n "${10}" ]; then
  solution_arg="--solution ${10}"
fi

python main.py -exp "$1" -bs 128 --sigma "$2" --pruner "$3" --architecture "$4" --dataset "$5" --pruning_rate "$6" \
  --modeltype "$7" --epochs "$8" --name "$9" -pop 1 --num_workers 8 ${solution_arg}
