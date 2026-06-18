#!/bin/bash
# Local (non-SLURM) worker script for running experiment 1 from main.py:
# one-shot stochastic pruning. It adds Gaussian noise of amplitude $sigma to the dense
# model, prunes it with $pruner at rate $pruning_rate, and immediately measures accuracy
# on the validation set (one_shot_static_sigma_stochastic_pruning). No fine-tuning is done.
#
# Run this from a shell where your python environment (conda/venv) is already activated.
#
# Positional arguments (mirrors local_SP_fine_tuning_run.sh so handlers stay consistent):
#   $1  experiment number (use 1 for one-shot stochastic pruning, 6 for deterministic)
#   $2  batch size
#   $3  sigma (noise amplitude)
#   $4  pruner [global, lamp, erk, random, grasp, synflow, manual]
#   $5  architecture [resnet18, resnet50, vgg19]
#   $6  dataset [cifar10, cifar100]
#   $7  pruning_rate
#   $8  modeltype [alternative, hub] - use "alternative" for the alternate_models implementation
#   $9  epochs (unused for one-shot pruning, kept for positional compatibility)
#   $10 name (run identifier used in output file names)
#   $11 (optional) path to a specific dense "solution" checkpoint to prune; if omitted,
#       main.py falls back to the paper's default checkpoint for $5/$6/$8

set -e

solution_arg=""
if [ -n "${11}" ]; then
  solution_arg="--solution ${11}"
fi

python main.py -exp "$1" -bs "$2" --sigma "$3" --pruner "$4" --architecture "$5" --dataset "$6" \
  --pruning_rate "$7" --modeltype "$8" --epochs "$9" --name "${10}" -pop 1 --num_workers 8 ${solution_arg}
