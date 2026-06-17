#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=00:09:00

# set name of job
#SBATCH --job-name=pytorch_test

#SBATCH --error=pytorch_test.err

#SBATCH --output=pytorch_test.output

# set partition (devel, small, big)

#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# send mail to this address
#SBATCH --mail-user=sclaam@leeds.ac.uk

# Worker script for SLURM (sbatch) jobs running experiment 11 from main.py:
# fine-tuning after stochastic pruning (fine_tune_after_stochastic_pruning_experiment).
# Also works for experiment 6 (deterministic pruning + fine-tuning), since main.py just
# dispatches on the experiment number $1.
#
# Positional arguments:
#   $1  experiment number (11 = fine-tune after stochastic pruning, 6 = deterministic)
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

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/sclaam/.conda/envs/work/lib"
export PYTHONPATH="/users/sclaam/.conda/envs/work/lib/python3.9/site-packages"

solution_arg=""
if [ -n "${10}" ]; then
  solution_arg="--solution ${10}"
fi

python main.py -exp $1 -bs 128 --sigma $2 --pruner $3 --architecture $4 --dataset $5 --pruning_rate $6 \
  --modeltype $7 --epochs $8 --name $9 -pop 1 --num_workers 8 ${solution_arg}
