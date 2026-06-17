#!/bin/bash
# Worker script for SLURM (sbatch) jobs running train_CIFAR10.py: trains a dense model from
# scratch on CIFAR10/CIFAR100. This is the preliminary step that produces the "solution"
# checkpoints later pruned by the stochastic pruning experiments (10, 11, 19) in main.py.
#
# Positional arguments:
#   $1  model [resnet18, resnet50, vgg19]
#   $2  dataset [cifar10, cifar100]
#   $3  epochs
#   $4  batch_size
#   $5  lr (learning rate)
#   $6  num_workers
#   $7  save_folder (directory where the trained checkpoint is written)
#   $8  data_folder (directory where the CIFAR dataset is/will be downloaded)
#   $9  name (run identifier used in output file names / checkpoint naming)

export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/users/sclaam/.conda/envs/work/lib"
export PYTHONPATH="/users/sclaam/.conda/envs/work/lib/python3.9/site-packages"

python train_CIFAR10.py --experiment 1 --model "$1" --dataset "$2" --epochs "$3" --batch_size "$4" \
  --lr "$5" --num_workers "$6" --save_folder "$7" --data_folder "$8" --seed_name "$9" --save 1
