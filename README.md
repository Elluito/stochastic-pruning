# Stochastic Pruning for Neural Networks

Code for reproducing the experiments from *"Stochastic Pruning for Neural Networks"*,
published in the Proceedings of the International Joint Conference on Neural Networks
DOI: 10.1109/IJCNN64981.2025.11228768

---

## Repository structure

```
stochastic-pruning/
├── main.py                          # All SP experiment functions + LeMain() entry point
├── train_CIFAR10.py                 # Standalone dense-model training script
├── sparse_ensemble_utils.py         # Shared utilities (model loading, pruning helpers)
├── feature_maps_utils.py            # Feature map / variance utilities
├── fpgm_pruner.py                   # FPGM magnitude pruner
├── plot_utils.py                    # Plotting helpers
│
├── alternate_models/                # ResNet18, ResNet50, VGG19 for CIFAR/ImageNet
├── GRASP/                           # GraSP pruner
├── layer_adaptive_sparsity/         # LAMP / ERK pruners (tools/pruners/)
├── synflow_snip_graps/              # SynFlow / SNIP pruners
├── flowandprune/                    # cal_grad helper used by sparse_ensemble_utils
├── shrinkbench/                     # FLOPs counter (metrics/flops.py)
│
├── trained_models/                  # Dense model checkpoints (input — place here or use --solution)
│   ├── cifar10/
│   └── cifar100/
├── stochastic_pruning_models/       # Pruned model outputs written by main.py
├── stochastic_pruning_data/         # Sigma/PR data files written by main.py
├── datasets/                        # CIFAR dataset root (auto-downloaded by PyTorch)
│
├── slurm_train_CIFAR10_handler.sh   # Launch dense-model training jobs (preliminary step)
├── slurm_train_CIFAR10_run.sh       # SLURM worker for train_CIFAR10.py
├── slurm_SP_one_shot_pruning_handler.sh  # Launch exp 10 (one-shot stochastic pruning)
├── slurm_SP_one_shot_pruning_run.sh      # SLURM worker for exp 10
├── slurm_SP_fine_tuning_handler.sh  # Launch exp 11 (SP + fine-tune) and exp 6 (det. baseline)
├── slurm_SP_fine_tuning_run.sh      # SLURM worker for exp 11 / exp 6
├── slurm_SP_moo_search_handler.sh   # Launch exp 19 (Optuna MOO sigma/PR search)
└── slurm_SP_moo_search_run.sh       # SLURM worker for exp 19
```

---

## Workflow

### Step 0 — Install dependencies

```bash
conda env create -f environment.yml
conda activate <env_name>
```

### Step 1 — Train dense models (or bring your own checkpoints)

```bash
bash slurm_train_CIFAR10_handler.sh
```

Trains ResNet18, ResNet50, VGG19 on CIFAR-10 and CIFAR-100 (200 epochs each).
Checkpoints are saved to `trained_models/{cifar10,cifar100}/`.

To use existing checkpoints instead, pass them via `--solution` on the command line
(see *Running experiments manually* below).

### Step 2 — One-shot stochastic pruning (Table 1, exp 10)

```bash
bash slurm_SP_one_shot_pruning_handler.sh
```

Runs `one_shot_static_sigma_stochastic_pruning()` for all 6 model/dataset combinations
from Table 1 of the paper (5 independent seeds via `--array=1-5`).

### Step 3 — Fine-tune after stochastic pruning (Table 1, exp 11) + deterministic baseline (exp 6)

```bash
bash slurm_SP_fine_tuning_handler.sh
```

Submits both exp=11 (stochastic) and exp=6 (deterministic) jobs per combination.

### Step 4 — Multi-objective Optuna search for σ and pruning rate (exp 19)

> **Note**: the `optuna.create_study` / `study.optimize` block inside
> `run_pr_sigma_search_MOO_for_cfg()` (exp 19) is commented out in `main.py`.
> Uncomment lines ~1686–1707 before running. The equivalent fully-working pattern
> is in exp 21 (`run_pr_sigma_fine_tuned_search_MOO_for_cfg`).

```bash
bash slurm_SP_moo_search_handler.sh
```

Optuna results are stored in `find_pr_sigma_database_MOO_*.dep` (SQLite) so the
search can be resumed across jobs.

---

## Running experiments manually

```bash
python main.py \
  -exp 10 \
  --architecture resnet18 \
  --dataset cifar10 \
  --pruner global \
  --sigma 0.005 \
  --pruning_rate 0.9 \
  --modeltype alternative \
  --epochs 0 \
  --name my_run \
  --solution trained_models/cifar10/my_resnet18.pth
```

Key CLI arguments (`run_le_Main_with_external_parameters`):

| Flag | Description |
|------|-------------|
| `-exp` | Experiment number (10=one-shot, 11=fine-tune, 19=MOO search) |
| `--architecture` | `resnet18`, `resnet50`, `vgg19` |
| `--dataset` | `cifar10`, `cifar100` |
| `--pruner` | `global`, `lamp`, `erk`, `random`, `grasp`, `synflow` |
| `--sigma` | Gaussian noise std added before pruning |
| `--pruning_rate` | Fraction of weights to prune (e.g. `0.9` = 90% sparse) |
| `--modeltype` | `alternative` (use `alternate_models/`) |
| `--epochs` | Fine-tuning epochs (0 for one-shot, 200 for Table 1 FT) |
| `--solution` | Path to pretrained dense checkpoint. If omitted, falls back to the hardcoded default path for this dataset/modeltype/architecture in `LeMain()`. |
| `--sampler` | Optuna sampler: `tpe` or `nsga` (exp 18/19/21 only) |
| `--trials` | Number of Optuna trials (exp 18/19/21 only) |
| `--functions` | Fitness function index: `1` or `2` (exp 18/19/21 only) |

---

## Output files

| Location | Contents |
|----------|----------|
| `stochastic_pruning_models/` | Saved pruned model checkpoints (`.pth`) |
| `stochastic_pruning_data/` | Per-layer sigma/PR data files (`.pth`, `.csv`) |
| `trained_models/` | Dense model inputs (written by `train_CIFAR10.py`, read by `main.py`) |
| `find_pr_sigma_*_database_MOO_*.dep` | Optuna SQLite study files (exp 19/21) |
| `*.out` / `*.err` | SLURM stdout/stderr logs |

---

## Table 1 parameters

| Model | Dataset | σ | Pruning rate |
|-------|---------|---|-------------|
| ResNet-18 | CIFAR-10 | 0.005 | 0.90 |
| ResNet-18 | CIFAR-100 | 0.003 | 0.90 |
| ResNet-50 | CIFAR-10 | 0.003 | 0.95 |
| ResNet-50 | CIFAR-100 | 0.001 | 0.85 |
| VGG-19 | CIFAR-10 | 0.003 | 0.95 |
| VGG-19 | CIFAR-100 | 0.001 | 0.80 |
