#!/bin/bash
#SBATCH --job-name=dflex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=runs/dflex_%j.out
#SBATCH --error=runs/dflex_%j.out

set -e
cd $HOME/dflex_proj
source .venv/bin/activate
export TMPDIR=$HOME/dflex_proj/tmp
export PYTHONUNBUFFERED=1
mkdir -p runs "$TMPDIR"

echo "host=$(hostname)"; nvidia-smi -L
timeout 6 curl -sI https://api.wandb.ai >/dev/null 2>&1 \
  && echo "compute egress: YES" || echo "compute egress: NO (wandb stays offline; sync from login node later)"

python train_dflex.py \
  --data_dir cp2107-textworld-trajectories/scienceworld-compact-v2 --out_dir runs/dflex_full \
  --steps 3000 --batch_size 16 --max_len 1024 \
  --d_model 512 --layers 6 --nhead 8 --lr 3e-4 \
  --val_every 150 --run_name D-Flex-scienceworld \
  --wandb_mode offline --wandb_project cp2107-dflex
echo "FINISHED"
