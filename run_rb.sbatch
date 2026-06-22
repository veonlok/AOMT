#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x_%j.out
set -e
cd $HOME/dflex_proj; source .venv/bin/activate
export HF_HOME=$HOME/dflex_proj/hf TMPDIR=$HOME/dflex_proj/tmp HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p runs "$TMPDIR"
echo "host=$(hostname) run=$1 obj=$2 sched=$3"; nvidia-smi -L
python train_llada_rb.py --run_name "$1" --objective_set "$2" --mask_schedule "$3" \
  --mask_rate 0.20 --steps 400 --grad_accum 4 --batch_size 1 --max_len 1024 \
  --lr 1e-4 --val_every 25 --out_dir "runs/$1" --wandb_mode offline
echo FINISHED
