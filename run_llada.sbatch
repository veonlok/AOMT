#!/bin/bash
#SBATCH --job-name=llada_dflex
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=runs/llada_%j.out

set -e
cd $HOME/dflex_proj
source .venv/bin/activate
export HF_HOME=$HOME/dflex_proj/hf TMPDIR=$HOME/dflex_proj/tmp HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p runs "$TMPDIR"
echo "host=$(hostname)"; nvidia-smi -L
python train_llada.py \
  --lora_targets query_key_value,dense \
  --steps 200 --grad_accum 4 --batch_size 1 --max_len 384 \
  --lr 1e-4 --val_every 25 \
  --out_dir runs/llada_dflex_full \
  --run_name LLaDA-DFlex-scienceworld --wandb_mode offline
echo "FINISHED"
