#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x_%j.out
# Usage: sbatch -J <run_name> run_v4.sbatch <run_name> <objective_set> <schedule> [steps] [seed]
# e.g.:  sbatch -J dflex_v3 run_v4.sbatch dflex_v3 d_flex       constant 600 0
#        sbatch -J dprog    run_v4.sbatch dprog    d_progressive constant 600 0
#        sbatch -J dar_v3   run_v4.sbatch dar_v3   d_ar_refined  constant 600 0
set -e
RUN=$1; OBJ=$2; SCHED=$3; STEPS=${4:-600}; SEED=${5:-0}

cd $HOME/dflex_proj
source .venv/bin/activate
export HF_HOME=$HOME/dflex_proj/hf TMPDIR=$HOME/dflex_proj/tmp HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p runs "$TMPDIR"
echo "host=$(hostname) run=$RUN obj=$OBJ sched=$SCHED steps=$STEPS seed=$SEED"
nvidia-smi -L

python train_llada_rb.py \
  --run_name "$RUN" --objective_set "$OBJ" --mask_schedule "$SCHED" \
  --mask_rate 0.20 --steps "$STEPS" --grad_accum 4 --batch_size 1 --max_len 8192 \
  --lr 1e-4 --weight_decay 0.1 --lora_dropout 0.1 --ema_decay 0.995 \
  --val_every 20 --val_batches 15 --seed "$SEED" \
  --full_val_at_end --probe_after_train \
  --out_dir "runs/$RUN" --wandb_mode offline

# --- post-training: fix adapter path, upload to HF, sync wandb ---
# Must unset HF_HUB_OFFLINE and HF_HOME so the subprocess picks up the write
# token from ~/.cache/huggingface/token (set via `hf auth login` on login node).
sed -i 's|/home/a/arnavk4/dflex_proj/llada2mini|inclusionAI/LLaDA2.0-mini|g' \
    "runs/$RUN/lora_adapter/adapter_config.json" 2>/dev/null || true
find "runs/$RUN/lora_adapter" -name "README.md" -exec \
    sed -i 's|/home/a/arnavk4/dflex_proj/llada2mini|inclusionAI/LLaDA2.0-mini|g' {} \; 2>/dev/null || true

unset HF_HUB_OFFLINE HF_HOME

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='runs/$RUN/lora_adapter', repo_id='AK2802/AOMT',
                  path_in_repo='$RUN', repo_type='model')
print('[hf] uploaded to AK2802/AOMT/$RUN')
"

wandb sync --sync-all
echo "FINISHED $RUN"
