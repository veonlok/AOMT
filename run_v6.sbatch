#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x_%j.out
# Usage: sbatch -J <name> run_v6.sbatch <name> <obj> <sched> [steps] [seed] [prog_exp] [mask_rate] [datasets] [grad_accum]
# Examples:
#   sbatch -J dar_section_v2    run_v6.sbatch dar_section_v2    d_ar_section  constant 800 0 1.0 0.20 scienceworld-v2 4
#   sbatch -J dprog_fast_v2     run_v6.sbatch dprog_fast_v2     d_progressive constant 800 0 1.0 0.20 scienceworld-v2 4
#   sbatch -J dprog_med_v2      run_v6.sbatch dprog_med_v2      d_progressive constant 800 0 2.0 0.20 scienceworld-v2 4
#   sbatch -J dprog_slow_v2     run_v6.sbatch dprog_slow_v2     d_progressive constant 800 0 3.0 0.20 scienceworld-v2 4
#   sbatch -J dprog_fast_alf_v2 run_v6.sbatch dprog_fast_alf_v2 d_progressive constant 800 0 1.0 0.20 alfworld-v2 8
set -e
RUN=$1; OBJ=$2; SCHED=$3; STEPS=${4:-800}; SEED=${5:-0}; PROG_EXP=${6:-1.0}
MASK_RATE=${7:-0.20}; DATASETS=${8:-scienceworld-v2}; GRAD_ACCUM=${9:-4}

cd $HOME/dflex_proj
source .venv/bin/activate
export HF_HOME=$HOME/dflex_proj/hf TMPDIR=$HOME/dflex_proj/tmp HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p runs "$TMPDIR"
echo "host=$(hostname) run=$RUN obj=$OBJ sched=$SCHED steps=$STEPS seed=$SEED prog_exp=$PROG_EXP mask_rate=$MASK_RATE datasets=$DATASETS grad_accum=$GRAD_ACCUM"
nvidia-smi -L

python train_llada_rb.py \
  --run_name "$RUN" --objective_set "$OBJ" --mask_schedule "$SCHED" \
  --mask_rate "$MASK_RATE" --steps "$STEPS" --grad_accum "$GRAD_ACCUM" \
  --batch_size 1 --max_len 8192 \
  --lr 1e-4 --weight_decay 0.1 --lora_dropout 0.1 --ema_decay 0.995 \
  --prog_exponent "$PROG_EXP" --prog_max_span 25 \
  --val_every 20 --val_batches 15 --seed "$SEED" \
  --datasets "$DATASETS" --data_dir data --data_dir_tw data/textworld \
  --full_val_at_end \
  --out_dir "runs/$RUN" --wandb_mode offline
# OAE semantic eval (SAS + NLI OOC) is enabled by default — disable with --no_sem_eval

# --- post-training: fix adapter path, upload to HF, sync wandb ---
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
