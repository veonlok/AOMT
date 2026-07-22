#!/bin/bash
#SBATCH --job-name=eval_nextobs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.out

# Usage: sbatch run_evaluation_suite.sbatch [decode] [decode_steps]
# decode: one_shot_argmax, iterative_diffusion, or semi_autoregressive
set -euo pipefail

DECODE=${1:-one_shot_argmax}
DECODE_STEPS=${2:-16}
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/dflex_proj}
BASE_MODEL=${BASE_MODEL:-$PROJECT_ROOT/models/LLaDA2.0-mini}
ADAPTER_ROOT=${ADAPTER_ROOT:-$PROJECT_ROOT/models/AOMT}
OUTPUT_DIR=$PROJECT_ROOT/runs/evaluation-suite-v1/$DECODE
FIXTURES=${FIXTURES:-$PROJECT_ROOT/runs/evaluation-data/scienceworld-compact-v2-validation-last.jsonl}
DATASET_MANIFEST=${DATASET_MANIFEST:-$PROJECT_ROOT/cp2107-textworld-trajectories/scienceworld-compact-v2/manifest.json}

cd "$PROJECT_ROOT"
source .venv/bin/activate
export HF_HOME=$PROJECT_ROOT/hf
export TMPDIR=$PROJECT_ROOT/tmp
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR" "$TMPDIR"

for required in \
  "$FIXTURES" \
  "$DATASET_MANIFEST" \
  "$BASE_MODEL/model.safetensors.index.json"
do
  if [[ ! -f "$required" ]]; then
    echo "missing required model file: $required" >&2
    exit 2
  fi
done

python -m evaluations.cli preflight \
  --workspace "$PROJECT_ROOT" \
  --output "$OUTPUT_DIR/preflight.json" \
  --require-ready

echo "host=$(hostname) decode=$DECODE decode_steps=$DECODE_STEPS"
nvidia-smi -L

python -m evaluations.cli run-suite \
  --fixtures "$FIXTURES" \
  --model-dir "$BASE_MODEL" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --run "causal20:0:$ADAPTER_ROOT/causal20" \
  --run "dar:0:$ADAPTER_ROOT/dar" \
  --run "dar_section:0:$ADAPTER_ROOT/dar_section" \
  --run "dar_v2:0:$ADAPTER_ROOT/dar_v2" \
  --run "dar_v3:0:$ADAPTER_ROOT/dar_v3" \
  --run "dflex:0:$ADAPTER_ROOT/dflex" \
  --run "dflex_v2:0:$ADAPTER_ROOT/dflex_v2" \
  --run "dflex_v3:0:$ADAPTER_ROOT/dflex_v3" \
  --run "dprog:0:$ADAPTER_ROOT/dprog" \
  --run "dprog_med:0:$ADAPTER_ROOT/dprog_med" \
  --run "drandom_v2:0:$ADAPTER_ROOT/drandom_v2" \
  --run "rb_const20:0:$ADAPTER_ROOT/rb_const20" \
  --run "rb_curric:0:$ADAPTER_ROOT/rb_curric" \
  --primary dflex_v2 \
  --decode "$DECODE" \
  --decode-steps "$DECODE_STEPS" \
  --batch-size 1 \
  --max-len 8192 \
  --device cuda \
  --dtype bfloat16 \
  --bootstrap-samples 2000 \
  --seed 0 \
  --output-dir "$OUTPUT_DIR"

echo "FINISHED evaluation suite: $OUTPUT_DIR"
