cd ~/dflex_proj
source .venv/bin/activate
export HF_HOME=$HOME/dflex_proj/hf
export TMPDIR=$HOME/dflex_proj/tmp
export PIP_CACHE_DIR=$HOME/dflex_proj/pipcache
export HF_HUB_OFFLINE=1
echo "dflex env ready. GPU work via srun/sbatch only (never on login node)."
