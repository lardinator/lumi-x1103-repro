#!/bin/bash
#SBATCH --job-name=x1103-fsdp2-repro
#SBATCH --account=project_465002901
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=fsdp2-repro-%j.out
#SBATCH --error=fsdp2-repro-%j.err
#
# FSDP2 training reproduction: x1103 vs x1205 memory issue on LUMI-G
#
# Usage:
#   sbatch --constraint=x1205 repro.sh   # Expected: PASS
#   sbatch --constraint=x1103 repro.sh   # Expected: HANG or OOM
#
# Prerequisites:
#   1. Venv overlay with transformers>=5.5.0 (see README.md)
#   2. Gemma 4 31B downloaded (see README.md)
#
# Override paths via environment:
#   MODEL_PATH=/path/to/model VENV_PATH=/path/to/venv sbatch ... repro.sh

set -euo pipefail

# --- Configuration (override via env vars) ---
SCRATCH=/scratch/${SLURM_JOB_ACCOUNT:-project_465002901}
MODEL_PATH="${MODEL_PATH:-$SCRATCH/models/gemma-4-31b-pt}"
VENV_PATH="${VENV_PATH:-$SCRATCH/venvs/repro-train}"
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BINDS="-B /pfs -B /scratch -B /project -B /flash -B /appl -B /opt/cray -B /usr/lib64/libcxi.so.1 -B /var/spool/slurmd"

echo "=============================================="
echo "  FSDP2 Memory Reproduction Test"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Model:      $MODEL_PATH"
echo "Venv:       $VENV_PATH"
echo "Container:  $SIF"

# Identify rack type
NODE_FEATURES=$(scontrol show node $(hostname) | grep ActiveFeatures)
echo "Features:   $NODE_FEATURES"

if echo "$NODE_FEATURES" | grep -q "x1103"; then
    echo "Rack type:  x1103 (EXPECTED TO FAIL)"
elif echo "$NODE_FEATURES" | grep -q "x1205"; then
    echo "Rack type:  x1205 (EXPECTED TO SUCCEED)"
else
    echo "Rack type:  UNKNOWN"
fi
echo "=============================================="

# Validate prerequisites
if [ ! -f "$SIF" ]; then
    echo "ERROR: Container not found at $SIF"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "       Download with: huggingface-cli download google/gemma-4-31b-pt --local-dir $MODEL_PATH"
    exit 1
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "WARNING: Venv not found at $VENV_PATH — using container packages only"
    echo "         Gemma 4 requires transformers>=5.5.0 which may not be in the container."
    echo "         See README.md for setup instructions."
fi

# RCCL/NCCL environment for Slingshot-11 CXI
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

# ROCm GPU visibility
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

VENV_SITE=$VENV_PATH/lib/python3.12/site-packages

echo ""
echo "Launching FSDP2 training on 8 GCDs..."
echo "If this hangs for >10 minutes, the node has the memory issue."
echo "(30-minute walltime limit; if the job hits the limit, it hung.)"
echo ""

srun --cpu-bind=none \
  singularity exec $BINDS \
    --env PYTHONPATH=$VENV_SITE \
    --env MODEL_PATH=$MODEL_PATH \
    $SIF \
    bash -c ". $VENV_PATH/bin/activate 2>/dev/null; python3 $SCRIPT_DIR/train_repro.py --model $MODEL_PATH"

echo ""
echo "=== Job finished at $(date) ==="
