#!/bin/bash
#SBATCH --job-name=x1103-fsdp2-repro
#SBATCH --account=project_465002901
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=fsdp2-repro-%j.out
#SBATCH --error=fsdp2-repro-%j.err
#
# FSDP2 training reproduction: x1103 vs x1205 memory issue on LUMI-G
#
# Uses Axolotl + FSDP2 to load Gemma 4 31B and run 3 training steps,
# matching the exact production memory profile.
#
# Usage:
#   sbatch --constraint=x1205 repro.sh   # Expected: PASS
#   sbatch --constraint=x1103 repro.sh   # Expected: HANG or OOM
#
# Prerequisites:
#   1. Venv overlay with transformers>=5.5.0 + axolotl + liger-kernel
#   2. Gemma 4 31B downloaded locally

set -euo pipefail

# --- Configuration (override via env vars) ---
SCRATCH=/scratch/${SLURM_JOB_ACCOUNT:-project_465002901}
MODEL_PATH="${MODEL_PATH:-$SCRATCH/models/gemma-4-31b-pt}"
VENV_PATH="${VENV_PATH:-$SCRATCH/venvs/forseti-train}"
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$HOME}"

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
    exit 1
fi

# RCCL/NCCL environment for Slingshot-11 CXI
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

VENV_SITE=$VENV_PATH/lib/python3.12/site-packages
OUTPUT_DIR=/tmp/repro-output-${SLURM_JOB_ID}

# Create a minimal dummy dataset
echo ""
echo "Creating dummy dataset..."
mkdir -p $OUTPUT_DIR

srun --ntasks=1 --cpu-bind=none \
  singularity exec $BINDS \
    --env PYTHONPATH=$VENV_SITE \
    $SIF \
    python3 -c "
import json, os
dataset_path = '$OUTPUT_DIR/train.jsonl'
text = 'Sverige är ett land i Norden med en lång historia av demokrati och rättsstat. ' * 200
with open(dataset_path, 'w') as f:
    for i in range(64):
        json.dump({'text': text}, f, ensure_ascii=False)
        f.write('\n')
print(f'Created {dataset_path} (64 samples)')
"

# Create Axolotl config matching production v6 CPT
cat > $OUTPUT_DIR/config.yaml << YAMLEOF
base_model: $MODEL_PATH
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: false

datasets:
  - path: $OUTPUT_DIR/train.jsonl
    ds_type: json
    type: completion

dataset_prepared_path: $OUTPUT_DIR/prepared
sequence_len: 4096
sample_packing: false

# FSDP2 — critical setting that drives memory pressure
fsdp_config:
  fsdp_version: 2
  reshard_after_forward: true

bf16: true
tf32: false

optimizer: adamw_torch_fused
learning_rate: 7.5e-6
lr_scheduler: cosine
weight_decay: 0.01
warmup_steps: 1

micro_batch_size: 1
gradient_accumulation_steps: 16

# Liger kernel — reduces logits memory by 73.7%
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_fused_linear_cross_entropy: true
liger_rope: true
liger_rms_norm: true
liger_swiglu: true

num_epochs: 1
max_steps: 3

logging_steps: 1
save_steps: 999999
eval_steps: 999999

output_dir: $OUTPUT_DIR/model
YAMLEOF

echo "Config: $OUTPUT_DIR/config.yaml"
echo ""
echo "Launching Axolotl FSDP2 training on 8 GCDs..."
echo "If this hangs for >10 minutes, the node has the memory issue."
echo "(30-minute walltime limit; if the job hits the limit, it hung.)"
echo ""

# Launch via accelerate (same as production)
srun --ntasks=1 --cpu-bind=none \
  singularity exec $BINDS \
    --env PYTHONPATH=$VENV_SITE \
    --env NCCL_SOCKET_IFNAME=hsn \
    --env NCCL_NET_GDR_LEVEL=PHB \
    --env MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH \
    --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR \
    --env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    $SIF \
    bash -c "
      export PYTHONPATH=$VENV_SITE:\$PYTHONPATH
      accelerate launch --num_processes 8 --num_machines 1 \
        --mixed_precision bf16 \
        -m axolotl.cli.train $OUTPUT_DIR/config.yaml
    "

EXIT_CODE=$?

echo ""
echo "=== Job finished at $(date) with exit code $EXIT_CODE ==="

if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: PASS — this node can run Gemma 4 31B FSDP2 training"
else
    echo "RESULT: FAIL — exit code $EXIT_CODE"
fi
