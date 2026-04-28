# LUMI-G x1103 vs x1205 Memory Issue — Reproduction

MI250X GCDs on **x1103** racks lose usable HBM under FSDP2 training workloads,
causing large model training to hang. The same workload succeeds on **x1205** racks.

## Observed Behavior

| | x1205 | x1103 |
|---|---|---|
| FSDP2 + Gemma 4 31B (seq_len=4096) | Runs normally | Hangs (42+ min) or OOM |
| Simple tensor allocation (52 GB) | PASS | PASS |
| rocm-smi reported VRAM | 68.70 GB | 68.70 GB |
| Production peak (training) | 52.96 GiB alloc, 59.82 GiB reserved | Never completes |

The issue does **not** appear in simple allocation tests — only under real distributed
training with FSDP2 all-gather + resharding + optimizer states.

## Environment

- **LUMI-G**: AMD MI250X (64 GB HBM per GCD, 8 GCDs per node)
- **Container**: `/appl/local/laifs/containers/lumi-multitorch-latest.sif`
- **PyTorch**: 2.9.1+rocm6.4 (from container)
- **Venv overlay**: transformers 5.5.0, axolotl 0.16.1, liger-kernel 0.7.0, torchft 0.1.1
- **Model**: Gemma 4 31B text-only (bf16, FSDP2 sharded across 8 GCDs)
- **Training**: Axolotl CPT with FSDP2, `reshard_after_forward=True`, AdamW fused (bf16)

## Reproduction Steps

### 1. Setup venv overlay (one-time)

```bash
# Inside a LUMI-G allocation:
VENV=/scratch/$PROJECT/venvs/repro-train

singularity exec \
  -B /scratch -B /appl \
  /appl/local/laifs/containers/lumi-multitorch-latest.sif \
  bash -c "
    python3 -m venv --system-site-packages $VENV
    . $VENV/bin/activate
    pip install --no-deps transformers==5.5.0 liger-kernel==0.7.0 axolotl==0.16.1
  "
```

### 2. Download model

Requires accepting the [Gemma 4 license](https://huggingface.co/google/gemma-4-31b-pt)
and setting `HF_TOKEN`.

```bash
singularity exec \
  -B /scratch -B /appl \
  /appl/local/laifs/containers/lumi-multitorch-latest.sif \
  bash -c "
    . $VENV/bin/activate
    huggingface-cli download google/gemma-4-31b-pt --local-dir /scratch/$PROJECT/models/gemma-4-31b-pt
  "
```

### 3. Run on both rack types

```bash
# Expected to SUCCEED:
sbatch --constraint=x1205 repro.sh

# Expected to HANG or OOM:
sbatch --constraint=x1103 repro.sh
```

### 4. Interpret results

- **x1205**: All 5 phases complete. Phase 5 (training step) reports peak ~53 GiB, ~60 GiB reserved.
- **x1103**: Hangs during Phase 4 or 5 (FSDP2 all-gather / training step), or OOM.
  If the job hits the 30-minute walltime, it hung.

## Files

| File | Purpose |
|------|---------|
| `repro.sh` | SLURM submission script (single node, 8 GCDs, 30 min) |
| `train_repro.py` | Minimal FSDP2 training: loads Gemma 4 31B, runs 3 training steps |
| `config.yaml` | Axolotl-style config matching production training |
| `requirements.txt` | Pip packages for venv overlay |

## Production Job Evidence

| Job ID | Rack | Node | Result |
|--------|------|------|--------|
| (see support email) | x1103 | (see sacct) | Hung 42 min, killed |
| (see support email) | x1205 | (see sacct) | Completed normally |

Same SLURM script, same container, same config. Only difference: `--constraint`.

## What We Think Is Happening

Something on x1103 racks consumes HBM that is invisible to `rocm-smi` and
`torch.cuda.memory_allocated()` but reduces the memory available for RCCL
collectives (all-gather during FSDP2 unsharding). Production training peaks at
~60 GiB reserved per GCD — on x1205 this fits in the 64 GiB HBM, on x1103 it
doesn't.

Possible causes:
- ROCm driver/firmware difference between rack generations
- HBM ECC or page table overhead difference
- RCCL transport buffer allocation difference (CXI configuration)
- Faulty GCDs with reduced effective HBM

## Contact

Alexandro Martini — EuroHPC project `project_465002901`
