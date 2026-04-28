#!/usr/bin/env python3
"""
Minimal FSDP2 training reproduction for LUMI-G x1103 vs x1205 memory issue.

Loads Gemma 4 31B with FSDP2 (reshard_after_forward=True) and runs 3 training
steps with AdamW + liger fused linear cross-entropy — the exact memory profile
of production CPT that hangs on x1103 but succeeds on x1205.

Usage:
    # Via SLURM (see repro.sh):
    srun ... python3 train_repro.py --model /path/to/gemma-4-31b-pt

    # Or via Axolotl (closer to production):
    srun ... accelerate launch -m axolotl.cli.train config.yaml
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


def log(msg, rank=0):
    """Print only from rank 0."""
    if int(os.environ.get("RANK", 0)) == rank:
        print(msg, flush=True)


def memory_report(local_rank, tag=""):
    """Report GPU memory for this rank."""
    alloc = torch.cuda.memory_allocated(local_rank) / 2**30
    peak = torch.cuda.max_memory_allocated(local_rank) / 2**30
    reserved = torch.cuda.memory_reserved(local_rank) / 2**30
    total = torch.cuda.get_device_properties(local_rank).total_memory / 2**30
    headroom = total - reserved
    return (f"  GCD {local_rank}: alloc={alloc:.2f} GiB, peak={peak:.2f} GiB, "
            f"reserved={reserved:.2f} GiB, headroom={headroom:.2f} GiB{tag}")


def report_all_ranks(local_rank, rank, world_size, tag=""):
    """Synchronized memory report from all ranks."""
    for r in range(world_size):
        if r == rank:
            print(memory_report(local_rank, tag), flush=True)
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.environ.get(
        "MODEL_PATH", "google/gemma-4-31b-pt"))
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--use-liger", action="store_true", default=True,
                        help="Use liger-kernel fused ops (default: True)")
    parser.add_argument("--no-liger", action="store_true",
                        help="Disable liger-kernel")
    args = parser.parse_args()

    if args.no_liger:
        args.use_liger = False

    # --- Distributed setup ---
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    total_mem = torch.cuda.get_device_properties(local_rank).total_memory / 2**30

    log(f"\n{'='*60}")
    log(f"  FSDP2 Training Reproduction")
    log(f"{'='*60}")
    log(f"  Model:      {args.model}")
    log(f"  Seq len:    {args.seq_len}")
    log(f"  Steps:      {args.steps}")
    log(f"  Liger:      {args.use_liger}")
    log(f"  World size: {world_size}")
    log(f"  HBM/GCD:    {total_mem:.2f} GiB")
    log(f"{'='*60}")

    # --- Phase 1: Apply liger-kernel patches ---
    if args.use_liger:
        log("\n=== Phase 1: Applying liger-kernel patches ===")
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gemma4

            apply_liger_kernel_to_gemma4(
                fused_linear_cross_entropy=True,
                rope=True,
                rms_norm=True,
                swiglu=True,
            )
            log("  Liger patches applied (fused_linear_cross_entropy=True)")
        except ImportError:
            log("  WARNING: liger-kernel not available, skipping")
            args.use_liger = False
        except Exception as e:
            log(f"  WARNING: liger-kernel failed: {e}")
            args.use_liger = False
    else:
        log("\n=== Phase 1: Liger disabled ===")

    # --- Phase 2: Load model ---
    log(f"\n=== Phase 2: Loading model ===")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Loaded in {time.time() - t0:.1f}s — {n_params / 1e9:.2f}B params")

    dist.barrier()

    # --- Phase 3: Apply FSDP2 ---
    log(f"\n=== Phase 3: Applying FSDP2 (reshard_after_forward=True) ===")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    # Shard each transformer layer individually
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True)

    model.cuda()
    dist.barrier()

    log("\n  Post-shard memory:")
    report_all_ranks(local_rank, rank, world_size)

    # --- Phase 4: Setup optimizer ---
    log(f"\n=== Phase 4: Creating AdamW fused optimizer ===")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=7.5e-6,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        fused=True,
    )

    dist.barrier()
    log("  Post-optimizer memory:")
    report_all_ranks(local_rank, rank, world_size)

    # --- Phase 5: Training steps ---
    log(f"\n=== Phase 5: Running {args.steps} training steps ===")
    log(f"  (This is where x1103 is expected to hang or OOM)")

    vocab_size = model.config.vocab_size

    for step in range(1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats(local_rank)
        t0 = time.time()

        # Random input (we only care about memory, not convergence)
        input_ids = torch.randint(0, vocab_size, (1, args.seq_len), device="cuda")
        labels = input_ids.clone()

        try:
            # Forward
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss

            # Backward
            loss.backward()

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            elapsed = time.time() - t0

            peak = torch.cuda.max_memory_allocated(local_rank) / 2**30
            reserved = torch.cuda.memory_reserved(local_rank) / 2**30

            log(f"\n  Step {step}/{args.steps}: loss={loss.item():.4f}, "
                f"time={elapsed:.1f}s, peak={peak:.2f} GiB, "
                f"reserved={reserved:.2f} GiB, "
                f"headroom={total_mem - reserved:.2f} GiB")

            if step == 1:
                log(f"\n  Step 1 detailed memory (all ranks):")
                report_all_ranks(local_rank, rank, world_size,
                                 tag=f"  — {'PASS' if total_mem - peak > 1.0 else 'TIGHT'}")

        except torch.cuda.OutOfMemoryError as e:
            peak = torch.cuda.max_memory_allocated(local_rank) / 2**30
            log(f"\n  Step {step}: OOM!")
            log(f"  Peak: {peak:.2f} GiB / {total_mem:.2f} GiB")
            log(f"  Error: {str(e)[:200]}")
            log(f"\n  RESULT: FAIL — OOM during training step {step}")
            break

        except Exception as e:
            log(f"\n  Step {step}: {type(e).__name__}: {str(e)[:300]}")
            log(f"\n  RESULT: FAIL")
            break
    else:
        log(f"\n  RESULT: PASS — all {args.steps} steps completed")

    dist.barrier()
    log(f"\n=== Complete at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
