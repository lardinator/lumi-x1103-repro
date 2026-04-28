#!/usr/bin/env python3
"""
Minimal FSDP2 training reproduction for LUMI-G x1103 vs x1205 memory issue.

Uses Axolotl's internal model loading (which handles FSDP2 sharding correctly)
with a tiny synthetic dataset to reproduce the exact memory profile of production
CPT that hangs on x1103 but succeeds on x1205.

Usage (via repro.sh):
    srun ... python3 train_repro.py --model /path/to/gemma-4-31b-pt --steps 3
"""

import argparse
import json
import os
import tempfile


def create_dummy_dataset(model_path, seq_len=4096, n_samples=32):
    """Create a minimal JSONL dataset for Axolotl completion training."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset_dir = tempfile.mkdtemp(prefix="repro_dataset_")
    dataset_path = os.path.join(dataset_dir, "train.jsonl")

    # Simple completion examples (content doesn't matter, just memory pressure)
    text = "Sverige är ett land i Norden. " * (seq_len // 6)

    with open(dataset_path, "w") as f:
        for i in range(n_samples):
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")

    return dataset_path


def create_axolotl_config(model_path, dataset_path, output_dir, seq_len=4096, steps=3):
    """Create minimal Axolotl config matching production v6 CPT."""
    config = {
        "base_model": model_path,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "load_in_8bit": False,
        "load_in_4bit": False,
        "datasets": [
            {
                "path": "json",
                "data_files": dataset_path,
                "type": "completion",
            }
        ],
        "dataset_prepared_path": os.path.join(output_dir, "prepared"),
        "sequence_len": seq_len,
        "sample_packing": False,
        # FSDP2 — critical setting
        "fsdp_config": {
            "fsdp_version": 2,
            "reshard_after_forward": True,
        },
        # Mixed precision
        "bf16": True,
        "tf32": False,
        # Optimizer — fused AdamW matches production
        "optimizer": "adamw_torch_fused",
        "learning_rate": 7.5e-6,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 1,
        # Batch size — matches production
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        # Liger kernel — critical for memory profile
        "plugins": ["liger_kernel"],
        "liger_fused_linear_cross_entropy": True,
        "liger_rope": True,
        "liger_rms_norm": True,
        "liger_swiglu": True,
        # Training
        "num_epochs": 1,
        "max_steps": steps,
        # Logging
        "logging_steps": 1,
        "save_steps": 999999,
        "eval_steps": 999999,
        "output_dir": os.path.join(output_dir, "model"),
    }

    config_path = os.path.join(output_dir, "config.yaml")

    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def main():
    parser = argparse.ArgumentParser(description="FSDP2 memory reproduction via Axolotl")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("MODEL_PATH", "google/gemma-4-31b-pt"),
                        help="Path to Gemma 4 31B model")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

    if rank == 0:
        import torch
        total_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

        print(f"\n{'='*60}")
        print(f"  FSDP2 Training Reproduction (via Axolotl)")
        print(f"{'='*60}")
        print(f"  Model:      {args.model}")
        print(f"  Seq len:    {args.seq_len}")
        print(f"  Steps:      {args.steps}")
        print(f"  World size: {world_size}")
        print(f"  HBM/GCD:    {total_mem:.2f} GiB")
        print(f"{'='*60}")

        print("\n=== Creating dummy dataset ===")

    # Only rank 0 creates the dataset
    output_dir = os.environ.get("REPRO_OUTPUT", "/tmp/repro-output")
    os.makedirs(output_dir, exist_ok=True)

    if rank == 0:
        dataset_path = create_dummy_dataset(args.model, args.seq_len)
        config_path = create_axolotl_config(
            args.model, dataset_path, output_dir, args.seq_len, args.steps
        )
        print(f"  Dataset: {dataset_path}")
        print(f"  Config:  {config_path}")
        print(f"\n=== Starting Axolotl training ===")
        print(f"  (This is where x1103 nodes hang or OOM)")
    else:
        # Other ranks wait for rank 0 to create files, then use same paths
        import time
        time.sleep(2)
        # Find the config
        config_path = os.path.join(output_dir, "config.yaml")

    # Use Axolotl's train CLI
    import sys
    sys.argv = ["axolotl", "train", config_path]

    from axolotl.cli.main import main as axolotl_main
    axolotl_main()


if __name__ == "__main__":
    main()
