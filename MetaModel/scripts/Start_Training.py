"""Helix2Text Training Script — Optimized."""

import os
import sys
import torch
import random
import traceback
import numpy as np
from datetime import datetime

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from MetaModel.configs.config import default_config
from MetaModel.training.train import train_model
from MetaModel.models.load_models import setup_fusion_and_deepspeed

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand
os.environ["NCCL_DEBUG"] = "INFO"    # Enable debugging

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_gpu_info():
    """Print detailed GPU information"""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count == 0:
        print("❌ No CUDA GPUs detected. Exiting.")
        sys.exit(1)
    print(f"🔍 Detected {gpu_count} GPU(s):")
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        memory_gb = gpu_props.total_memory / (1024**3)
        print(f"  GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
    print()


# =============================================================================
# TRAINING PROCESS
# =============================================================================

def train_process_torchrun(rank, world_size, local_rank):
    """Training process for torchrun launch."""
    # Set CUDA device FIRST
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] Set CUDA device to GPU {local_rank}")
    
    # Initialize distributed training with explicit device_id if world_size > 1
    if world_size > 1:
        if not torch.distributed.is_initialized():
            device_id = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else None
            torch.distributed.init_process_group(
                backend='nccl',
                device_id=device_id
            )
            print(f"[Rank {rank}] Initialized distributed training with device {device_id}")

    try:
        config = default_config
        config.experiment_name = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        print(f"[Rank {rank}] Setting up training...")
        setup_fusion_and_deepspeed(config)

        print(f"[Rank {rank}] Starting training: {config.training.num_epochs} epochs")
        train_model(config)
        print(f"[Rank {rank}] ✅ Training completed successfully!")

    except Exception as e:
        print(f"[Rank {rank}] ❌ Exception during training:")
        print(f"[Rank {rank}] Error: {str(e)}")
        print(f"[Rank {rank}] Full traceback:")
        traceback.print_exc()
        raise

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    set_seed(42)
    print("🚀 Helix2Text training Launcher (auto torchrun)")
    print("=" * 50)
    print_gpu_info()

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Only relaunch with torchrun if not already under torchrun and more than 1 GPU
    if "LOCAL_RANK" not in os.environ and gpu_count > 1:
        print(f"🔄 Relaunching with torchrun for {gpu_count} GPUs...")
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpu_count}",
            os.path.abspath(__file__)
        ]
        os.execvp(cmd[0], cmd)

    # If running under torchrun or single GPU, proceed as normal
    if gpu_count == 0:
        print("❌ No CUDA GPUs detected. Exiting.")
        sys.exit(1)
    elif gpu_count == 1 or "LOCAL_RANK" not in os.environ:
        print("✅ Single GPU detected or not under torchrun. Running in single process mode.")
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        train_process_torchrun(0, 1, 0)
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        print(f"✅ Launched by torchrun - Rank {rank}, Local Rank {local_rank}, World Size {world_size}")
        train_process_torchrun(rank, world_size, local_rank)

if __name__ == "__main__":
    main()
