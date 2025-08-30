"""DeepSpeed configuration utilities."""


def create_deepspeed_config(config, world_size=1):
    """Create DeepSpeed configuration optimized for GPT OSS with Harmony."""
    
    # Calculate microbatch size
    effective_batch_size = config.training.batch_size * config.training.grad_accum_steps * world_size
    micro_batch_size = config.training.batch_size

    # GPT OSS with Harmony always enabled - optimized configuration
    ds_config = {
        "train_batch_size": effective_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": config.training.grad_accum_steps,

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.training.learning_rate,
                "betas": [0.9, 0.95],  # Optimized for large language models
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.training.learning_rate,
                "warmup_num_steps": 100,
            }
        },

        # Enhanced memory optimization for GPT OSS + DNA-text fusion
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,  # Reduced bucket size to handle mixed dtypes
            "allgather_bucket_size": 5e7,
            "round_robin_gradients": True,  # Help with gradient bucketing
            "stage3_max_live_parameters": 1e9,  # Limit live parameters
            "stage3_max_reuse_distance": 1e9,   # Optimize parameter reuse
            "stage3_prefetch_bucket_size": 5e7,  # Prefetch optimization
            "stage3_param_persistence_threshold": 1e6,
            # Enable CPU offloading for higher ZeRO stages (stage 3 configured above)
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,  # Multiple buffers for overlap
                "fast_init": False
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            },
        },

        "gradient_clipping": 5.0,  # Standard clipping for stability

        # Mixed precision optimized for GPT OSS models
        "bf16": {
            "enabled": True,  # BF16 is preferred for GPT OSS models
        },
        "fp16": {
            "enabled": False,  # Disable FP16 when using BF16
        },
        
        # Additional optimizations for GPT OSS + Harmony
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 4,
        },
        
        # Communication optimizations
        "communication_data_type": "bf16",
        "allreduce_always_fp32": False,
        
        # Memory profiling (can be disabled in production)
        "memory_breakdown": False,
        
        # Harmony-specific optimizations
        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 3,
            "detailed": False,
        }
    }
    
    return ds_config

