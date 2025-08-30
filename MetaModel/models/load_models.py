import os
import torch
import deepspeed

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model

from ..data.load_dataset import get_data_loaders
from ..models.dna_text_fusion import DNATextFusion
from ..configs.deepspeed_config import create_deepspeed_config


def setup_fusion_and_deepspeed(config):
    """Loads models, creates fusion, initializes DeepSpeed."""

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    # Load tokenizers (lightweight - can load on all ranks)
    dna_tokenizer = AutoTokenizer.from_pretrained(config.model.dna_model_path, trust_remote_code=True)
    dna_tokenizer.model_max_length = config.model.dna_max_length
    text_tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_path, trust_remote_code=True)

    print(f"[Rank {rank}] Loading full models...")
    
    # Load DNA model with memory optimization
    dna_model = AutoModelForCausalLM.from_pretrained(
        config.model.dna_model_path,
        max_position_embeddings=config.model.dna_max_length,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        #trust_remote_code=True,
        use_cache=False,
        device_map=None
    )

    # Load text model with special handling for GPT OSS models
    text_model = AutoModelForCausalLM.from_pretrained(
        config.model.text_model_path,
        quantization_config=Mxfp4Config(dequantize=True),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache=False,
        device_map=None
    )

    text_lora_config = LoraConfig(
        r=config.model.text_r, # increase r will enable more trainable parameters
        lora_alpha=16, 
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )

    dna_lora_config = LoraConfig(
        r=config.model.dna_r, # increase r will enable more trainable parameters
        lora_alpha=16, 
        target_modules="all-linear",
    )

    text_model = get_peft_model(text_model, text_lora_config)
    dna_model = get_peft_model(dna_model, dna_lora_config)
    
    text_model.print_trainable_parameters()
    dna_model.print_trainable_parameters()

    # Fusion model creation only on rank 0
    print(f"[Rank {rank}] fusing models...")
    fusion_model = DNATextFusion(
        dna_model=dna_model,
        text_model=text_model,
    )

    # Create DeepSpeed configuration
    ds_config = create_deepspeed_config(
        config=config,
        world_size=world_size
    )

    # Ensure all processes are synchronized before DeepSpeed initialization
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # DeepSpeed initialization with model distribution
    if local_rank <= 0:
        print("🔧 Initializing DeepSpeed model engine...")
    
    model_engine, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=fusion_model,
        model_parameters=fusion_model.parameters(),
    )
    
    # For multi-GPU training, ensure model parameters are properly distributed
    if world_size > 1 and torch.distributed.is_initialized():
        # DeepSpeed automatically handles parameter broadcasting from rank 0
        torch.distributed.barrier()  # Sync all processes
        if local_rank <= 0:
            print(f"[Rank {rank}] Model parameters distributed across {world_size} GPUs")

    train_loader, test_loader, inference_loader = get_data_loaders(
        config, rank=rank, world_size=world_size
    )

    # Set device based on local rank
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)

    # Update config runtime references
    config.deepspeed_runtime = {
        "model_engine": model_engine,
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "dna_tokenizer": dna_tokenizer,
        "text_tokenizer": text_tokenizer,
        "train_loader": train_loader,  
        "inference_loader": inference_loader,
        "test_loader": test_loader     
    }

    return