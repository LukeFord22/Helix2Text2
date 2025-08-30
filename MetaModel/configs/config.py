"""Default configuration parameters for DNA-Text fusion models."""

from dataclasses import dataclass, field
from datetime import datetime
import os

os.environ['WANDB_API_KEY'] = 'f9ef5f01cec347e6598e41294b2dce5380dedbc1'

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 1
    num_epochs: int = 1
    learning_rate: float = 2e-4
    grad_accum_steps: int = 4

    train_data_fraction: float = 0.25
    train_split_fraction: float = 0.9  # Default: 90% for training, 10% for evaluation


    
@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""

    dna_model_path: str = "metagene-ai/METAGENE-1"
    text_model_path: str = "openai/gpt-oss-20b"

    text_r: int = 8
    dna_r: int = 16

    dna_max_length: int = 1024

    
@dataclass
class DataConfig:
    """Data processing configuration parameters."""
    train_data: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "MetaModel/data/data/training_data.pt.gz")
    inference_data: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "MetaModel/data/data/inference_data.pt.gz")
    dna_data: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "MetaModel/data/data/dna_lookup.pt.gz")

@dataclass
class Config:
    """Combined configuration for the entire project."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment_name: str = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    project_name: str = "Helix2Text"
    output_path: str = "best_model.pt"
    
# Default configuration
default_config = Config()