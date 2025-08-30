
Directory structure for modular Helix2Text implementation:

## Project Structure
```
Helix2Text/
├── helix2text_gui.py           # Main GUI application for inference
├── helix2text_train_gui.py     # GUI application for training models
├── requirements.txt            # Project dependencies
├── LICENSE                     # License information
├── README.md                   # This file
├── MetaModel/                  # Core modular implementation
│   ├── configs/                # Configuration files
│   │   ├── default_config.py   # Default configuration values
│   │   └── training_config.yaml# YAML configuration for experiments
│   ├── data/                   # Data processing modules
│   │   ├── dataset.py          # Dataset creation and processing
│   │   └── tokenizers.py       # Tokenizer handling and utilities
│   ├── models/                 # Model architecture definitions
│   │   ├── dna_text_fusion.py  # Main fusion model architecture
│   │   ├── attention.py        # Cross-attention mechanisms
│   │   └── components.py       # Reusable model components
│   ├── losses/                 # Loss function implementations
│   │   ├── contrastive.py      # InfoNCE and dual softmax losses
│   │   └── criterion.py        # Combined loss functions
│   ├── training/               # Training utilities
│   │   ├── trainer.py          # Training loop implementation
│   │   └── evaluation.py       # Evaluation metrics and visualization
│   ├── utils/                  # Utility functions
│   │   ├── logging.py          # WandB logging utilities
│   │   └── optimization.py     # Scheduler and optimizer utilities
│   ├── scripts/                # Runnable scripts
│   │   ├── train.py            # Main training script
│   │   ├── evaluate.py         # Standalone evaluation script
│   │   └── generate.py         # Text/DNA generation script
│   └── example_usage.py        # Example usage of the models
└── RAWModels/                  # Original model implementations
    ├── METAGENE-1model.py      # Implementation using METAGENE and BioGPT
    └── DNABERT2model.py        # Implementation using DNABERT-2
```