# Helix2Text

## Overview
Helix2Text is a project designed to integrate DNA sequence data with natural language descriptions. It leverages advanced machine learning models, such as METAGENE-1-BnB-4Bit, DNABERT-2, and BioGPT, to process DNA sequences and generate meaningful text annotations or vice versa. This project is particularly useful for bioinformatics applications, such as annotating genetic data, generating textual descriptions of DNA features, and facilitating the understanding of genomic information.

## Features
- **DNA-to-Text Generation**: Converts DNA sequences into descriptive text using transformer-based models.
- **Text-to-DNA Generation**: Generates DNA sequences based on textual descriptions.
- **Integration of Multiple Models**: Combines models like METAGENE-1-BnB-4Bit, DNABERT-2, and BioGPT for efficient processing of biological data.
- **Support for Quantized Models**: Utilizes 4-bit quantization for efficient computation on large-scale datasets.
- **Customizable Training and Inference**: Provides flexibility for fine-tuning and inference on custom datasets.
- **Toxicity Prediction**: Ability to analyze DNA sequences for potential toxic properties.
- **Bidirectional Translation**: Supports both DNA→Text and Text→DNA with multiple model architectures.
- **User-Friendly GUI Applications**: Includes graphical user interfaces for both model inference and training.
- **Batch Processing**: Support for processing multiple DNA sequences or text descriptions in batch mode.
- **Dataset Management**: Tools for handling and preparing training datasets.

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

## GUI Applications

### Helix2Text GUI (helix2text_gui.py)
A graphical user interface for using the Helix2Text model to generate text from DNA sequences and vice versa.

Features:
- Single generation mode for converting individual DNA sequences or text
- Batch processing mode for handling multiple inputs at once
- Dataset management functionality
- Model loading and configuration options
- Results display, copying, and saving

### Helix2Text Training GUI (helix2text_train_gui.py)
A graphical interface for training the Helix2Text model without using the terminal.

Features:
- Configuration management with loading and saving options
- Training data selection and parameter configuration
- Real-time training progress monitoring with metrics visualization
- Dataset management and preparation tools
- Export of training metrics and plots

## MetaModel
The MetaModel directory contains a modular and well-structured implementation of the Helix2Text system with the following components:

### Key Components
- **DNA-Text Fusion Model**: Core architecture combining DNA and text modalities
- **Cross-Attention Mechanisms**: For fusing information between modalities
- **Contrastive Losses**: For aligning DNA and text representations
- **Training and Evaluation**: Complete pipeline for model training and testing
- **Configuration System**: Flexible configuration using YAML files

## Models Used
- **METAGENE-1-BnB-4Bit**: A 4-bit quantized DNA language model trained on genomic sequences.
- **DNABERT-2**: A specialized transformer-based model for DNA sequence analysis.
- **BioGPT**: A language model specialized for biomedical text generation, replacing the previously used DistilGPT2.

## Datasets
The project utilizes several specialized datasets:
- **NCBI Sequence Data**: DNA sequences obtained from the National Center for Biotechnology Information.
- **Protein to DNA Sequence Descriptions**: Mapping between DNA sequences and protein descriptions used for training.
- **Toxin and Bioactive Protein Annotations**: Data relating DNA sequences to toxins and bioactive proteins from various organisms.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pre-trained models:
   - METAGENE-1-BnB-4Bit can be downloaded from Hugging Face
   - DNABERT-2 and BioGPT will be automatically downloaded through the Transformers library
   
4. Send Data:
```Bash
# runpod
runpodctl config --apiKey rpa_SOO146XCJMOTK6HOLXTJ9WLU7YPG2I3V17BVNXBFxhdz9x
runpodctl send Helix2Text
./runpodctl.exe receive

# croc
curl https://getcroc.schollz.com | bash
nix-env -i croc
croc send Gene.pth
```

## Usage

### GUI Application
To launch the graphical user interface for inference:
```bash
python helix2text_gui.py
```

To launch the training GUI:
```bash
python helix2text_train_gui.py
```

### Command Line (Using MetaModel Scripts)
For training:
```bash
python -m MetaModel.scripts.train --config_path path/to/config.yaml
```

For evaluation:
```bash
python -m MetaModel.scripts.evaluate --model_path best_model.pt --data_path test_data.csv
```

For generation:
```bash
python -m MetaModel.scripts.generate --model_path best_model.pt --input "ATGCGTACCTAGGTC..."
```

### Programmatic API
```python
# Import the model
from MetaModel.models.dna_text_fusion import load_models, DNATextFusion

# Load tokenizers and model components
dna_tokenizer, text_tokenizer, dna_model, text_model = load_models(
    dna_model_path="path/to/dna/model",
    text_model_path="path/to/text/model"
)

# Initialize the fusion model
model = DNATextFusion(
    dna_model=dna_model,
    text_model=text_model,
    cross_attn_interval=4
)

# Load trained weights
import torch
checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Process a DNA sequence
dna_sequence = "ATGCGTACCTAGGTC..."  # Your DNA sequence
# Tokenize
dna_inputs = dna_tokenizer(
    dna_sequence, 
    return_tensors="pt",
    padding=True,
    truncation=True
)
# Generate text
generated_ids = model.generate_text_from_dna(dna_inputs, max_length=128)
generated_text = text_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
```

## Model Architecture
Helix2Text implements two distinct fusion architectures:

1. **METAGENE + BioGPT Fusion**:
   - Uses METAGENE-1-BnB-4Bit for DNA sequence encoding
   - Leverages BioGPT for biomedical text generation
   - Employs gated cross-attention to fuse the two modalities
   
2. **DNABERT-2 Based Fusion**:
   - Uses DNABERT-2 for DNA sequence processing
   - Integrates with language modeling capabilities
   - Projects between DNA and text representation spaces

Both architectures enable bidirectional translation between DNA sequences and human-readable text descriptions, with metrics tracked using Weights & Biases.

## Performance Metrics
The models are evaluated using multiple metrics:
- BLEU scores for measuring text generation quality
- ROUGE scores for assessing text similarity
- Contrastive loss for semantic alignment between modalities

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License
This software is proprietary and confidential. Unauthorized copying, use, modification, or distribution of this software is strictly prohibited. All rights reserved.

See the LICENSE file for full terms and conditions.