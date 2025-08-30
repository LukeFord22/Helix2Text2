"""Dataset creation and processing for DNA-Text fusion models with separate DNA lookup."""

import torch
import torch.nn.functional as F
import gzip
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data_list, dna_lookup, config):
        """
        Initialize dataset with text data and DNA lookup.
        
        Args:
            tokenized_data_list: List of text samples with DNA part ID references
            dna_lookup: Dictionary mapping part IDs to tokenized DNA data
        """
        self.data = tokenized_data_list
        self.dna_lookup = dna_lookup
        self.max_length = config.model.dna_max_length
        
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Get DNA part IDs for this sample
        dna_part_ids = entry.get('dna_part_ids', [])
        
        # Look up DNA sequences from the DNA lookup dictionary
        dna_input_ids = []
        
        for part_id in dna_part_ids:
            if part_id in self.dna_lookup:
                dna_data = self.dna_lookup[part_id]
                dna_input_ids.append(dna_data['input_ids'])
            else:
                # Handle missing DNA parts with empty tensors
                print(f"Warning: DNA part '{part_id}' not found in lookup")
                # Create empty tensors with same shape as expected
                # Get expected DNA length from first available DNA part
                expected_length = 1024  # Default DNA max length
                if self.dna_lookup:
                    first_dna = next(iter(self.dna_lookup.values()))
                    expected_length = len(first_dna['input_ids'])
                
                empty_ids = torch.zeros(expected_length, dtype=torch.long)
                dna_input_ids.append(empty_ids)

        
        return {
            "dna_input_ids": dna_input_ids,
            "text_input_ids": entry['text_input_ids'],
            "dna_mask_positions": entry.get('dna_mask_positions', []),  # Precomputed DNA mask positions
            "dna_part_ids": entry.get('dna_part_ids', [])  # Include part IDs for training
        }
        
    def __len__(self):
        return len(self.data)

def get_data_loaders(config, rank=0, world_size=1):
    # Load tokenized data from separate paths
    tokenized_train_path = config.data.train_data
    tokenized_inference_path = config.data.inference_data
    tokenized_dna_path = config.data.dna_data
    
    print(f"Loading training data from {tokenized_train_path}")
    with gzip.open(tokenized_train_path, 'rb') as f:
        train_data = torch.load(f)
    
    print(f"Loading test data from {tokenized_inference_path}")
    with gzip.open(tokenized_inference_path, 'rb') as f:
        inference_data = torch.load(f)
    
    print(f"Loading DNA lookup from {tokenized_dna_path}")
    with gzip.open(tokenized_dna_path, 'rb') as f:
        dna_lookup = torch.load(f)
    
    train_data_fraction = getattr(config.training, 'train_data_fraction', 1.0)
    train_split_fraction = getattr(config.training, 'train_split_fraction', 0.9)
    
    # Sample training data if needed
    total_train_size = len(train_data)
    sampled_train_size = int(total_train_size * train_data_fraction)
    
    if sampled_train_size < total_train_size:
        random.seed(42)
        train_data = random.sample(train_data, sampled_train_size)
    
    # Split data into train/test (90%/10%)
    random.seed(42)
    shuffled_data = train_data.copy()
    random.shuffle(shuffled_data)
    
    split_idx = int(len(shuffled_data) * train_split_fraction)
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:] 

    print(f"Total training dataset size: {total_train_size}")
    print(f"Sampled training dataset size: {len(train_data)}")
    print(f"Sampled test dataset size: {len(test_data)}")
    print(f"Inference dataset size: {len(inference_data)}")
    print(f"DNA lookup size: {len(dna_lookup)} unique parts")
    
    if len(train_data) == 0:
        raise ValueError("Training dataset is empty after sampling")
    if len(test_data) == 0:
        raise ValueError("Training dataset is empty after sampling")
    if len(inference_data) == 0:
        raise ValueError("Test dataset is empty")
    if len(dna_lookup) == 0:
        raise ValueError("DNA lookup is empty")

    train_dataset = TokenizedDataset(train_data, dna_lookup, config=config)
    test_dataset = TokenizedDataset(test_data, dna_lookup, config=config)
    inference_dataset = TokenizedDataset(inference_data, dna_lookup, config=config)

    # Use DistributedSampler for multi-GPU training
    train_sampler = None
    test_sampler = None
    inference_sampler = None
    shuffle = True
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        inference_sampler = DistributedSampler(inference_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False  # Don't shuffle when using DistributedSampler
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=inference_sampler,
        pin_memory=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    return train_loader, test_loader, inference_loader

def collate_fn(batch):
    """
    Custom collate function to handle variable-length DNA sequences.
    Flash attention handles padding and masking natively - no explicit masks needed.
    
    Args:
        batch: List of samples from TokenizedDataset
        
    Returns:
        Batched data with proper structure for DNA-Text fusion model
    """
    # Extract components from batch - pad text sequences to same length
    text_sequences = [item['text_input_ids'] for item in batch]
    max_text_len = max(seq.size(0) for seq in text_sequences)
    
    # Pad text sequences to max length
    padded_text_sequences = []
    for seq in text_sequences:
        pad_len = max_text_len - seq.size(0)
        if pad_len > 0:
            padded_seq = F.pad(seq, (0, pad_len), value=0)  # Pad with 0 (padding token)
        else:
            padded_seq = seq
        padded_text_sequences.append(padded_seq)
    
    text_input_ids = torch.stack(padded_text_sequences)
    
    # Handle variable-length DNA sequences per sample - no attention masks needed
    batch_dna_input_ids = []
    
    for item in batch:
        sample_dna_ids = item['dna_input_ids']
        # Keep DNA sequences as lists - flash attention will handle padding natively
        batch_dna_input_ids.append(sample_dna_ids)
    
    # Extract dna_part_ids and mask positions for training
    batch_dna_part_ids = [item.get('dna_part_ids', []) for item in batch]
    batch_dna_mask_positions = [item.get('dna_mask_positions', []) for item in batch]
    
    return {
        'dna_input_ids': batch_dna_input_ids,
        'text_input_ids': text_input_ids,
        'dna_mask_positions': batch_dna_mask_positions,
        'dna_part_ids': batch_dna_part_ids
    }
