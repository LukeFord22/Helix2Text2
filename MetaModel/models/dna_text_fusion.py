"""DNA-Text fusion model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FlashAttentionPool
from .components import AdvancedFusionBlock, EfficientModalityAdapter, EfficientJointProjection, CrossModalityTransformerBlock

class DNATextFusion(nn.Module):
    """
    DNA-Text fusion model for bidirectional generation.
    
    This model combines DNA and text encoders with cross-attention
    to enable bidirectional generation and retrieval.
    """

# ================================================================================================================================
#                                                                                                                   Initialization
# ================================================================================================================================

    def __init__(self, dna_model, text_model):
        super().__init__()
        self.dna_model = dna_model
        self.text_model = text_model
        
        # Get model dimensions
        dna_hidden = dna_model.config.hidden_size
        text_hidden = text_model.config.hidden_size
            
        # Handle PEFT/LoRA model structure
        if hasattr(text_model, 'base_model'):
            base_text_model = text_model.base_model
            if hasattr(base_text_model, 'base_model'):
                base_text_model = base_text_model.base_model
        else:
            base_text_model = text_model
            
        if hasattr(dna_model, 'base_model'):
            base_dna_model = dna_model.base_model
            if hasattr(base_dna_model, 'base_model'):
                base_dna_model = base_dna_model.base_model
        else:
            base_dna_model = dna_model
        
        # Store base models for encoding (to avoid LoRA dtype issues)
        self.base_dna_model = base_dna_model
        self.base_text_model = base_text_model
        
        # Reference model heads without creating new copies
        self.norm = base_text_model.norm
        self.text_lm_head = text_model.lm_head  # Use PEFT-wrapped head
        self.dna_lm_head = dna_model.lm_head    # Use PEFT-wrapped head

        # Enable gradient checkpointing for encoding efficiency
        self.text_model.gradient_checkpointing_enable()
        self.dna_model.gradient_checkpointing_enable()
        
        # Advanced modality-specific adapters with parallel FFN
        self.dna_adapter = EfficientModalityAdapter(
            dim=dna_hidden, 
            expansion_factor=2.0,
        )
        self.text_adapter = EfficientModalityAdapter(
            dim=text_hidden, 
            expansion_factor=2.0,
        )
        
        # Efficient cross-modality projections with bottleneck architecture
        bottleneck_dim = min(dna_hidden, text_hidden) // 2
        
        self.text_to_dna_proj = nn.Sequential(
            nn.Linear(text_hidden, bottleneck_dim, bias=False),
            nn.GELU(), 
            nn.Linear(bottleneck_dim, dna_hidden, bias=False)
        )  

        # Flash Attention v2 based pooling layers
        self.text_flash_pool = FlashAttentionPool(embed_dim=text_hidden, num_heads=8)
        self.dna_flash_pool = FlashAttentionPool(embed_dim=dna_hidden, num_heads=8)

        # Efficient DNA projection for joint embedding with bottleneck
        joint_bottleneck = text_hidden // 2
        self.project_dna_for_joint = nn.Sequential(
            nn.Linear(dna_hidden, joint_bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(joint_bottleneck, text_hidden, bias=False)
        )
        
        # Advanced joint projection with multiple pathways and attention
        self.joint_projection = EfficientJointProjection(
            input_dim=text_hidden * 2,
            output_dim=text_hidden,
            num_heads=8,
        )

        # Symmetric cross-attention blocks for bidirectional fusion (reduced for memory)
        # DNA-to-Text blocks: DNA attends to text context
        self.dna_to_text_blocks = nn.ModuleList([
            CrossModalityTransformerBlock(
                dim=dna_hidden,
                cross_dim=text_hidden,
                n_heads=8
            )
            for _ in range(2)  # Reduced from 6 to 2 for memory efficiency
        ])
        
        # Text-to-DNA blocks: Text attends to DNA context  
        self.text_to_dna_blocks = nn.ModuleList([
            CrossModalityTransformerBlock(
                dim=text_hidden,
                cross_dim=dna_hidden,
                n_heads=8
            )
            for _ in range(2)  # Reduced from 6 to 2 for memory efficiency
        ])
        
        
        # Advanced fusion block with cross-attention and multiplicative interactions
        self.fusion_block = AdvancedFusionBlock(
            hidden_dim=text_hidden,
            input_dim_x=text_hidden,
            input_dim_y=dna_hidden,
            num_heads=8,
            compression_ratio=4
        )
        

# ================================================================================================================================
#                                                                                                                          Forward
# ================================================================================================================================

    def forward(self, dna_inputs, text_inputs, dna_mask_positions=None):
        """
        Forward pass for DNA-Text fusion model with multiple DNA sequence support
        
        Args:
            dna_inputs: Dictionary with lists of DNA input_ids 
            text_inputs: Dictionary with text input_ids 
            dna_mask_positions: precomputed DNA generation positions per batch
            
        Returns:
            Dictionary containing model outputs
        """
        # STEP 1: INITIAL ENCODING - Convert raw tokens to contextual embeddings
        
        # Efficient batched DNA processing
        device = text_inputs["input_ids"].device
        batch_size = len(dna_inputs["dna_input_ids"])
        dna_hidden_size = self.dna_model.config.hidden_size
        
        # We'll build the contexts dynamically due to variable sequence lengths
        batch_dna_contexts = []
        batch_dna_sequence_contexts = []
        
        # Collect all valid DNA sequences for batched processing
        all_sequences = []
        sequence_batch_mapping = []  # Maps sequence index to batch index
        sequence_sample_mapping = []  # Maps sequence index to sample within batch
        
        for batch_idx, sample_dna_ids in enumerate(dna_inputs["dna_input_ids"]):
            sample_sequence_indices = []
            
            if sample_dna_ids and len(sample_dna_ids) > 0:
                for sample_idx, dna_ids in enumerate(sample_dna_ids):
                    if len(dna_ids) > 0:  # Skip empty sequences
                        # Ensure tensors are on correct device (single transfer per sequence)
                        if dna_ids.device != device:
                            dna_ids = dna_ids.to(device)
                        
                        all_sequences.append(dna_ids)
                        sequence_batch_mapping.append(batch_idx)
                        sequence_sample_mapping.append(sample_idx)
                        sample_sequence_indices.append(len(all_sequences) - 1)
                        
            batch_dna_sequence_contexts.append(sample_sequence_indices)
        
        # Process all sequences in optimized batches
        if all_sequences:
            # Determine optimal batch size for GPU memory
            max_batch_size = min(32, len(all_sequences))  # Process up to 32 sequences at once
            sequence_contexts = []
            
            for i in range(0, len(all_sequences), max_batch_size):
                batch_end = min(i + max_batch_size, len(all_sequences))
                batch_sequences = all_sequences[i:batch_end]
                
                # Pad sequences to same length within batch for efficient processing
                max_seq_len = max(seq.size(0) for seq in batch_sequences)
                
                batched_ids = []
                
                for dna_ids in batch_sequences:
                    # Pad to max length with zeros - flash attention will handle natively
                    pad_len = max_seq_len - dna_ids.size(0)
                    if pad_len > 0:
                        dna_ids = F.pad(dna_ids, (0, pad_len), value=0)
                    
                    batched_ids.append(dna_ids)
                
                # Stack into efficient batch tensors
                batch_input_ids = torch.stack(batched_ids)
                
                # Single batched encode call - flash attention handles padding natively
                batch_contexts = self.encode_dna({
                    "input_ids": batch_input_ids,
                })
                
                # Store results
                for context in batch_contexts:
                    sequence_contexts.append(context)
            
            # Distribute contexts back to batch samples
            for batch_idx in range(batch_size):
                sample_indices = batch_dna_sequence_contexts[batch_idx]
                if sample_indices:
                    # Average all sequences for this sample
                    sample_contexts = [sequence_contexts[idx] for idx in sample_indices]
                    avg_context = torch.stack(sample_contexts).mean(dim=0)
                    # Ensure consistent shape
                    if len(avg_context.shape) == 2:
                        avg_context = avg_context.unsqueeze(0)  # Add batch dimension
                    batch_dna_contexts.append(avg_context)
                    # Store individual contexts for position mapping
                    batch_dna_sequence_contexts[batch_idx] = sample_contexts
                else:
                    # Create zero context with consistent shape
                    zero_context = torch.zeros(1, 1, dna_hidden_size, device=device)
                    batch_dna_contexts.append(zero_context)
                    batch_dna_sequence_contexts[batch_idx] = []
        else:
            # All samples have empty DNA sequences - create zero contexts for all
            for _ in range(batch_size):
                zero_context = torch.zeros(1, 1, dna_hidden_size, device=device)
                batch_dna_contexts.append(zero_context)
            batch_dna_sequence_contexts = [[] for _ in range(batch_size)]
        
        # Stack all contexts into a single tensor for efficient processing
        dna_context = torch.cat(batch_dna_contexts, dim=0)  # [batch_size, seq_len, hidden_dim]
        individual_dna_contexts = batch_dna_sequence_contexts
        
        # Encode text inputs -> [batch_size, text_seq_len, text_hidden_dim]  
        text_context = self.encode_text(text_inputs)

        # STEP 2: SYMMETRIC CROSS-MODAL ATTENTION - Each modality attends to the other
        # DNA-to-Text Cross-Attention: DNA representations enhanced by text context
        enhanced_dna = dna_context  # Initialize with original DNA context
        for block in self.dna_to_text_blocks:
            # Each block: DNA queries attend to text keys/values
            # Output: DNA representations informed by text semantics
            enhanced_dna = block(enhanced_dna, text_context)
            
        # Text-to-DNA Cross-Attention: Text representations enhanced by DNA context  
        enhanced_text = text_context  # Initialize with original text context
        for block in self.text_to_dna_blocks:
            # Each block: Text queries attend to DNA keys/values
            # Output: Text representations informed by DNA structure
            enhanced_text = block(enhanced_text, enhanced_dna)

        # STEP 3: FLASH ATTENTION V2 POOLING - Convert sequences to fixed-size representations
        # Pool enhanced text to single vector per sample using Flash Attention v2
        text_pooled = self.text_flash_pool(
            enhanced_text, 
        )  # [batch_size, text_hidden_dim]
        
        # Pool enhanced DNA to single vector per sample using Flash Attention v2
        dna_pooled = self.dna_flash_pool(
            enhanced_dna,
        )  # [batch_size, dna_hidden_dim]

        # STEP 4: MULTIMODAL FUSION - Combine modalities for downstream tasks
        # Project DNA pooled vector to text dimensions for joint embedding
        dna_pooled_proj = self.project_dna_for_joint(dna_pooled)  # [batch_size, text_hidden_dim]
        
        # Learnable fusion of text and DNA pooled representations
        fused_representation = self.fusion_block(text_pooled, dna_pooled)  # [batch_size, text_hidden_dim]
        
        # Contrastive joint embedding: concatenate then project
        joint_input = torch.cat([text_pooled, dna_pooled_proj], dim=-1)  # [batch_size, 2*text_hidden_dim]
        joint_embedding = self.joint_projection(joint_input)  # [batch_size, text_hidden_dim]

        # STEP 5: GENERATION LOGITS - Compute next-token predictions for both modalities
        # Text generation: enhanced text -> next text tokens
        text_logits = self.text_lm_head(self.norm(enhanced_text))  # [batch_size, text_seq_len, text_vocab_size]
        
        # Use precomputed DNA mask positions if provided, otherwise no dna in text
        batch_size = text_inputs["input_ids"].size(0)
        all_mask_positions = []
        dna_positions = dna_mask_positions if dna_mask_positions is not None else []
        if dna_mask_positions is not None:
            # Use precomputed positions from tokenizer/data loader
            for batch_idx, positions in enumerate(dna_positions):
                for pos in positions:
                    all_mask_positions.append((batch_idx, pos))

        # DNA sequence generation - Memory efficient computation at DNA mask positions
        dna_logits = None
        if all_mask_positions:
            batch_size, seq_len, _ = enhanced_text.shape
            dna_vocab_size = self.dna_lm_head.out_features
            
            # Create sparse tensor only if we have positions (much smaller than before)
            # Only allocate memory for positions that actually exist
            unique_positions = list(set(all_mask_positions))
            
            if unique_positions:
                # Initialize minimal sparse tensor - only create it if we have valid positions
                dna_logits = torch.zeros(batch_size, seq_len, dna_vocab_size, 
                                       device=enhanced_text.device, dtype=enhanced_text.dtype)
                
                # Process each mask position individually to save intermediate memory
                for batch_idx, pos_idx in unique_positions:
                    if pos_idx < seq_len:
                        # Extract single position hidden state
                        mask_hidden = enhanced_text[batch_idx, pos_idx, :]  # [text_hidden_dim]
                        
                        # Project to DNA space
                        dna_repr = self.text_to_dna_proj(mask_hidden.unsqueeze(0))  # [1, dna_hidden_dim]
                        
                        # Compute DNA logits
                        dna_logit = self.dna_lm_head(dna_repr)  # [1, dna_vocab_size]
                        
                        # Store directly in sparse tensor at specific position
                        dna_logits[batch_idx, pos_idx, :] = dna_logit.squeeze(0)
                        
                        # Clear intermediate tensors immediately
                        del mask_hidden, dna_repr, dna_logit
        
        # Convert to format expected by training loop
        if dna_positions and any(len(positions) > 0 for positions in dna_positions):
            dna_positions_output = dna_positions
        else:
            dna_positions_output = None
        
        # STEP 6: OUTPUT ASSEMBLY - Package all computed representations
        result = {
            "text_logits": text_logits,          # For text generation loss
            "dna_logits": dna_logits,            # For DNA generation loss at DNA mask positions
            "dna_positions": dna_positions_output, # Positions where DNA mask tokens are found
            "individual_dna_contexts": individual_dna_contexts,  # Individual DNA contexts for position mapping
            "joint_embedding": joint_embedding,   # For contrastive learning
            "fused_representation": fused_representation,  # For classification tasks
            "text_pooled": text_pooled,          # Text summary vector
            "dna_pooled": dna_pooled,            # DNA summary vector
        }

        # STEP 7: MEMORY CLEANUP - Free intermediate tensors to prevent OOM
        # Selective cleanup - only delete large intermediate tensors
        large_intermediate_vars = [enhanced_text, enhanced_dna, text_context, 
                                 joint_input]
        for var in large_intermediate_vars:
            if var is not None:
                del var
        
        # Less frequent cache clearing to reduce synchronization overhead
        # Only clear cache if we processed a large number of sequences
        if torch.cuda.is_available() and len(individual_dna_contexts) > 16:
            torch.cuda.empty_cache()

        return result
        
# ================================================================================================================================
#                                                                                                                        Encoding
# ================================================================================================================================


    def encode_dna(self, dna_inputs):
        """Process DNA inputs through the DNA model with optimized batched processing."""
        input_ids = dna_inputs["input_ids"]
        
        # Handle empty sequences
        if input_ids.numel() == 0:
            batch_size = max(1, input_ids.size(0) if len(input_ids.shape) > 1 else 1)
            seq_len = 1
            hidden_size = self.dna_model.config.hidden_size
            device = next(self.base_dna_model.parameters()).device
            return torch.zeros(batch_size, seq_len, hidden_size, device=device)
        
        # Pre-compute model dtype and device to avoid repeated lookups
        model_dtype = next(self.base_dna_model.parameters()).dtype
        model_device = next(self.base_dna_model.parameters()).device
        
        # Ensure inputs are on correct device and dtype (batch operation)
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
            
        # Use optimized autocast with consistent dtype
        with torch.autocast(device_type='cuda', dtype=model_dtype, enabled=torch.cuda.is_available()):
            dna_outputs = self.dna_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False  # Disable KV cache for memory efficiency in batch processing
            )

        dna_hidden = dna_outputs.hidden_states[-1]
        #dna_outputs.last_hidden_state.mean(dim=1) could be used without output_hidden_states=ture?
        
        # Clear intermediate outputs immediately
        del dna_outputs
        
        # Pre-compute adapter dtype to avoid repeated lookups
        adapter_dtype = next(self.dna_adapter.parameters()).dtype
        
        # Efficient dtype conversion only if necessary
        if dna_hidden.dtype != adapter_dtype:
            dna_hidden = dna_hidden.to(dtype=adapter_dtype)
        
        # Apply adapter with memory-efficient processing
        adapted_dna = self.dna_adapter(dna_hidden)
        
        # Clear intermediate tensors
        del dna_hidden
        
        return adapted_dna
    
    def encode_text(self, text_inputs):
        """Encode text inputs with optimized processing."""
        input_ids = text_inputs["input_ids"]
        
        # Pre-compute model dtype and device to avoid repeated lookups
        model_dtype = next(self.base_text_model.parameters()).dtype

        # Use optimized autocast
        with torch.autocast(device_type='cuda', dtype=model_dtype, enabled=torch.cuda.is_available()):
            text_outputs = self.text_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False  # Disable KV cache for memory efficiency
            )

        text_hidden = text_outputs.hidden_states[-1]
        
        # Clear intermediate outputs
        del text_outputs
        
        # Pre-compute adapter dtype
        adapter_dtype = next(self.text_adapter.parameters()).dtype
        
        # Efficient dtype conversion only if necessary
        if text_hidden.dtype != adapter_dtype:
            text_hidden = text_hidden.to(dtype=adapter_dtype)
            
        adapted_text = self.text_adapter(text_hidden)
        
        # Clear intermediate tensors
        del text_hidden
        
        return adapted_text
    
    
# ================================================================================================================================
#                                                                                                            Multimodal generation
# ================================================================================================================================

    @torch.no_grad()
    def generate_unified(self, text_inputs, text_tokenizer, dna_tokenizer, max_length=512, temperature=0.8, top_k=50, top_p=0.9, dna_max_length=128, dna_inputs=None, dna_mask_positions=None):
        """
        Unified generation with DNA mask routing: text model generates until <|reserved_200010|> then routes to DNA model.
        
        Args:
            text_inputs: Dictionary with text input_ids
            text_tokenizer: Text tokenizer for token handling
            dna_tokenizer: DNA tokenizer for DNA sequence generation
            max_length: Maximum total sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            dna_max_length: Maximum DNA sequence length when routing
            dna_inputs: Optional DNA inputs from data loader (for multimodal context)
            dna_mask_positions: Optional precomputed DNA mask positions from data loader
            
        Returns:
            Dictionary with generated sequences and routing info
        """
        self.eval()
        device = text_inputs["input_ids"].device
        batch_size = text_inputs["input_ids"].size(0)
        
        # Token IDs for DNA routing
        DNA_MASK_TOKEN = 200010  # <|reserved_200010|>
        
        # Initialize with input context
        input_ids = text_inputs["input_ids"].clone()
        
        # Determine if we have multimodal context from data loader
        has_dna_context = dna_inputs is not None and any(
            len(sample_dna) > 0 and any(len(seq) > 0 for seq in sample_dna) 
            for sample_dna in dna_inputs.get("dna_input_ids", [])
        )
        
        # If we have DNA context, get initial multimodal representations ONCE
        enhanced_initial_context = None
        if has_dna_context:
            # Get enhanced representations using full forward pass ONCE
            initial_outputs = self.forward(
                dna_inputs=dna_inputs,
                text_inputs=text_inputs,
                dna_mask_positions=dna_mask_positions
            )
            # Extract enhanced text representations for conditioning subsequent generation
            enhanced_initial_context = initial_outputs.get('text_logits', None)  # [batch_size, seq_len, vocab_size]
        
        # Track generation state
        curr_length = input_ids.size(1)
        past_key_values = None
        unfinished_sequences = torch.ones(batch_size, device=device, dtype=torch.long)
        
        # Track DNA routing events
        dna_routing_events = []
        generated_dna_sequences = []
        
        while curr_length < max_length and unfinished_sequences.max() > 0:
            # For first step with DNA context, use enhanced logits from initial forward pass
            if curr_length == input_ids.size(1) and enhanced_initial_context is not None:
                # Use enhanced logits from the initial multimodal forward pass
                next_token_logits = enhanced_initial_context[:, -1, :] / temperature  # Last position logits
            else:
                # Standard text generation for subsequent steps
                position_ids = torch.arange(0, input_ids.size(1), device=device).unsqueeze(0)
                
                text_outputs = self.base_text_model(
                    input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                    position_ids=position_ids[:, -1:] if past_key_values is not None else position_ids,
                    output_hidden_states=True,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                
                hidden_states = text_outputs.hidden_states[-1]
                past_key_values = text_outputs.past_key_values
                
                # Get next token logits
                next_token_hidden = self.norm(hidden_states[:, -1, :])
                next_token_logits = self.text_lm_head(next_token_hidden) / temperature
            
            # Apply sampling filters
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for DNA mask token routing
            dna_mask_positions = (next_token == DNA_MASK_TOKEN).nonzero(as_tuple=False)
            
            if len(dna_mask_positions) > 0:
                # DNA routing triggered
                for batch_idx, _ in dna_mask_positions:
                    batch_idx = batch_idx.item()
                    
                    # Create context for DNA generation (current sequence up to this point)
                    current_context = input_ids[batch_idx:batch_idx+1]
                    
                    # Generate DNA sequence using DNA model - flash attention handles masking
                    dna_sequence = self._generate_dna_sequence(
                        current_context, dna_tokenizer, 
                        max_length=dna_max_length, temperature=temperature, 
                        top_k=top_k, top_p=top_p
                    )
                    
                    # Replace DNA mask token with actual DNA sequence
                    # Remove BOS/EOS tokens from DNA sequence if present
                    dna_clean = dna_sequence[1:-1] if dna_sequence.size(1) > 2 else dna_sequence
                    
                    # Store routing event
                    dna_routing_events.append({
                        'batch_idx': batch_idx,
                        'position': curr_length,
                        'dna_sequence': dna_clean.clone()
                    })
                    generated_dna_sequences.append(dna_clean)
                    
                    # Update the next token to be the first DNA token instead of mask
                    if dna_clean.size(1) > 0:
                        next_token[batch_idx] = dna_clean[0, 0]
            
            # Check for EOS and update unfinished sequences
            unfinished_sequences = unfinished_sequences.mul((next_token != text_tokenizer.eos_token_id).long())
            
            # Append next token - flash attention handles sequence extension natively
            input_ids = torch.cat([input_ids, next_token], dim=1)
            curr_length += 1
            
            # If we just added a DNA token, we need to continue adding the rest of the DNA sequence
            for event in dna_routing_events:
                if event['position'] == curr_length - 1 and event['dna_sequence'].size(1) > 1:
                    batch_idx = event['batch_idx']
                    remaining_dna = event['dna_sequence'][0, 1:]  # Skip first token (already added)
                    
                    # Add remaining DNA tokens
                    for dna_token in remaining_dna:
                        if curr_length < max_length:
                            dna_token_tensor = torch.full((batch_size, 1), text_tokenizer.pad_token_id, device=device, dtype=torch.long)
                            dna_token_tensor[batch_idx, 0] = dna_token
                            input_ids = torch.cat([input_ids, dna_token_tensor], dim=1)
                            curr_length += 1
        
        return {
            'generated_ids': input_ids,
            'dna_routing_events': dna_routing_events,
            'generated_dna_sequences': generated_dna_sequences
        }
    
    def _generate_dna_sequence(self, text_context, dna_tokenizer, max_length=128, temperature=0.8, top_k=50, top_p=0.9):
        """Helper method to generate DNA sequence conditioned on text context."""
        device = text_context.device
        
        # Encode text context for DNA conditioning - flash attention handles masking natively
        text_context_encoded = self.encode_text({
            "input_ids": text_context,
        })
        
        # Start DNA generation with BOS token - flash attention handles masking natively
        dna_input_ids = torch.full((1, 1), dna_tokenizer.bos_token_id or 3, device=device, dtype=torch.long)
        
        dna_past_key_values = None
        dna_length = 1
        
        while dna_length < max_length:
            # DNA model forward pass
            position_ids = torch.arange(0, dna_input_ids.size(1), device=device).unsqueeze(0)
            
            dna_outputs = self.base_dna_model(
                input_ids=dna_input_ids[:, -1:] if dna_past_key_values is not None else dna_input_ids,
                position_ids=position_ids[:, -1:] if dna_past_key_values is not None else position_ids,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=dna_past_key_values
            )
            
            dna_hidden = dna_outputs.hidden_states[-1]
            dna_past_key_values = dna_outputs.past_key_values
            
            # Apply text conditioning
            text_context_pooled = text_context_encoded.mean(dim=1, keepdim=True)
            text_proj = self.text_to_dna_proj(text_context_pooled)
            dna_hidden = dna_hidden + text_proj
            
            # Get next DNA token
            if hasattr(self.base_dna_model, 'norm'):
                next_dna_hidden = self.base_dna_model.norm(dna_hidden[:, -1, :])
            else:
                next_dna_hidden = dna_hidden[:, -1, :]
            
            next_dna_logits = self.dna_lm_head(next_dna_hidden) / temperature
            
            # Apply sampling
            if top_k > 0:
                top_k = min(top_k, next_dna_logits.size(-1))
                indices_to_remove = next_dna_logits < torch.topk(next_dna_logits, top_k)[0][..., -1, None]
                next_dna_logits = next_dna_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_dna_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_dna_logits = next_dna_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            probs = F.softmax(next_dna_logits, dim=-1)
            next_dna_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            if next_dna_token.item() == (dna_tokenizer.eos_token_id or 4):
                break
                
            dna_input_ids = torch.cat([dna_input_ids, next_dna_token], dim=1)
            dna_length += 1
        
        return dna_input_ids
    