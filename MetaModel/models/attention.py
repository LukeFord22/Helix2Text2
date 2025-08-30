import torch
import torch.nn as nn
from flash_attn import flash_attn_func


# ================================================================================================================================
#                                                                                                                Cross modality
# ================================================================================================================================

class MQAFlashAttention(nn.Module):
    """Multi-Query Attention block with Flash-style speedup."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = min(embed_dim // num_heads, 256)  # Ensure head_dim <= 256 for FlashAttention

        # Separate projections for queries
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)

        # SHARED projections for keys/values (only ONE key, ONE value per position)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

        self.out_proj = nn.Linear(self.num_heads * self.head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        q = self.q_proj(query)  # (B, Lq, H*D)
        k = self.k_proj(key)    # (B, Lk, D)
        v = self.v_proj(value)  # (B, Lk, D)

        # Ensure all tensors have the same dtype (FlashAttention requirement)
        target_dtype = q.dtype
        if target_dtype == torch.float32:
            # FlashAttention doesn't support float32, convert to bfloat16
            target_dtype = torch.bfloat16
        
        q = q.to(dtype=target_dtype)
        k = k.to(dtype=target_dtype)  
        v = v.to(dtype=target_dtype)

        # Reshape for FlashAttention-2: (batch, seqlen, num_heads, head_dim)
        q = q.view(B, Lq, self.num_heads, self.head_dim)
        
        # For MQA: repeat k,v across heads to match q
        k = k.unsqueeze(2).expand(B, Lk, self.num_heads, self.head_dim)
        v = v.unsqueeze(2).expand(B, Lk, self.num_heads, self.head_dim)
        
        # Convert mask to FlashAttention format if provided
        if mask is not None:
            # FlashAttention expects mask as (batch, seqlen_q, seqlen_k) or None
            if mask.dim() == 2:  # (B, Lk) -> (B, Lq, Lk)
                mask = mask.unsqueeze(1).expand(B, Lq, Lk)
            elif mask.dim() == 4:  # (B, 1, 1, Lk) -> (B, Lq, Lk)
                mask = mask.squeeze(1).squeeze(1).expand(B, Lq, Lk)
            # Ensure mask is boolean for FlashAttention
            mask = mask.bool()
        
        # Use FlashAttention-2 
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            window_size=(-1, -1),  # No sliding window
            alibi_slopes=None,
            return_attn_probs=False
        )

        # Reshape output: (B, Lq, H, D) -> (B, Lq, H*D)
        attn_output = attn_output.view(B, Lq, self.num_heads * self.head_dim)
            
        return self.out_proj(attn_output)


class NormalizedGatedCrossAttention(nn.Module):
    """Highly optimized cross-attention block with LayerNorm, projection, MQA, gating."""
    def __init__(self, query_dim, context_dim=None, n_heads=8, dropout=0.0):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim if context_dim is not None else query_dim
        self.n_heads = n_heads

        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_context = nn.LayerNorm(self.context_dim)

        if self.query_dim != self.context_dim:
            self.context_proj = nn.Linear(self.context_dim, query_dim)
        else:
            self.context_proj = nn.Identity()

        # Always use MQA
        self.cross_attn = MQAFlashAttention(
            embed_dim=query_dim,
            num_heads=n_heads,
            dropout=dropout
        )
        
        self.gate = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim),
            nn.Sigmoid()
        )

    def forward(self, query, context, mask=None):
        # Ensure consistent dtypes for all inputs
        target_dtype = query.dtype
        context = context.to(dtype=target_dtype)
        
        q = self.norm_query(query)
        k = self.norm_context(context)
        k = self.context_proj(k)

        out = self.cross_attn(
            query=q,
            key=k,
            value=k,
            mask=mask
        )

        gate_val = self.gate(torch.cat([q, out], dim=-1))
        return query + gate_val * out


# ================================================================================================================================
#                                                                                                                          Pooling
# ================================================================================================================================


class FlashAttentionPool(nn.Module):
    """Flash Attention v2 based pooling layer for sequence-to-vector reduction."""
    
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = min(embed_dim // num_heads, 256)  # Ensure head_dim <= 256 for FlashAttention
        
        # Learnable query parameter
        self.query_param = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Combined projection for Q, K, V - use actual head dimensions
        self.actual_embed_dim = self.num_heads * self.head_dim
        self.qkv_proj = nn.Linear(embed_dim, self.actual_embed_dim * 3, bias=False)
        
        # Output projection - from actual_embed_dim back to original embed_dim
        self.out_proj = nn.Linear(self.actual_embed_dim, embed_dim, bias=False)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, hidden_states):
        """
        Pool variable-length sequences to fixed-size vectors using Flash Attention v2.
        Flash attention handles padding natively without explicit masks.
        
        Args:
            hidden_states: [batch_size, seq_len, embed_dim]
            
        Returns:
            pooled_output: [batch_size, embed_dim]
        """
        batch_size, _, embed_dim = hidden_states.shape
        
        # Expand learnable query to match batch size
        query = self.query_param.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        
        # Ensure dtype consistency
        query = query.to(dtype=hidden_states.dtype, device=hidden_states.device)
        
        # Concatenate query with hidden states for combined processing
        # Query will attend to all hidden states
        combined_input = torch.cat([query, hidden_states], dim=1)  # [batch_size, 1+seq_len, embed_dim]
        
        # Project to Q, K, V
        qkv = self.qkv_proj(combined_input)  # [batch_size, 1+seq_len, 3*actual_embed_dim]
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        
        # Split into Q, K, V and rearrange for Flash Attention
        # Flash attention expects [batch, seqlen, nheads, headdim]
        q, k, v = qkv.unbind(dim=2)
        
        # Only use query token's Q (first position), but all K and V
        q_query = q[:, :1]  # [batch_size, 1, num_heads, head_dim]
        k_all = k  # [batch_size, 1+seq_len, num_heads, head_dim]
        v_all = v  # [batch_size, 1+seq_len, num_heads, head_dim]
        
            
        # Apply Flash Attention v2 - handles padding natively
        attn_output = flash_attn_func(
            q_query, k_all, v_all,
            dropout_p=0.0,
            softmax_scale=None,  # Will use default 1/sqrt(head_dim)
            causal=False,
            window_size=(-1, -1),  # No sliding window
            return_attn_probs=False
        )  # [batch_size, 1, num_heads, head_dim]
            
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, 1, self.actual_embed_dim)  # [batch_size, 1, actual_embed_dim]
        output = self.out_proj(attn_output)  # [batch_size, 1, embed_dim]
        
        # Apply layer norm and squeeze
        output = self.norm(output.squeeze(1))  # [batch_size, embed_dim]
        
        return output

# ================================================================================================================================
#                                                                                                           Fusion Cross attention
# ================================================================================================================================

class FlashCrossAttention(nn.Module):
    """Flash Attention v2 optimized cross-attention for fusion operations."""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.0, use_bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = min(embed_dim // num_heads, 256)  # Ensure head_dim <= 256 for FlashAttention
        self.dropout = dropout
        
        # Fused QKV projections for efficiency - use actual head dimensions
        self.actual_embed_dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, self.actual_embed_dim, bias=use_bias)
        self.kv_proj = nn.Linear(embed_dim, self.actual_embed_dim * 2, bias=use_bias)
        
        # Output projection - from actual_embed_dim back to original embed_dim
        self.out_proj = nn.Linear(self.actual_embed_dim, embed_dim, bias=use_bias)
        
        # Normalization layers
        self.q_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Learnable temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** 0.5))
        
    def forward(self, query, key_value):
        """
        Flash Attention v2 cross-attention.
        Flash attention handles padding natively without explicit masks.
        
        Args:
            query: Query tensor [batch_size, query_len, embed_dim] 
            key_value: Key/Value tensor [batch_size, kv_len, embed_dim]
            
        Returns:
            Output tensor [batch_size, query_len, embed_dim]
        """
        batch_size, query_len, embed_dim = query.shape
        kv_len = key_value.shape[1]
        
        # Normalize inputs
        q_normed = self.q_norm(query)
        kv_normed = self.k_norm(key_value)
        
        # Project to Q, K, V
        q = self.q_proj(q_normed)  # [batch_size, query_len, actual_embed_dim]
        kv = self.kv_proj(kv_normed)  # [batch_size, kv_len, 2*actual_embed_dim]
        
        # Split K and V
        k, v = torch.chunk(kv, 2, dim=-1)  # Each: [batch_size, kv_len, actual_embed_dim]
        
        # Reshape for Flash Attention: [batch, seqlen, num_heads, head_dim]
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim) 
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim)
        
        # Ensure consistent dtypes for Flash Attention
        target_dtype = q.dtype
        if target_dtype == torch.float32:
            target_dtype = torch.float16  # Flash Attention prefers float16
        
        q = q.to(dtype=target_dtype)
        k = k.to(dtype=target_dtype)
        v = v.to(dtype=target_dtype)
        
        # Apply Flash Attention v2 - handles padding natively
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=1.0 / self.temperature.clamp(min=1.0),
            causal=False,
            window_size=(-1, -1),  # No sliding window
            return_attn_probs=False
        )
        
        # Reshape output: [batch, query_len, num_heads, head_dim] -> [batch, query_len, actual_embed_dim]
        attn_output = attn_output.reshape(batch_size, query_len, self.actual_embed_dim)
        
        # Final output projection
        output = self.out_proj(attn_output)
        
        return output


class FlashFusionAttention(nn.Module):
    """Specialized Flash Attention v2 for multimodal fusion with residual connections."""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Flash cross-attention for deep interaction
        self.flash_cross_attn = FlashCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_bias=False
        )
        
        # Gated residual connection
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim, bias=False),
            nn.Sigmoid()
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
    def forward(self, x, y):
        """
        Fused attention between two modalities.
        Flash attention handles padding natively without explicit masks.
        
        Args:
            x: Primary modality [batch_size, x_len, embed_dim] or [batch_size, embed_dim]
            y: Secondary modality [batch_size, y_len, embed_dim] or [batch_size, embed_dim]
            
        Returns:
            Enhanced x with information from y
        """
        # Handle vector inputs by expanding to sequences
        x_was_vector = x.dim() == 2
        y_was_vector = y.dim() == 2
        
        if x_was_vector:
            x = x.unsqueeze(1)  # [batch_size, 1, embed_dim]
        if y_was_vector:
            y = y.unsqueeze(1)  # [batch_size, 1, embed_dim]
            
        # Apply Flash cross-attention: x attends to y - flash attention handles padding natively
        attended = self.flash_cross_attn(
            query=x,
            key_value=y
        )
        
        # Gated residual connection
        combined = torch.cat([x, attended], dim=-1)
        gate_weights = self.gate(combined)
        
        # Weighted combination
        output = x + gate_weights * attended
        
        # Apply output normalization
        output = self.output_norm(output)
        
        # Return to original shape if input was vector
        if x_was_vector:
            output = output.squeeze(1)
            
        return output
