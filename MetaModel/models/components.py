"""Reusable components for DNA-text fusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedFusionBlock(nn.Module):
    """Advanced multimodal fusion with cross-attention, multiplicative interactions, and efficient gating."""
    
    def __init__(self, hidden_dim, input_dim_x=None, input_dim_y=None, 
                 num_heads=8, compression_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.input_dim_x = input_dim_x or hidden_dim
        self.input_dim_y = input_dim_y or hidden_dim
        
        # Efficient projections with weight sharing when possible
        self.project_x = nn.Linear(self.input_dim_x, hidden_dim, bias=False) if self.input_dim_x != hidden_dim else nn.Identity()
        self.project_y = nn.Linear(self.input_dim_y, hidden_dim, bias=False) if self.input_dim_y != hidden_dim else nn.Identity()
        
        # Flash Attention v2 cross-attention for deep interaction
        from .attention import FlashFusionAttention
        self.flash_cross_attention = FlashFusionAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Multiplicative interactions (Hadamard product + learned scaling)
        self.mult_gate = nn.Parameter(torch.ones(hidden_dim))
        self.mult_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Efficient gating with compression
        compressed_dim = hidden_dim // compression_ratio
        self.gate_compress = nn.Sequential(
            nn.Linear(hidden_dim * 2, compressed_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_dim, 3, bias=True),  # 3-way gating: x, y, fusion
            nn.Softmax(dim=-1)
        )
        
        # Fusion with residual connections
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection for residual connection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, x, y):
        # Project inputs to common dimension
        x_proj = self.project_x(x)  # [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        y_proj = self.project_y(y)
        
        # Handle dimension consistency - ensure both have same number of dimensions
        x_is_vector = x_proj.dim() == 2
        y_is_vector = y_proj.dim() == 2
        
        # Apply Flash Attention v2 cross-attention for deep multimodal interaction
        # Flash cross-attention handles both vector and sequence inputs automatically
        x_proj = self.flash_cross_attention(x_proj, y_proj)
        
        # Handle different dimension combinations
        if x_is_vector and y_is_vector:
            # Both are vectors - no broadcasting needed
            mult_interaction = (x_proj * y_proj) * self.mult_gate + self.mult_bias
        elif x_is_vector and not y_is_vector:
            # x is vector, y is sequence - broadcast x
            x_proj = x_proj.unsqueeze(1).expand(-1, y_proj.size(1), -1)
            mult_interaction = (x_proj * y_proj) * self.mult_gate + self.mult_bias
        elif not x_is_vector and y_is_vector:
            # x is sequence, y is vector - broadcast y
            y_proj = y_proj.unsqueeze(1).expand(-1, x_proj.size(1), -1)
            mult_interaction = (x_proj * y_proj) * self.mult_gate + self.mult_bias
        else:
            # Both are sequences
            mult_interaction = (x_proj * y_proj) * self.mult_gate + self.mult_bias

        # Prepare for gating - flatten if needed for vector computation
        if x_proj.dim() == 3:
            # For sequences, pool for gating computation
            x_for_gate = x_proj.mean(dim=1)  # [batch, hidden_dim]
            y_for_gate = y_proj.mean(dim=1) if y_proj.dim() == 3 else y_proj
        else:
            x_for_gate, y_for_gate = x_proj, y_proj
            
        # Compute gating weights
        combined_for_gate = torch.cat([x_for_gate, y_for_gate], dim=-1)
        gate_weights = self.gate_compress(combined_for_gate)  # [batch, 3]
        
        # Split gate weights
        gate_x, gate_y, gate_fused = gate_weights.unbind(dim=-1)
        
        # Apply gating - broadcast for sequences if needed
        if x_proj.dim() == 3:
            gate_x = gate_x.unsqueeze(1)
            gate_y = gate_y.unsqueeze(1) 
            gate_fused = gate_fused.unsqueeze(1)
        
        # Fusion computation - ensure same dimensions
        combined = torch.cat([x_proj, y_proj], dim=-1)
        fused = self.fusion(combined)
        
        # Weighted combination with multiplicative term
        output = gate_x * x_proj + gate_y * y_proj + gate_fused * fused
        
        if mult_interaction is not None:
            # Add multiplicative interaction with learned weight
            mult_weight = torch.sigmoid(self.mult_gate.mean())  # Single weight for all dims
            output = output + mult_weight * mult_interaction
        
        # Output projection with residual
        output = self.output_proj(output) + x_proj  # Residual from primary input
        
        return output


class CrossModalityTransformerBlock(nn.Module):
    """Transformer block with Flash Self-Attention + Cross-Attention."""
    def __init__(self, dim, cross_dim=None, n_heads=8, dropout=0.1, ff_expansion=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim)

        # Use MQA for self-attention as well for consistency and efficiency
        from .attention import MQAFlashAttention
        self.self_attn = MQAFlashAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout
        )

        from .attention import NormalizedGatedCrossAttention
        self.cross_attn = NormalizedGatedCrossAttention(
            query_dim=dim,
            context_dim=cross_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)

        ff_hidden = int(dim * ff_expansion)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.SiLU(),
            nn.Linear(ff_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, self_mask=None, context_mask=None):
        # Ensure consistent dtypes
        target_dtype = x.dtype
        context = context.to(dtype=target_dtype)
        
        x_norm = self.norm1(x)

        # Use MQAFlashAttention for self-attention (now with FlashAttention-2)
        attn_output = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            mask=self_mask
        )

        x = x + attn_output
        x = self.cross_attn(x, context, mask=context_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class EfficientModalityAdapter(nn.Module):
    """High-efficiency modality adapter with parallel FFN structure."""
    
    def __init__(self, dim, expansion_factor=2.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.expansion_factor = expansion_factor
        
        hidden_dim = int(dim * expansion_factor)
        
        # Pre-normalization layer
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        
        # Parallel FFN structure (like in PaLM)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        
        # Learnable scaling for residual connection
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Pre-norm
        normed_x = self.norm(x)
        
        # Parallel FFN: gate and up projections computed in parallel
        gate = self.gate_proj(normed_x)
        up = self.up_proj(normed_x)
        
        # Apply activation to gate and multiply with up
        hidden = F.silu(gate) * up
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Down projection
        adapter_output = self.down_proj(hidden)
        
        # Residual connection with learnable scaling
        return x + self.residual_scale * adapter_output





class EfficientJointProjection(nn.Module):
    """Efficient joint projection with multiple pathways and self-attention."""
    
    def __init__(self, input_dim, output_dim, num_heads=8, num_pathways=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_pathways = num_pathways
        
        # Multiple projection pathways for diverse representations
        self.pathways = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=False),
                nn.GELU() if i == 0 else nn.SiLU(),  # Different activations
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim, bias=False)
            ) for i in range(num_pathways)
        ])
        
        # Pathway weighting
        self.pathway_weights = nn.Parameter(torch.ones(num_pathways) / num_pathways)
        
        # Self-attention for sequence inputs
        from .attention import MQAFlashAttention
        self.self_attention = MQAFlashAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(output_dim)
        
        # Final projection with residual connection
        self.final_proj = nn.Linear(output_dim, output_dim, bias=False)
        
        # Skip connection if dimensions match
        self.use_skip = input_dim == output_dim
        if not self.use_skip:
            self.skip_proj = nn.Linear(input_dim, output_dim, bias=False)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(output_dim, eps=1e-6)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, input_dim] or [batch_size, seq_len, input_dim]
        
        Returns:
            Projected tensor [batch_size, output_dim] or [batch_size, seq_len, output_dim]
        """
        # Apply multiple pathways
        pathway_outputs = []
        for pathway in self.pathways:
            pathway_outputs.append(pathway(x))
        
        # Weighted combination of pathways
        weights = torch.softmax(self.pathway_weights, dim=0)
        combined = sum(w * out for w, out in zip(weights, pathway_outputs))
        
        # Self-attention for sequence inputs
        if hasattr(self, 'use_self_attention') and self.use_self_attention and x.dim() == 3:
            attn_out = self.self_attention(
                query=combined,
                key=combined, 
                value=combined
            )
            combined = self.attn_norm(combined + attn_out)
        
        # Final projection
        output = self.final_proj(combined)
        
        # Skip connection
        if self.use_skip:
            output = output + x
        else:
            skip_input = self.skip_proj(x) if hasattr(self, 'skip_proj') else x
            output = output + skip_input
        
        # Output normalization
        output = self.output_norm(output)
        
        return output

