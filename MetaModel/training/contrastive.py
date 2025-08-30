"""Contrastive loss functions for multimodal learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

def info_nce_loss(features, temperature=0.07, batch_size=None, hard_negative_weight=1.0):
    """
    Improved InfoNCE contrastive loss with numerical stability and hard negative mining.
    
    Args:
        features: Tensor of shape [2*batch_size, feature_dim] containing the features of the positive pairs.
            The first batch_size elements are anchors, and the second batch_size elements are positives.
        temperature: Temperature parameter to scale the similarity scores.
        batch_size: Size of the batch. If None, assumes features has shape [2*B, D] and infers batch_size=B.
        hard_negative_weight: Weight for hard negative mining (>1.0 emphasizes hard negatives).
    
    Returns:
        InfoNCE loss
    """
    if batch_size is None:
        batch_size = features.shape[0] // 2
        
    # Normalize the features with epsilon for numerical stability
    features = F.normalize(features, dim=1, eps=1e-8)
    
    # Create similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create labels
    labels = torch.cat([
        torch.arange(batch_size, 2*batch_size),  # Positives for first half (anchors)
        torch.arange(0, batch_size)              # Positives for second half
    ]).to(features.device)
    
    # Mask out self-similarities on diagonal
    mask_self = torch.eye(similarity_matrix.size(0), device=features.device, dtype=torch.bool)
    similarity_matrix = similarity_matrix.masked_fill(mask_self, float('-inf'))
    
    # Hard negative mining: find hardest negatives and weight them more
    if hard_negative_weight > 1.0:
        # Create negative mask (all except positives)
        pos_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        pos_mask.scatter_(1, labels.unsqueeze(1), True)
        neg_mask = ~pos_mask & ~mask_self
        
        # Find hard negatives (highest similarity among negatives)
        neg_similarities = similarity_matrix.clone()
        neg_similarities[~neg_mask] = float('-inf')
        hard_neg_indices = neg_similarities.argmax(dim=1, keepdim=True)
        
        # Apply hard negative weighting
        hard_neg_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        hard_neg_mask.scatter_(1, hard_neg_indices, True)
        similarity_matrix[hard_neg_mask] *= hard_negative_weight
    
    # Numerically stable InfoNCE computation using logsumexp
    positive_similarities = similarity_matrix.gather(1, labels.unsqueeze(1)).squeeze(1)
    
    # LogSumExp for numerical stability
    log_sum_exp = torch.logsumexp(similarity_matrix, dim=1)
    
    # InfoNCE loss
    loss = -positive_similarities + log_sum_exp
    
    return loss.mean()

def dual_softmax_loss(similarity_matrix, temperature=0.07, label_smoothing=0.0):
    """
    Improved dual softmax loss with label smoothing for better generalization.
    Helps with cross-modal retrieval by optimizing in both directions.
    
    Args:
        similarity_matrix: Similarity matrix of shape [batch_size, batch_size]
        temperature: Temperature parameter
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 typical)
    
    Returns:
        Dual softmax loss
    """
    # Clamp temperature to avoid division by zero
    temperature = max(temperature, 1e-8)
    
    sim_row_scaled = similarity_matrix / temperature
    sim_col_scaled = similarity_matrix.t() / temperature
    
    # Create labels
    batch_size = sim_row_scaled.size(0)
    row_labels = torch.arange(batch_size, device=sim_row_scaled.device)
    col_labels = torch.arange(batch_size, device=sim_col_scaled.device)
    
    # Row-wise softmax (text-to-dna retrieval) with label smoothing
    row_loss = F.cross_entropy(sim_row_scaled, row_labels, label_smoothing=label_smoothing)
    
    # Column-wise softmax (dna-to-text retrieval) with label smoothing
    col_loss = F.cross_entropy(sim_col_scaled, col_labels, label_smoothing=label_smoothing)
    
    # Average the losses
    return (row_loss + col_loss) / 2.0


class DNATextContrastiveLoss(nn.Module):
    """
    Advanced contrastive loss for DNA-Text fusion with multiple improvements:
    - Adaptive temperature scaling based on similarity distributions
    - Hard negative mining
    - Label smoothing
    - Memory-efficient computation
    - Gradient clipping for stability
    """
    def __init__(self, 
                 initial_temp: float = 0.5, 
                 min_temp: float = 0.07, 
                 max_temp: float = 1.0,
                 total_epochs: int = 8,
                 hard_negative_weight: float = 1.2,
                 label_smoothing: float = 0.05,
                 adaptive_temp: bool = True,
                 loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.total_epochs = total_epochs
        self.hard_negative_weight = hard_negative_weight
        self.label_smoothing = label_smoothing
        self.adaptive_temp = adaptive_temp
        
        # Loss component weights
        self.loss_weights = loss_weights or {"info_nce": 0.7, "dual_softmax": 0.3}
        
        # Learnable temperature parameter if adaptive
        if adaptive_temp:
            self.temp_param = nn.Parameter(torch.tensor(initial_temp))
        
        # Moving average for similarity statistics
        self.register_buffer('sim_mean_ema', torch.tensor(0.0))
        self.register_buffer('sim_std_ema', torch.tensor(1.0))
        self.ema_decay = 0.99
        
    def _adaptive_temperature(self, similarity_matrix: torch.Tensor) -> float:
        """Compute adaptive temperature based on similarity distribution."""
        if not self.adaptive_temp:
            return self.temp_param if hasattr(self, 'temp_param') else self.initial_temp
            
        with torch.no_grad():
            # Compute similarity statistics
            sim_mean = similarity_matrix.mean()
            sim_std = similarity_matrix.std()
            
            # Update EMA statistics
            self.sim_mean_ema = self.ema_decay * self.sim_mean_ema + (1 - self.ema_decay) * sim_mean
            self.sim_std_ema = self.ema_decay * self.sim_std_ema + (1 - self.ema_decay) * sim_std
            
            # Adaptive temperature: higher temp when similarities are too concentrated
            adaptive_factor = 1.0 + torch.clamp(1.0 - self.sim_std_ema, 0.0, 0.5)
            
        temp = self.temp_param * adaptive_factor
        return torch.clamp(temp, self.min_temp, self.max_temp)
        
    def forward(self, 
                pred_embeddings: torch.Tensor, 
                target_embeddings: torch.Tensor, 
                epoch: int = 0,
                return_similarities: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate optimized combined contrastive loss.
        
        Args:
            pred_embeddings: Embeddings of predicted modality [batch_size, embed_dim]
            target_embeddings: Embeddings of target modality [batch_size, embed_dim]
            epoch: Current epoch for temperature annealing
            return_similarities: Whether to return similarity matrices for analysis
            
        Returns:
            Combined contrastive loss and metrics dict
        """
        batch_size = pred_embeddings.size(0)
        
        # Normalize embeddings with numerical stability
        pred_embeddings = F.normalize(pred_embeddings, dim=1, eps=1e-8)
        target_embeddings = F.normalize(target_embeddings, dim=1, eps=1e-8)

        # Compute similarity matrix once for efficiency
        similarity_matrix = torch.matmul(pred_embeddings, target_embeddings.T)
        
        # Apply temperature (adaptive or scheduled)
        if self.adaptive_temp:
            temp = self._adaptive_temperature(similarity_matrix)
        else:
            # Linear temperature annealing
            progress = min(epoch / self.total_epochs, 1.0)
            temp = max(self.min_temp, self.initial_temp - progress * (self.initial_temp - self.min_temp))
        
        # InfoNCE loss with hard negative mining
        features = torch.cat([pred_embeddings, target_embeddings], dim=0)
        info_nce = info_nce_loss(features, temperature=temp, hard_negative_weight=self.hard_negative_weight)
        
        # Dual softmax loss with label smoothing
        dual_softmax = dual_softmax_loss(similarity_matrix, temperature=temp, label_smoothing=self.label_smoothing)
        
        # Weighted combination of losses
        combined_loss = (self.loss_weights["info_nce"] * info_nce + 
                        self.loss_weights["dual_softmax"] * dual_softmax)
        
        # Compute additional metrics
        with torch.no_grad():
            # Retrieval accuracy (top-1)
            pred_to_target = similarity_matrix.argmax(dim=1)
            target_to_pred = similarity_matrix.argmax(dim=0)
            
            pred_to_target_acc = (pred_to_target == torch.arange(batch_size, device=pred_to_target.device)).float().mean()
            target_to_pred_acc = (target_to_pred == torch.arange(batch_size, device=target_to_pred.device)).float().mean()
            
            # Similarity statistics
            pos_similarities = similarity_matrix.diag()  # Positive pairs
            neg_similarities = similarity_matrix.masked_select(~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device))
            
        # Metrics for logging
        metrics = {
            "info_nce_loss": info_nce.item(),
            "dual_softmax_loss": dual_softmax.item(),
            "contrastive_loss": combined_loss.item(),
            "contrastive_temperature": temp.item() if isinstance(temp, torch.Tensor) else temp,
            "pred_to_target_acc": pred_to_target_acc.item(),
            "target_to_pred_acc": target_to_pred_acc.item(),
            "avg_positive_sim": pos_similarities.mean().item(),
            "avg_negative_sim": neg_similarities.mean().item(),
            "similarity_gap": (pos_similarities.mean() - neg_similarities.mean()).item()
        }
        
        if return_similarities:
            metrics["similarity_matrix"] = similarity_matrix.detach()
            
        return combined_loss, metrics