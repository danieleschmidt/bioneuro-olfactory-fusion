"""
Advanced fusion techniques for neuromorphic gas detection.
Implements sophisticated multi-modal fusion strategies and adaptive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class FusionConfiguration:
    """Configuration for advanced fusion techniques."""
    fusion_strategy: str = "hierarchical_attention"
    attention_heads: int = 8
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    temperature: float = 1.0
    use_temporal_context: bool = True
    context_window_size: int = 10
    adaptive_weights: bool = True
    uncertainty_estimation: bool = True


class AttentionFusionLayer(nn.Module):
    """Multi-head attention fusion layer for combining modalities."""
    
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, chemical_features: torch.Tensor, 
                audio_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention fusion.
        
        Args:
            chemical_features: [batch_size, seq_len, input_dim]
            audio_features: [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Fused features and attention weights
        """
        batch_size, seq_len, _ = chemical_features.shape
        
        # Create combined input for cross-modal attention
        combined_input = torch.stack([chemical_features, audio_features], dim=2)  # [B, L, 2, D]
        combined_input = combined_input.view(batch_size, seq_len * 2, self.input_dim)
        
        # Project to query, key, value
        Q = self.query_proj(combined_input)
        K = self.key_proj(combined_input)
        V = self.value_proj(combined_input)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len * 2, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len * 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len * 2, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len * 2, self.input_dim
        )
        output = self.output_proj(attended)
        
        # Residual connection and normalization
        output = self.layer_norm(output + combined_input)
        
        # Split back to chemical and audio components
        output = output.view(batch_size, seq_len, 2, self.input_dim)
        chemical_fused = output[:, :, 0, :]
        audio_fused = output[:, :, 1, :]
        
        return chemical_fused + audio_fused, attention_weights


class TemporalContextModule(nn.Module):
    """Captures temporal dependencies in multi-modal data."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8, batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal context modeling.
        
        Args:
            features: [batch_size, seq_len, input_dim]
            
        Returns:
            Temporally contextualized features
        """
        # LSTM for temporal modeling
        lstm_out, _ = self.lstm(features)
        
        # Self-attention for long-range dependencies
        attended, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # Project back to original dimension
        output = self.output_proj(attended)
        
        return output + features  # Residual connection


class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty in fusion predictions."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, num_classes)
        self.variance_head = nn.Linear(input_dim, num_classes)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate prediction mean and variance.
        
        Args:
            features: Input features
            
        Returns:
            Prediction mean and variance
        """
        mean = self.mean_head(features)
        log_var = self.variance_head(features)
        variance = torch.exp(log_var)
        
        return mean, variance


class AdaptiveWeightingModule(nn.Module):
    """Learns adaptive weights for different modalities."""
    
    def __init__(self, num_modalities: int, hidden_dim: int = 64):
        super().__init__()
        self.num_modalities = num_modalities
        
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modalities),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute adaptive weights for modalities.
        
        Args:
            modality_features: List of feature tensors from different modalities
            
        Returns:
            Adaptive weights for each modality
        """
        # Global average pooling for each modality
        pooled_features = []
        for features in modality_features:
            pooled = F.adaptive_avg_pool1d(
                features.transpose(1, 2), 1
            ).squeeze(-1)
            pooled_features.append(pooled)
        
        # Concatenate and compute weights
        combined = torch.cat(pooled_features, dim=-1)
        weights = self.weight_network(combined)
        
        return weights


class HierarchicalFusionNetwork(nn.Module):
    """
    Hierarchical fusion network with multiple fusion stages.
    
    Implements a sophisticated fusion strategy that combines:
    - Early fusion for low-level features
    - Intermediate fusion with attention mechanisms  
    - Late fusion for high-level decision making
    """
    
    def __init__(self, config: FusionConfiguration):
        super().__init__()
        self.config = config
        
        # Early fusion layers
        self.early_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Attention fusion
        self.attention_fusion = AttentionFusionLayer(
            config.hidden_dim, config.attention_heads, config.dropout_rate
        )
        
        # Temporal context if enabled
        if config.use_temporal_context:
            self.temporal_context = TemporalContextModule(
                config.hidden_dim, config.hidden_dim // 2
            )
        
        # Adaptive weighting if enabled
        if config.adaptive_weights:
            self.adaptive_weighting = AdaptiveWeightingModule(2, config.hidden_dim)
        
        # Late fusion layers
        self.late_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 8)  # Number of gas classes
        )
        
        # Uncertainty estimation if enabled
        if config.uncertainty_estimation:
            self.uncertainty_estimator = UncertaintyEstimator(
                config.hidden_dim // 4, 8
            )
        
    def forward(self, chemical_features: torch.Tensor,
                audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical fusion network.
        
        Args:
            chemical_features: Chemical sensor features
            audio_features: Audio sensor features
            
        Returns:
            Dictionary containing predictions and metadata
        """
        batch_size, seq_len, feature_dim = chemical_features.shape
        
        # Early fusion - simple concatenation and projection
        early_fused = torch.cat([chemical_features, audio_features], dim=-1)
        early_fused = self.early_fusion(early_fused)
        
        # Attention-based fusion
        attention_fused, attention_weights = self.attention_fusion(
            chemical_features, audio_features
        )
        
        # Combine early and attention fusion
        combined_features = early_fused + attention_fused
        
        # Temporal context modeling
        if self.config.use_temporal_context:
            combined_features = self.temporal_context(combined_features)
        
        # Adaptive weighting
        if self.config.adaptive_weights:
            modality_weights = self.adaptive_weighting([
                chemical_features, audio_features
            ])
            
            # Apply adaptive weights
            weighted_chemical = chemical_features * modality_weights[:, 0:1].unsqueeze(-1)
            weighted_audio = audio_features * modality_weights[:, 1:2].unsqueeze(-1)
            combined_features = combined_features + weighted_chemical + weighted_audio
        
        # Global average pooling for sequence dimension
        pooled_features = F.adaptive_avg_pool1d(
            combined_features.transpose(1, 2), 1
        ).squeeze(-1)
        
        # Late fusion for final prediction
        late_features = self.late_fusion[:-1](pooled_features)  # All layers except last
        
        outputs = {}
        
        if self.config.uncertainty_estimation:
            # Uncertainty-aware prediction
            pred_mean, pred_variance = self.uncertainty_estimator(late_features)
            outputs['predictions'] = pred_mean
            outputs['uncertainty'] = pred_variance
            outputs['confidence'] = 1.0 / (1.0 + pred_variance.mean(dim=-1))
        else:
            # Standard prediction
            predictions = self.late_fusion[-1](late_features)
            outputs['predictions'] = predictions
            outputs['confidence'] = F.softmax(predictions / self.config.temperature, dim=-1).max(dim=-1)[0]
        
        # Add attention weights and modality weights to outputs
        outputs['attention_weights'] = attention_weights
        if self.config.adaptive_weights:
            outputs['modality_weights'] = modality_weights
        
        outputs['fused_features'] = combined_features
        outputs['pooled_features'] = pooled_features
        
        return outputs


class MultiScaleFusionNetwork(nn.Module):
    """
    Multi-scale fusion network that processes features at different temporal scales.
    """
    
    def __init__(self, input_dim: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        # Multi-scale convolutions
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size=2**i + 1, 
                     padding=2**i, dilation=1)
            for i in range(num_scales)
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(input_dim * num_scales, input_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale processing.
        
        Args:
            features: [batch_size, seq_len, input_dim]
            
        Returns:
            Multi-scale fused features
        """
        # Transpose for conv1d
        features = features.transpose(1, 2)  # [B, D, L]
        
        scale_outputs = []
        for conv in self.scale_convs:
            scale_out = F.relu(conv(features))
            scale_outputs.append(scale_out)
        
        # Concatenate scales
        multi_scale = torch.cat(scale_outputs, dim=1)
        
        # Transpose back and fuse scales
        multi_scale = multi_scale.transpose(1, 2)  # [B, L, D*num_scales]
        fused = self.scale_fusion(multi_scale)
        
        return fused


class CrossModalContrastiveLearning(nn.Module):
    """
    Contrastive learning module for cross-modal representation learning.
    """
    
    def __init__(self, feature_dim: int, projection_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
        self.chemical_projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.audio_projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, chemical_features: torch.Tensor,
                audio_features: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between modalities.
        
        Args:
            chemical_features: Chemical sensor features
            audio_features: Audio sensor features
            
        Returns:
            Contrastive loss
        """
        # Project features
        chemical_proj = F.normalize(self.chemical_projector(chemical_features), dim=-1)
        audio_proj = F.normalize(self.audio_projector(audio_features), dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(chemical_proj, audio_proj.transpose(-2, -1)) / self.temperature
        
        # Create labels (positive pairs are on diagonal)
        batch_size = chemical_features.shape[0]
        labels = torch.arange(batch_size, device=chemical_features.device)
        
        # Contrastive loss
        loss_chemical = F.cross_entropy(logits, labels)
        loss_audio = F.cross_entropy(logits.transpose(-2, -1), labels)
        
        return (loss_chemical + loss_audio) / 2


class AdvancedFusionSystem:
    """
    Complete advanced fusion system with multiple techniques.
    """
    
    def __init__(self, config: FusionConfiguration):
        self.config = config
        
        # Main fusion network
        self.fusion_network = HierarchicalFusionNetwork(config)
        
        # Multi-scale processing
        self.multi_scale = MultiScaleFusionNetwork(config.hidden_dim)
        
        # Contrastive learning
        self.contrastive_learning = CrossModalContrastiveLearning(config.hidden_dim)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_weight = 0.1
        
    def forward(self, chemical_input: torch.Tensor,
                audio_input: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through advanced fusion system.
        
        Args:
            chemical_input: Chemical sensor data
            audio_input: Audio sensor data
            targets: Ground truth labels (for training)
            
        Returns:
            Dictionary with predictions and losses
        """
        # Multi-scale processing
        chemical_multiscale = self.multi_scale(chemical_input)
        audio_multiscale = self.multi_scale(audio_input)
        
        # Main fusion
        fusion_outputs = self.fusion_network(chemical_multiscale, audio_multiscale)
        
        results = fusion_outputs.copy()
        
        if targets is not None:
            # Main classification loss
            main_loss = self.criterion(fusion_outputs['predictions'], targets)
            
            # Contrastive loss
            contrastive_loss = self.contrastive_learning(
                chemical_multiscale, audio_multiscale
            )
            
            # Combined loss
            total_loss = main_loss + self.contrastive_weight * contrastive_loss
            
            results.update({
                'main_loss': main_loss,
                'contrastive_loss': contrastive_loss,
                'total_loss': total_loss
            })
        
        return results
    
    def predict(self, chemical_input: torch.Tensor,
                audio_input: torch.Tensor) -> Dict[str, Any]:
        """
        Make predictions with confidence estimation.
        
        Args:
            chemical_input: Chemical sensor data
            audio_input: Audio sensor data
            
        Returns:
            Predictions with confidence scores
        """
        self.fusion_network.eval()
        
        with torch.no_grad():
            results = self.forward(chemical_input, audio_input)
            
            predictions = results['predictions']
            probabilities = F.softmax(predictions, dim=-1)
            
            predicted_classes = torch.argmax(probabilities, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
            
            output = {
                'predicted_classes': predicted_classes.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'confidence_scores': confidence_scores.cpu().numpy(),
            }
            
            if 'uncertainty' in results:
                output['uncertainty'] = results['uncertainty'].cpu().numpy()
            
            if 'modality_weights' in results:
                output['modality_weights'] = results['modality_weights'].cpu().numpy()
            
            return output