"""
Personal Food Taste Classification Model

EfficientNet-B3 based architecture for learning personal food taste preferences.

Author: Noam & Shany
Course: 046211 Deep Learning, Technion

Attribution:
- EfficientNet: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
- timm library: https://github.com/rwightman/pytorch-image-models
- PyTorch: https://pytorch.org/
- Transfer Learning: Yosinski, J., et al. (2014). How transferable are features in deep neural networks? NIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Tuple


class Food101TasteClassifier(nn.Module):
    """
    EfficientNet-B3 based classifier for personal food taste preferences.
    
    Architecture:
    - Backbone: tf_efficientnet_b3 (ImageNet pretrained)
    - Head: Global pooling -> Linear(emb_dim, 256) + GELU + Dropout + Linear(256, 3)
    
    Attribution:
    - EfficientNet architecture: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling 
      for convolutional neural networks. ICML.
    - Implementation via timm: https://github.com/rwightman/pytorch-image-models
    - Transfer learning approach: Yosinski, J., et al. (2014). How transferable are features 
      in deep neural networks? NIPS.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = 'tf_efficientnet_b3',
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load pretrained backbone
        # Attribution: timm library for pre-trained models
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            global_pool='avg'  # Global average pooling
        )
        
        # Get embedding dimension
        self.embedding_dim = self.backbone.num_features
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # For EMA (Exponential Moving Average) weights
        self.register_buffer('ema_decay', torch.tensor(0.999))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            logits: Class logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)  # [B, embedding_dim]
        
        # Classification
        logits = self.head(features)  # [B, num_classes]
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without classification."""
        return self.backbone(x)
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        print(f"Backbone {'frozen' if freeze else 'unfrozen'}")
    
    def unfreeze_last_blocks(self, num_blocks: int = 1):
        """
        Unfreeze last N blocks of backbone for fine-tuning.
        Used in Stage C of curriculum learning.
        """
        # This is backbone-specific. For EfficientNet, we unfreeze last blocks
        if hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            for i in range(max(0, total_blocks - num_blocks), total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
            print(f"Unfroze last {num_blocks} backbone blocks")
        else:
            print("Warning: Could not identify backbone blocks for partial unfreezing")


def expected_taste_score(probs: torch.Tensor) -> torch.Tensor:
    """
    Convert class probabilities to expected taste score.
    Used for consistency loss with CLIP scores.
    
    Args:
        probs: Class probabilities [B, 3] (disgusting, neutral, tasty)
        
    Returns:
        expected_scores: Expected taste scores [B] in range [0, 1]
    """
    # Map class indices to taste values
    taste_values = probs.new_tensor([0.0, 0.5, 1.0])  # disgusting, neutral, tasty
    
    # Compute expectation: sum(prob_i * value_i)
    expected_scores = torch.sum(probs * taste_values, dim=1)
    
    return expected_scores


def create_model(
    model_config: Dict,
    device: torch.device = None
) -> Food101TasteClassifier:
    """
    Factory function to create model with configuration.
    
    Args:
        model_config: Dictionary with model parameters
        device: Target device
        
    Returns:
        Initialized model
    """
    model = Food101TasteClassifier(
        num_classes=model_config.get('num_classes', 3),
        backbone=model_config.get('backbone', 'tf_efficientnet_b3'),
        pretrained=model_config.get('pretrained', True),
        dropout=model_config.get('dropout', 0.3),
        hidden_dim=model_config.get('hidden_dim', 256)
    )
    
    if device:
        model = model.to(device)
    
    print(f"Created {model.backbone_name} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


class ModelEMA:
    """
    Exponential Moving Average for model weights.
    Improves model stability and performance.
    
    Attribution:
    - EMA technique: Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic 
      approximation by averaging. SIAM Journal on Control and Optimization.
    - Implementation inspired by: https://github.com/rwightman/pytorch-image-models
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class TasteClassificationLoss(nn.Module):
    """
    Combined loss for personal taste classification.
    
    Combines:
    1. Cross-entropy loss with class weights for imbalanced data
    2. Optional consistency loss with original preference scores
    
    Attribution:
    - Cross-entropy loss: Standard PyTorch implementation
    - Label smoothing: Szegedy, C., et al. (2016). Rethinking the inception architecture 
      for computer vision. CVPR.
    - Class weighting: He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE TKDE.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        use_consistency: bool = True,
        consistency_weight: float = 0.1
    ):
        super().__init__()
        
        self.use_consistency = use_consistency
        self.consistency_weight = consistency_weight
        
        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        # MSE for consistency
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        clip_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions [B, 3]
            targets: True class labels [B]
            clip_scores: Original CLIP scores [B] (optional)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Primary cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        loss_dict = {'ce_loss': ce_loss.item()}
        total_loss = ce_loss
        
        # Optional consistency loss
        if self.use_consistency and clip_scores is not None:
            probs = F.softmax(logits, dim=1)
            expected_scores = expected_taste_score(probs)
            
            consistency_loss = self.mse_loss(expected_scores, clip_scores)
            loss_dict['consistency_loss'] = consistency_loss.item()
            
            total_loss = total_loss + self.consistency_weight * consistency_loss
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


def get_parameter_groups(
    model: Food101TasteClassifier,
    head_lr: float = 3e-4,
    backbone_lr: float = 3e-5,
    weight_decay: float = 0.05
) -> list:
    """
    Create parameter groups with discriminative learning rates for transfer learning.
    
    Args:
        model: The model
        head_lr: Learning rate for classification head (higher for new layers)
        backbone_lr: Learning rate for backbone (lower to preserve features)
        weight_decay: Weight decay for regularization
        
    Returns:
        List of parameter groups for optimizer
        
    Attribution:
    - Discriminative learning rates: Howard, J., & Ruder, S. (2018). Universal language model 
      fine-tuning for text classification. ACL.
    - Applied to computer vision transfer learning: Standard practice in fine-tuning
    """
    # Separate head and backbone parameters
    head_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    parameter_groups = [
        {
            'params': head_params,
            'lr': head_lr,
            'weight_decay': weight_decay,
            'name': 'head'
        },
        {
            'params': backbone_params,
            'lr': backbone_lr,
            'weight_decay': weight_decay,
            'name': 'backbone'
        }
    ]
    
    print(f"Created parameter groups: head_lr={head_lr}, backbone_lr={backbone_lr}")
    return parameter_groups


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = {
        'num_classes': 3,
        'backbone': 'tf_efficientnet_b3',
        'pretrained': True,
        'dropout': 0.3,
        'hidden_dim': 256
    }
    
    model = create_model(model_config, device)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # Test loss
    class_weights = torch.tensor([1.17, 1.35, 0.71])  # From our analysis
    loss_fn = TasteClassificationLoss(class_weights=class_weights)
    
    targets = torch.tensor([0, 2])
    clip_scores = torch.tensor([0.3, 0.8])
    
    total_loss, loss_dict = loss_fn(logits, targets, clip_scores)
    print(f"Loss: {loss_dict}")
