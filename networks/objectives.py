from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(ABC, nn.Module):
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self._reduction = reduction
    
    @abstractmethod
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass


class CrossEntropyLoss(BaseLoss):
    
    def __init__(self, label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self._loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction=reduction
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(predictions, targets)


class FocalLoss(BaseLoss):
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self._alpha = alpha
        self._gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self._alpha * (1 - pt) ** self._gamma * ce_loss
        
        if self._reduction == 'mean':
            return focal_loss.mean()
        elif self._reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(BaseLoss):
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__(reduction)
        self._num_classes = num_classes
        self._smoothing = smoothing
        self._confidence = 1.0 - smoothing
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(predictions, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self._smoothing / (self._num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self._confidence)
        
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        if self._reduction == 'mean':
            return loss.mean()
        elif self._reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(BaseLoss):
    
    def __init__(self, losses: list, weights: list):
        super().__init__()
        self._losses = nn.ModuleList(losses)
        self._weights = weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss_fn, weight in zip(self._losses, self._weights):
            total_loss += weight * loss_fn(predictions, targets)
        return total_loss


class LossFactory:
    
    @staticmethod
    def create(config) -> BaseLoss:
        loss_type = config.get('optimization', 'loss', default='cross_entropy')
        
        if loss_type == 'cross_entropy':
            return CrossEntropyLoss(
                label_smoothing=config.get('optimization', 'label_smoothing', default=0.0)
            )
        elif loss_type == 'focal':
            return FocalLoss(
                alpha=config.get('optimization', 'focal_alpha', default=0.25),
                gamma=config.get('optimization', 'focal_gamma', default=2.0)
            )
        elif loss_type == 'label_smoothing':
            return LabelSmoothingLoss(
                num_classes=config.get('network', 'num_classes'),
                smoothing=config.get('optimization', 'smoothing', default=0.1)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
