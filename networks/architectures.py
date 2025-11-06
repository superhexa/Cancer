from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torchvision import models


class BaseArchitecture(ABC, nn.Module):
    
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self._num_classes = num_classes
        self._dropout = dropout
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True


class DenseNetClassifier(BaseArchitecture):
    
    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = True):
        super().__init__(num_classes, dropout)
        
        backbone = models.densenet121(pretrained=pretrained)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        num_features = backbone.classifier.in_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class ResNetClassifier(BaseArchitecture):
    
    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = True):
        super().__init__(num_classes, dropout)
        
        backbone = models.resnet50(pretrained=pretrained)
        
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        self.avgpool = backbone.avgpool
        num_features = backbone.fc.in_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class EfficientNetClassifier(BaseArchitecture):
    
    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = True):
        super().__init__(num_classes, dropout)
        
        backbone = models.efficientnet_b0(pretrained=pretrained)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        num_features = backbone.classifier[1].in_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 384),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(384),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(384, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class ModelRegistry:
    
    _architectures = {
        'densenet121': DenseNetClassifier,
        'resnet50': ResNetClassifier,
        'efficientnet_b0': EfficientNetClassifier
    }
    
    @classmethod
    def create(cls, config) -> BaseArchitecture:
        architecture_name = config.get('network', 'architecture')
        num_classes = config.get('network', 'num_classes')
        dropout = config.get('network', 'dropout', default=0.3)
        pretrained = config.get('network', 'pretrained', default=True)
        
        if architecture_name not in cls._architectures:
            raise ValueError(f"Unknown architecture: {architecture_name}")
        
        model = cls._architectures[architecture_name](
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained
        )
        
        if config.get('network', 'freeze_features', default=False):
            model.freeze_backbone()
        
        return model
    
    @classmethod
    def register(cls, name: str, architecture_class: type) -> None:
        cls._architectures[name] = architecture_class
