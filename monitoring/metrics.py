from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix


class BaseMetric(ABC):
    
    def __init__(self, name: str):
        self._name = name
        self.reset()
    
    @abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pass
    
    @abstractmethod
    def compute(self) -> float:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @property
    def name(self) -> str:
        return self._name


class Accuracy(BaseMetric):
    
    def __init__(self):
        super().__init__('accuracy')
    
    def reset(self) -> None:
        self._correct = 0
        self._total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred_classes = torch.argmax(predictions, dim=1)
        self._correct += (pred_classes == targets).sum().item()
        self._total += targets.size(0)
    
    def compute(self) -> float:
        if self._total == 0:
            return 0.0
        return self._correct / self._total


class TopKAccuracy(BaseMetric):
    
    def __init__(self, k: int = 2):
        self._k = k
        super().__init__(f'top{k}_accuracy')
    
    def reset(self) -> None:
        self._correct = 0
        self._total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        _, topk_preds = predictions.topk(self._k, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(topk_preds)
        self._correct += (topk_preds == targets_expanded).any(dim=1).sum().item()
        self._total += targets.size(0)
    
    def compute(self) -> float:
        if self._total == 0:
            return 0.0
        return self._correct / self._total


class Precision(BaseMetric):
    
    def __init__(self, average: str = 'binary'):
        self._average = average
        super().__init__('precision')
    
    def reset(self) -> None:
        self._all_predictions = []
        self._all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred_classes = torch.argmax(predictions, dim=1)
        self._all_predictions.extend(pred_classes.cpu().numpy())
        self._all_targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        if len(self._all_targets) == 0:
            return 0.0
        precision, _, _, _ = precision_recall_fscore_support(
            self._all_targets,
            self._all_predictions,
            average=self._average,
            zero_division=0
        )
        return float(precision)


class Recall(BaseMetric):
    
    def __init__(self, average: str = 'binary'):
        self._average = average
        super().__init__('recall')
    
    def reset(self) -> None:
        self._all_predictions = []
        self._all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred_classes = torch.argmax(predictions, dim=1)
        self._all_predictions.extend(pred_classes.cpu().numpy())
        self._all_targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        if len(self._all_targets) == 0:
            return 0.0
        _, recall, _, _ = precision_recall_fscore_support(
            self._all_targets,
            self._all_predictions,
            average=self._average,
            zero_division=0
        )
        return float(recall)


class F1Score(BaseMetric):
    
    def __init__(self, average: str = 'binary'):
        self._average = average
        super().__init__('f1_score')
    
    def reset(self) -> None:
        self._all_predictions = []
        self._all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred_classes = torch.argmax(predictions, dim=1)
        self._all_predictions.extend(pred_classes.cpu().numpy())
        self._all_targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        if len(self._all_targets) == 0:
            return 0.0
        _, _, f1, _ = precision_recall_fscore_support(
            self._all_targets,
            self._all_predictions,
            average=self._average,
            zero_division=0
        )
        return float(f1)


class ROCAUC(BaseMetric):
    
    def __init__(self):
        super().__init__('roc_auc')
    
    def reset(self) -> None:
        self._all_probabilities = []
        self._all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        probabilities = torch.softmax(predictions, dim=1)
        self._all_probabilities.extend(probabilities.cpu().numpy())
        self._all_targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        if len(self._all_targets) == 0:
            return 0.0
        probabilities = np.array(self._all_probabilities)
        targets = np.array(self._all_targets)
        
        if len(np.unique(targets)) < 2:
            return 0.0
        
        return roc_auc_score(targets, probabilities[:, 1])


class MetricsTracker:
    
    def __init__(self, metrics: List[BaseMetric]):
        self._metrics = {metric.name: metric for metric in metrics}
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        for metric in self._metrics.values():
            metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        return {name: metric.compute() for name, metric in self._metrics.items()}
    
    def reset(self) -> None:
        for metric in self._metrics.values():
            metric.reset()
    
    def get(self, name: str) -> float:
        if name not in self._metrics:
            raise KeyError(f"Metric {name} not found")
        return self._metrics[name].compute()


class MetricsFactory:
    
    @staticmethod
    def create_standard_metrics() -> List[BaseMetric]:
        return [
            Accuracy(),
            TopKAccuracy(k=2),
            Precision(average='binary'),
            Recall(average='binary'),
            F1Score(average='binary'),
            ROCAUC()
        ]
