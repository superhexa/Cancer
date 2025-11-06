from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from monitoring.metrics import MetricsTracker
from monitoring.visualization import ConsoleLogger


class ModelEvaluator:
    
    def __init__(self, model: nn.Module, criterion: nn.Module, device: str):
        self._model = model.to(device)
        self._criterion = criterion.to(device)
        self._device = device
        self._console = ConsoleLogger(verbosity=2)
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        metrics_tracker: MetricsTracker
    ) -> Dict[str, float]:
        self._model.eval()
        metrics_tracker.reset()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(data_loader, desc="Evaluating")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            
            outputs = self._model(inputs)
            loss = self._criterion(outputs, targets)
            
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            metrics_tracker.update(outputs, targets)
        
        avg_loss = total_loss / len(data_loader)
        metrics = metrics_tracker.compute()
        metrics['loss'] = avg_loss
        
        self._console.info("\n=== Evaluation Results ===")
        self._console.info(f"Loss: {avg_loss:.4f}")
        for name, value in metrics.items():
            if name != 'loss':
                self._console.info(f"{name.capitalize()}: {value:.4f}")
        
        return metrics, np.array(all_predictions), np.array(all_targets)
    
    def compute_class_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: list
    ) -> Dict[str, Dict[str, float]]:
        from sklearn.metrics import classification_report
        
        report = classification_report(
            targets,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        self._console.info("\n=== Per-Class Metrics ===")
        for class_name in class_names:
            metrics = report[class_name]
            self._console.info(f"\n{class_name}:")
            self._console.info(f"  Precision: {metrics['precision']:.4f}")
            self._console.info(f"  Recall: {metrics['recall']:.4f}")
            self._console.info(f"  F1-Score: {metrics['f1-score']:.4f}")
        
        return report
    
    def compute_confusion_matrix(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(targets, predictions)


class TestRunner:
    
    def __init__(self, config):
        self._config = config
        self._console = ConsoleLogger(verbosity=2)
    
    def run(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        metrics_tracker: MetricsTracker,
        checkpoint_path: str
    ) -> None:
        device = torch.device(self._config.device if torch.cuda.is_available() else 'cpu')
        
        self._console.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        evaluator = ModelEvaluator(model, criterion, device)
        
        metrics, predictions, targets = evaluator.evaluate(test_loader, metrics_tracker)
        
        class_names = ['benign', 'malignant']
        class_metrics = evaluator.compute_class_metrics(predictions, targets, class_names)
        
        confusion_mat = evaluator.compute_confusion_matrix(predictions, targets)
        
        self._console.info("\n=== Confusion Matrix ===")
        self._console.info(f"\n{confusion_mat}")
        
        self._save_results(metrics, class_metrics, confusion_mat)
    
    def _save_results(
        self,
        metrics: Dict,
        class_metrics: Dict,
        confusion_matrix: np.ndarray
    ) -> None:
        import json
        from pathlib import Path
        
        results_dir = Path(self._config.get('paths', 'predictions'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'overall_metrics': metrics,
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix.tolist()
        }
        
        with open(results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self._console.info(f"\nResults saved to {results_dir / 'evaluation_results.json'}")
