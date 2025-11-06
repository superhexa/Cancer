from pathlib import Path
from typing import Dict, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TensorBoardLogger:
    
    def __init__(self, log_dir: str, enabled: bool = True):
        self._log_dir = Path(log_dir)
        self._enabled = enabled
        self._writer = None
        
        if self._enabled:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(str(self._log_dir))
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._enabled and self._writer:
            self._writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int) -> None:
        if self._enabled and self._writer:
            self._writer.add_scalars(main_tag, values, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        if self._enabled and self._writer:
            self._writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        if self._enabled and self._writer:
            self._writer.add_image(tag, image, step)
    
    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        if self._enabled and self._writer:
            self._writer.add_figure(tag, figure, step)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        if self._enabled and self._writer:
            self._writer.add_text(tag, text, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        if self._enabled and self._writer:
            self._writer.add_graph(model, input_tensor)
    
    def close(self) -> None:
        if self._writer:
            self._writer.close()


class TrainingMonitor:
    
    def __init__(self, logger: TensorBoardLogger, log_interval: int = 50):
        self._logger = logger
        self._log_interval = log_interval
        self._training_losses = []
        self._validation_losses = []
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        metrics: Dict[str, float]
    ) -> None:
        global_step = epoch * 1000 + step
        
        self._logger.log_scalar('train/loss', loss, global_step)
        
        for name, value in metrics.items():
            self._logger.log_scalar(f'train/{name}', value, global_step)
        
        self._training_losses.append(loss)
    
    def log_validation_epoch(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float]
    ) -> None:
        self._logger.log_scalar('val/loss', loss, epoch)
        
        for name, value in metrics.items():
            self._logger.log_scalar(f'val/{name}', value, epoch)
        
        self._validation_losses.append(loss)
    
    def log_learning_rate(self, epoch: int, lr: float) -> None:
        self._logger.log_scalar('train/learning_rate', lr, epoch)
    
    def log_model_parameters(self, model: torch.nn.Module, epoch: int) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._logger.log_histogram(f'parameters/{name}', param, epoch)
                if param.grad is not None:
                    self._logger.log_histogram(f'gradients/{name}', param.grad, epoch)
    
    def log_confusion_matrix(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: list,
        epoch: int
    ) -> None:
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(targets, predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        
        self._logger.log_figure('val/confusion_matrix', fig, epoch)
        plt.close(fig)
    
    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self._training_losses, label='Training Loss', alpha=0.7)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self._validation_losses, label='Validation Loss', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss Over Epochs')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)


class ConsoleLogger:
    
    def __init__(self, verbosity: int = 2):
        self._verbosity = verbosity
    
    def info(self, message: str) -> None:
        if self._verbosity >= 1:
            print(f"[INFO] {message}")
    
    def debug(self, message: str) -> None:
        if self._verbosity >= 2:
            print(f"[DEBUG] {message}")
    
    def warning(self, message: str) -> None:
        print(f"[WARNING] {message}")
    
    def error(self, message: str) -> None:
        print(f"[ERROR] {message}")
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float]
    ) -> None:
        self.info(f"\nEpoch {epoch} Summary:")
        self.info(f"  Train Loss: {train_loss:.4f}")
        self.info(f"  Val Loss: {val_loss:.4f}")
        for name, value in metrics.items():
            self.info(f"  {name.capitalize()}: {value:.4f}")
