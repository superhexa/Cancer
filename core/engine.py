from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from monitoring.metrics import MetricsTracker
from monitoring.visualization import TensorBoardLogger, ConsoleLogger, TrainingMonitor
from infrastructure.checkpointing import CheckpointManager, EarlyStoppingMonitor


class TrainingEngine:
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config,
        device: str
    ):
        self._model = model.to(device)
        self._criterion = criterion.to(device)
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._config = config
        self._device = device
        
        self._tensorboard = TensorBoardLogger(
            log_dir=config.get('paths', 'logs'),
            enabled=config.get('logging', 'tensorboard', default=True)
        )
        
        self._console = ConsoleLogger(verbosity=2)
        self._monitor = TrainingMonitor(self._tensorboard)
        
        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.get('paths', 'checkpoints'),
            save_top_k=config.get('logging', 'save_top_k', default=3)
        )
        
        self._early_stopping = EarlyStoppingMonitor(
            patience=config.get('training', 'patience'),
            mode=config.get('training', 'mode', default='maximize')
        )
        
        self._train_metrics = None
        self._val_metrics = None
        self._current_epoch = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metrics_tracker: MetricsTracker
    ) -> None:
        epochs = self._config.epochs
        
        self._console.info(f"Starting training for {epochs} epochs")
        self._console.info(f"Device: {self._device}")
        self._console.info(f"Model parameters: {sum(p.numel() for p in self._model.parameters() if p.requires_grad):,}")
        
        for epoch in range(1, epochs + 1):
            self._current_epoch = epoch
            
            train_loss, train_metrics = self._train_epoch(train_loader, metrics_tracker)
            val_loss, val_metrics = self._validate_epoch(val_loader, metrics_tracker)
            
            self._monitor.log_validation_epoch(epoch, val_loss, val_metrics)
            self._monitor.log_learning_rate(epoch, self._get_current_lr())
            
            if epoch % self._config.get('logging', 'checkpoint_interval', default=5) == 0:
                self._monitor.log_model_parameters(self._model, epoch)
            
            self._console.log_epoch_summary(epoch, train_loss, val_loss, val_metrics)
            
            metric_value = val_metrics.get(self._config.get('training', 'metric'))
            is_best = self._early_stopping.update(metric_value)
            
            self._checkpoint_manager.save(
                model=self._model,
                optimizer=self._optimizer,
                scheduler=self._scheduler,
                epoch=epoch,
                metrics=val_metrics,
                is_best=is_best
            )
            
            if self._early_stopping.should_stop:
                self._console.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            if self._scheduler:
                self._scheduler.step()
        
        self._tensorboard.close()
        self._console.info("Training completed!")
    
    def _train_epoch(
        self,
        data_loader: DataLoader,
        metrics_tracker: MetricsTracker
    ) -> tuple:
        self._model.train()
        metrics_tracker.reset()
        
        total_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {self._current_epoch} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            
            self._optimizer.zero_grad()
            
            outputs = self._model(inputs)
            loss = self._criterion(outputs, targets)
            
            loss.backward()
            self._optimizer.step()
            
            total_loss += loss.item()
            metrics_tracker.update(outputs.detach(), targets)
            
            if batch_idx % self._config.get('logging', 'console_interval', default=50) == 0:
                current_metrics = metrics_tracker.compute()
                self._monitor.log_training_step(
                    self._current_epoch,
                    batch_idx,
                    loss.item(),
                    current_metrics
                )
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(data_loader)
        metrics = metrics_tracker.compute()
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def _validate_epoch(
        self,
        data_loader: DataLoader,
        metrics_tracker: MetricsTracker
    ) -> tuple:
        self._model.eval()
        metrics_tracker.reset()
        
        total_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {self._current_epoch} [Val]")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            
            outputs = self._model(inputs)
            loss = self._criterion(outputs, targets)
            
            total_loss += loss.item()
            metrics_tracker.update(outputs, targets)
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(data_loader)
        metrics = metrics_tracker.compute()
        
        return avg_loss, metrics
    
    def _get_current_lr(self) -> float:
        return self._optimizer.param_groups[0]['lr']
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        self._console.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = self._checkpoint_manager.load(
            checkpoint_path,
            self._model,
            self._optimizer,
            self._scheduler,
            self._device
        )
        self._current_epoch = checkpoint['epoch']
        self._console.info(f"Resumed from epoch {self._current_epoch}")


class OptimizerFactory:
    
    @staticmethod
    def create(config, model_parameters) -> torch.optim.Optimizer:
        optimizer_type = config.get('optimization', 'algorithm', default='adamw').lower()
        lr = config.learning_rate
        weight_decay = config.get('optimization', 'weight_decay', default=0.01)
        
        if optimizer_type == 'adamw':
            return torch.optim.AdamW(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=config.get('optimization', 'amsgrad', default=True)
            )
        elif optimizer_type == 'adam':
            return torch.optim.Adam(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(
                model_parameters,
                lr=lr,
                momentum=config.get('optimization', 'momentum', default=0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")


class SchedulerFactory:
    
    @staticmethod
    def create(config, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        scheduler_type = config.get('scheduler', 'type', default='cosine').lower()
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.get('scheduler', 'min_lr', default=1e-6)
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get('scheduler', 'step_size', default=10),
                gamma=config.get('scheduler', 'gamma', default=0.1)
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
