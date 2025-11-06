from pathlib import Path
from typing import Dict, Optional, List
import torch
import json


class CheckpointManager:
    
    def __init__(self, checkpoint_dir: str, save_top_k: int = 3):
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_top_k = save_top_k
        self._checkpoints = []
        self._best_score = None
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint_path = self._checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        metadata = {
            'epoch': epoch,
            'path': str(checkpoint_path),
            'metrics': metrics
        }
        
        self._checkpoints.append(metadata)
        self._manage_checkpoints()
        
        if is_best:
            best_path = self._checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self._save_metadata(best_path, metadata)
    
    def _manage_checkpoints(self) -> None:
        if len(self._checkpoints) > self._save_top_k:
            self._checkpoints.sort(
                key=lambda x: x['metrics'].get('val_accuracy', 0),
                reverse=True
            )
            
            for checkpoint_info in self._checkpoints[self._save_top_k:]:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists() and 'best' not in str(checkpoint_path):
                    checkpoint_path.unlink()
            
            self._checkpoints = self._checkpoints[:self._save_top_k]
    
    def _save_metadata(self, checkpoint_path: Path, metadata: Dict) -> None:
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Dict:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def load_best(
        self,
        model: torch.nn.Module,
        device: str = 'cuda'
    ) -> Dict:
        best_path = self._checkpoint_dir / 'best_model.pth'
        if not best_path.exists():
            raise FileNotFoundError("Best model checkpoint not found")
        return self.load(str(best_path), model, device=device)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        checkpoints = sorted(
            self._checkpoint_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        return str(checkpoints[0]) if checkpoints else None


class EarlyStoppingMonitor:
    
    def __init__(self, patience: int, mode: str = 'maximize', delta: float = 0.0):
        self._patience = patience
        self._mode = mode
        self._delta = delta
        self._counter = 0
        self._best_score = None
        self._should_stop = False
    
    def update(self, score: float) -> bool:
        if self._best_score is None:
            self._best_score = score
            return False
        
        if self._mode == 'maximize':
            improved = score > self._best_score + self._delta
        else:
            improved = score < self._best_score - self._delta
        
        if improved:
            self._best_score = score
            self._counter = 0
            return True
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._should_stop = True
            return False
    
    @property
    def should_stop(self) -> bool:
        return self._should_stop
    
    def reset(self) -> None:
        self._counter = 0
        self._best_score = None
        self._should_stop = False


class ModelSnapshot:
    
    @staticmethod
    def create_snapshot(
        model: torch.nn.Module,
        save_path: str,
        metadata: Optional[Dict] = None
    ) -> None:
        snapshot = {
            'model_state_dict': model.state_dict(),
            'architecture': model.__class__.__name__,
            'metadata': metadata or {}
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, save_path)
    
    @staticmethod
    def load_snapshot(
        model: torch.nn.Module,
        snapshot_path: str,
        device: str = 'cuda'
    ) -> Dict:
        snapshot = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(snapshot['model_state_dict'])
        return snapshot.get('metadata', {})
