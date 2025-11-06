from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigurationManager:
    
    def __init__(self, config_path: str):
        self._config_path = Path(config_path)
        self._config = self._load()
        self._validate()
    
    def _load(self) -> Dict[str, Any]:
        with open(self._config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate(self) -> None:
        required_sections = ['experiment', 'dataset', 'network', 'optimization', 'training']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required section: {section}")
    
    def get(self, *keys: str, default: Any = None) -> Any:
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result
    
    def set(self, value: Any, *keys: str) -> None:
        target = self._config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    
    @property
    def experiment_name(self) -> str:
        return self.get('experiment', 'name')
    
    @property
    def seed(self) -> int:
        return self.get('experiment', 'seed', default=42)
    
    @property
    def device(self) -> str:
        return self.get('experiment', 'device', default='cuda')
    
    @property
    def batch_size(self) -> int:
        return self.get('optimization', 'batch_size')
    
    @property
    def learning_rate(self) -> float:
        return self.get('optimization', 'learning_rate')
    
    @property
    def epochs(self) -> int:
        return self.get('training', 'epochs')
    
    @property
    def num_workers(self) -> int:
        return self.get('dataset', 'workers')
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
