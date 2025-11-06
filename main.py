import argparse
import random
import numpy as np
import torch
from pathlib import Path

from configuration.settings import ConfigurationManager
from infrastructure.datapipeline import DatasetFactory
from networks.architectures import ModelRegistry
from networks.objectives import LossFactory
from monitoring.metrics import MetricsFactory, MetricsTracker
from core.engine import TrainingEngine, OptimizerFactory, SchedulerFactory
from core.evaluator import TestRunner
from core.predictor import InferencePipeline, BatchPredictor


class Application:
    
    def __init__(self, config_path: str):
        self._config = ConfigurationManager(config_path)
        self._setup_environment()
        self._device = self._get_device()
    
    def _setup_environment(self) -> None:
        seed = self._config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _get_device(self) -> str:
        device_config = self._config.device
        if device_config == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def train(self) -> None:
        train_loader, val_loader = DatasetFactory.create_dataloaders(self._config)
        
        model = ModelRegistry.create(self._config)
        criterion = LossFactory.create(self._config)
        
        optimizer = OptimizerFactory.create(
            self._config,
            filter(lambda p: p.requires_grad, model.parameters())
        )
        
        scheduler = SchedulerFactory.create(self._config, optimizer)
        
        metrics = MetricsFactory.create_standard_metrics()
        metrics_tracker = MetricsTracker(metrics)
        
        engine = TrainingEngine(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self._config,
            device=self._device
        )
        
        engine.train(train_loader, val_loader, metrics_tracker)
    
    def evaluate(self, checkpoint_path: str) -> None:
        _, test_loader = DatasetFactory.create_dataloaders(self._config)
        
        model = ModelRegistry.create(self._config)
        criterion = LossFactory.create(self._config)
        
        metrics = MetricsFactory.create_standard_metrics()
        metrics_tracker = MetricsTracker(metrics)
        
        test_runner = TestRunner(self._config)
        test_runner.run(model, test_loader, criterion, metrics_tracker, checkpoint_path)
    
    def predict(self, image_path: str, checkpoint_path: str) -> None:
        model = ModelRegistry.create(self._config)
        
        pipeline = InferencePipeline(model, self._config, self._device)
        pipeline.load_checkpoint(checkpoint_path)
        
        result = pipeline.predict_single(image_path)
        
        return result
    
    def predict_batch(
        self,
        directory_path: str,
        checkpoint_path: str,
        export_format: str = 'json'
    ) -> None:
        model = ModelRegistry.create(self._config)
        
        batch_predictor = BatchPredictor(self._config)
        batch_predictor.predict_directory(
            model,
            directory_path,
            checkpoint_path,
            export_format
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Histopathology Cancer Detection Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'predict', 'predict_batch'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configuration/experiment.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to image for prediction'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        default=None,
        help='Path to directory for batch prediction'
    )
    
    parser.add_argument(
        '--export-format',
        type=str,
        default='json',
        choices=['json', 'csv'],
        help='Export format for batch predictions'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    app = Application(args.config)
    
    if args.mode == 'train':
        app.train()
    
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for evaluation mode")
        app.evaluate(args.checkpoint)
    
    elif args.mode == 'predict':
        if not args.image or not args.checkpoint:
            raise ValueError("--image and --checkpoint are required for prediction mode")
        app.predict(args.image, args.checkpoint)
    
    elif args.mode == 'predict_batch':
        if not args.directory or not args.checkpoint:
            raise ValueError("--directory and --checkpoint are required for batch prediction mode")
        app.predict_batch(args.directory, args.checkpoint, args.export_format)


if __name__ == '__main__':
    main()
