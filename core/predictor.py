from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np

from infrastructure.datapipeline import SingleImageLoader
from monitoring.visualization import ConsoleLogger


class InferencePipeline:
    
    def __init__(self, model: nn.Module, config, device: str):
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        self._config = config
        self._image_loader = SingleImageLoader(config)
        self._console = ConsoleLogger(verbosity=2)
        self._class_names = ['benign', 'malignant']
    
    @torch.no_grad()
    def predict_single(self, image_path: str) -> Dict:
        image_tensor = self._image_loader.load(image_path).to(self._device)
        
        outputs = self._model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        result = {
            'image_path': image_path,
            'predicted_class': self._class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                self._class_names[i]: probabilities[0, i].item()
                for i in range(len(self._class_names))
            }
        }
        
        self._console.info(f"\n=== Prediction Results ===")
        self._console.info(f"Image: {Path(image_path).name}")
        self._console.info(f"Predicted: {result['predicted_class']}")
        self._console.info(f"Confidence: {result['confidence']:.2%}")
        self._console.info("\nProbabilities:")
        for class_name, prob in result['probabilities'].items():
            self._console.info(f"  {class_name}: {prob:.2%}")
        
        return result
    
    @torch.no_grad()
    def predict_batch(self, image_paths: list) -> list:
        results = []
        for image_path in image_paths:
            result = self.predict_single(image_path)
            results.append(result)
        return results
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        self._console.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._console.info("Checkpoint loaded successfully")


class PredictionExporter:
    
    @staticmethod
    def export_to_json(predictions: list, output_path: str) -> None:
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    @staticmethod
    def export_to_csv(predictions: list, output_path: str) -> None:
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            if not predictions:
                return
            
            fieldnames = ['image_path', 'predicted_class', 'confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for pred in predictions:
                writer.writerow({
                    'image_path': pred['image_path'],
                    'predicted_class': pred['predicted_class'],
                    'confidence': pred['confidence']
                })


class BatchPredictor:
    
    def __init__(self, config):
        self._config = config
        self._console = ConsoleLogger(verbosity=2)
    
    def predict_directory(
        self,
        model: nn.Module,
        directory_path: str,
        checkpoint_path: str,
        export_format: str = 'json'
    ) -> None:
        device = torch.device(self._config.device if torch.cuda.is_available() else 'cpu')
        
        pipeline = InferencePipeline(model, self._config, device)
        pipeline.load_checkpoint(checkpoint_path)
        
        directory = Path(directory_path)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in directory.rglob('*')
            if p.suffix.lower() in image_extensions
        ]
        
        self._console.info(f"Found {len(image_paths)} images in {directory_path}")
        
        predictions = pipeline.predict_batch(image_paths)
        
        output_dir = Path(self._config.get('paths', 'predictions'))
        
        if export_format == 'json':
            output_path = output_dir / 'predictions.json'
            PredictionExporter.export_to_json(predictions, str(output_path))
        elif export_format == 'csv':
            output_path = output_dir / 'predictions.csv'
            PredictionExporter.export_to_csv(predictions, str(output_path))
        
        self._console.info(f"\nPredictions saved to {output_path}")
