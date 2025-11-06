# ancer Detection Framework

Advanced deep learning framework for automated detection of malignant and benign breast tumors from histopathological images.

## Overview

This framework implements state-of-the-art computer vision techniques for binary classification of breast cancer histology samples. The system processes microscopic tissue images captured at multiple magnification levels (40X-400X) to distinguish between benign and malignant cases.

### Dataset Information

Utilizes the BreakHis dataset containing 7,909 microscopic images from 82 patients:
- **Benign samples**: 2,480 images (adenosis, fibroadenoma, phyllodes tumor, tubular adenoma)
- **Malignant samples**: 5,429 images (ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma)
- **Resolution**: 700x460 pixels, RGB, PNG format

## Architecture

The framework employs a modular design pattern with clear separation of concerns:

```
├── core/
│   ├── engine.py          # Training orchestration
│   ├── evaluator.py       # Model assessment
│   └── predictor.py       # Inference pipeline
├── infrastructure/
│   ├── datapipeline.py    # Data loading and preprocessing
│   └── checkpointing.py   # Model persistence
├── networks/
│   ├── architectures.py   # Neural network definitions
│   └── objectives.py      # Loss functions
├── monitoring/
│   ├── metrics.py         # Performance tracking
│   └── visualization.py   # TensorBoard integration
├── configuration/
│   └── settings.py        # Hyperparameters
└── main.py                # Entry point
```

## Installation

```bash
python -m venv venv_medical
source venv_medical/bin/activate  # Windows: venv_medical\Scripts\activate
pip install torch torchvision numpy pillow pyyaml tensorboard scikit-learn
```

## Usage

### Training

```bash
python main.py --mode train --config configuration/experiment.yaml
```

### Evaluation

```bash
python main.py --mode evaluate --checkpoint outputs/models/best.pth
```

### Inference

```bash
python main.py --mode predict --image path/to/image.png --checkpoint outputs/models/best.pth
```

## Configuration

Modify `configuration/experiment.yaml` to adjust:
- Network architecture (DenseNet121, ResNet50, EfficientNet)
- Batch size and learning rate
- Augmentation strategies
- Training duration

## Performance Metrics

The framework tracks:
- Classification accuracy
- Precision, recall, F1-score
- ROC-AUC
- Confusion matrix

Results are logged to TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

## Dataset Preparation

1. Download BreakHis from Kaggle
2. Extract to `datasets/breakhis/`
3. Structure should be:
```
datasets/breakhis/
├── benign/
│   ├── adenosis/
│   ├── fibroadenoma/
│   └── ...
└── malignant/
    ├── ductal_carcinoma/
    ├── lobular_carcinoma/
    └── ...
```

## Citation

If you use this framework, please cite:
```
@article{breakhis2024,
  title={Histopathology Cancer Detection Framework},
  author={Medical Imaging Research Lab},
  year={2024}
}
```

## License

Apache License 2.0 - See LICENSE file for details
