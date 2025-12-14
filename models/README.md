# SentiSign Models

This directory contains pre-trained model weights for the SentiSign project.

## Emotion Detection Models

### `emotion/resnet_emotion.pth`

**Architecture**: CNN_NeuralNet (ResNet-inspired)
**Task**: Facial emotion classification
**Classes**: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)

**Input Specifications**:
- Image size: 44×44 pixels
- Channels: 3 (RGB)
- Normalization: ImageNet normalization
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

**Model Architecture**:
- Convolutional layers: 4 blocks (64 → 128 → 256 → 512 channels)
- Residual connections: 2 blocks
- Classifier: 2048 → 7 output features
- Parameters: ~25 MB

**Usage**:
```python
from sentisign.emotion.inference import load_model, EMOTION_LABELS
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(
    model_path="models/emotion/resnet_emotion.pth",
    num_classes=len(EMOTION_LABELS),
    in_channels=3,
    linear_in_features=2048,
    device=device
)
```

## Sign Language Recognition Models

### `sign_language/` (Coming Soon)

Planned models for sign language gesture recognition using Transformer architecture.

## Model Management

- Models are stored in Git for easy access
- For large models in the future, consider using Git LFS
- Model checkpoints include full state dictionaries
- Compatible with PyTorch 2.0+
