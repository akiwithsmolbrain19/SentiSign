# Emotion Detection Module

Facial emotion recognition using a ResNet-inspired CNN architecture.

## Overview

This module provides real-time emotion detection from facial images. It includes:
- CNN model architecture (ResNet-inspired)
- Model loading and inference functions
- Image preprocessing utilities
- Support for both grayscale and RGB inputs

## Supported Emotions

The model classifies faces into 7 emotion categories:
1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## API Usage

### Loading a Model

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

### Predicting Emotion from a Face ROI

```python
from sentisign.emotion.inference import predict_emotion
import cv2

# Load image and extract face ROI
frame = cv2.imread("face.jpg")
face_roi = frame[y:y+h, x:x+w]  # Face bounding box

# Predict emotion
emotion, confidence = predict_emotion(
    model=model,
    face_roi=face_roi,
    device=device,
    img_size=44,
    in_channels=3,
    imagenet_norm=True
)

print(f"Detected: {emotion} ({confidence*100:.1f}% confidence)")
```

### Batch Prediction

```python
from sentisign.emotion.inference import predict_emotions_batch

# Detect multiple faces in a frame
faces = [(x1, y1, w1, h1), (x2, y2, w2, h2)]  # Face bounding boxes
results = predict_emotions_batch(
    model=model,
    frame=frame,
    face_boxes=faces,
    device=device
)

for box, emotion, confidence in results:
    print(f"Face at {box}: {emotion} ({confidence*100:.1f}%)")
```

## Model Architecture

The `CNN_NeuralNet` class implements a ResNet-inspired architecture:
- 4 convolutional blocks with increasing channels (64 → 128 → 256 → 512)
- 2 residual blocks with skip connections
- MaxPooling for spatial dimension reduction
- Fully connected classifier layer

## Preprocessing

Images are preprocessed with:
1. Resize to 44×44 pixels
2. Convert BGR to RGB (or grayscale)
3. Normalize to [0, 1]
4. Optional ImageNet normalization
5. Optional CLAHE for contrast enhancement

## Files

- `model.py` - CNN_NeuralNet architecture
- `inference.py` - Model loading, preprocessing, and prediction functions
