# Module Development Guide

This guide explains how to add new modules to the SentiSign project.

## Module Structure

Each module in SentiSign follows a consistent structure:

```
sentisign/
└── your_module/
    ├── __init__.py       # Package initialization, exports
    ├── model.py          # Model architecture (if ML-based)
    ├── inference.py      # Inference and processing logic
    ├── utils.py          # Module-specific utilities (optional)
    └── README.md         # Module documentation and API
```

## Step-by-Step Guide

### Step 1: Create Module Directory

```bash
mkdir -p sentisign/your_module
touch sentisign/your_module/__init__.py
```

### Step 2: Define Model Architecture (if applicable)

Create `sentisign/your_module/model.py`:

```python
"""
Your module model architecture.

Brief description of what this model does.
"""

import torch
import torch.nn as nn


class YourModel(nn.Module):
    """Model description.

    Architecture details, input/output specifications, etc.

    Args:
        param1: Description
        param2: Description
    """

    def __init__(self, param1: int, param2: int):
        super().__init__()

        # Define layers
        self.layer1 = nn.Linear(param1, param2)
        # ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, ...)

        Returns:
            Output tensor of shape (batch, ...)
        """
        x = self.layer1(x)
        # ...
        return x
```

### Step 3: Implement Inference Logic

Create `sentisign/your_module/inference.py`:

```python
"""
Your module inference functions.

Provides functions for loading models, preprocessing data,
and running predictions.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .model import YourModel


def load_model(
    model_path: str,
    device: torch.device,
    **model_kwargs
) -> nn.Module:
    """Load model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: torch.device for model
        **model_kwargs: Additional model parameters

    Returns:
        Loaded model in eval mode

    Raises:
        RuntimeError: If checkpoint format is unsupported
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        # Extract state dict and create model
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = YourModel(**model_kwargs)
        model.load_state_dict(state_dict)
    else:
        raise RuntimeError("Unsupported checkpoint format")

    return model.eval().to(device)


def preprocess(input_data, **kwargs):
    """Preprocess input data for model.

    Args:
        input_data: Raw input data
        **kwargs: Preprocessing parameters

    Returns:
        Preprocessed tensor
    """
    # Implement preprocessing
    pass


def predict(
    model: nn.Module,
    input_data,
    device: torch.device,
    **kwargs
) -> Tuple:
    """Run prediction.

    Args:
        model: Trained model
        input_data: Input data
        device: torch.device
        **kwargs: Additional parameters

    Returns:
        Prediction results
    """
    # Preprocess
    tensor = preprocess(input_data, **kwargs)
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(tensor)

    # Post-process and return
    result = output.cpu().numpy()
    return result
```

### Step 4: Package Initialization

Create `sentisign/your_module/__init__.py`:

```python
"""
Your Module for SentiSign.

Brief description of module functionality.
"""

from .model import YourModel
from .inference import load_model, predict

__all__ = [
    "YourModel",
    "load_model",
    "predict",
]
```

### Step 5: Create Module README

Create `sentisign/your_module/README.md`:

```markdown
# Your Module Name

Brief description of what this module does.

## Overview

Detailed explanation of the module's purpose and capabilities.

## Features

- Feature 1
- Feature 2
- Feature 3

## API Usage

### Loading a Model

\`\`\`python
from sentisign.your_module import load_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("path/to/model.pth", device=device)
\`\`\`

### Making Predictions

\`\`\`python
from sentisign.your_module import predict

result = predict(model, input_data, device)
print(f"Result: {result}")
\`\`\`

## Model Architecture

Details about the model architecture, input/output specifications, etc.

## Training

Information about how the model was trained (dataset, hyperparameters, etc.)

## Performance

Performance metrics, benchmarks, etc.

## Files

- `model.py` - Model architecture
- `inference.py` - Inference functions
- `utils.py` - Helper utilities
```

### Step 6: Create Demo Script

Create `demos/run_your_module.py`:

```python
#!/usr/bin/env python3
"""
Your module demonstration script.

Description of what this demo does.

Usage:
    python demos/run_your_module.py
    python demos/run_your_module.py --option value
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentisign.your_module import load_model, predict
from sentisign.utils.device import get_device, print_device_info


def main():
    parser = argparse.ArgumentParser(
        description="Your module demo"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/your_module/model.pth"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device"
    )
    # Add more arguments as needed

    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    print_device_info(device)

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(str(args.model), device=device)
    print("Model loaded successfully!")

    # Run demo
    print("\nRunning demo...")
    # Implement demo logic

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
```

### Step 7: Add Unit Tests

Create `tests/test_your_module.py`:

```python
"""
Unit tests for your module.
"""

import pytest
import torch
from sentisign.your_module.model import YourModel


def test_model_initialization():
    """Test model can be initialized."""
    model = YourModel(param1=10, param2=20)
    assert model is not None


def test_model_forward():
    """Test model forward pass."""
    model = YourModel(param1=10, param2=20)
    input_tensor = torch.randn(1, 10)

    output = model(input_tensor)

    assert output.shape == (1, 20)
    assert not torch.isnan(output).any()


def test_model_different_batch_sizes():
    """Test model handles different batch sizes."""
    model = YourModel(param1=10, param2=20)

    for batch_size in [1, 4, 16]:
        input_tensor = torch.randn(batch_size, 10)
        output = model(input_tensor)
        assert output.shape == (batch_size, 20)


# Add more tests...
```

### Step 8: Update Main README

Add your module to the main `README.md`:

```markdown
### Your Module Name

Brief description.
- Key feature 1
- Key feature 2
- See [sentisign/your_module/README.md](sentisign/your_module/README.md)
```

### Step 9: Update Architecture Documentation

Add your module to `docs/ARCHITECTURE.md`:

```markdown
### N. Your Module Name

**Status**: 🚧 In Development / ✅ Complete

**Responsibility**: What this module does

**Architecture**:
- Technical details

**Input**:
- Input specifications

**Output**:
- Output specifications

**Files**:
- `sentisign/your_module/model.py` - Architecture
- `sentisign/your_module/inference.py` - Inference
```

## Integration Guidelines

### Connecting to Other Modules

If your module needs to interact with other modules:

```python
# In your_module/inference.py
from sentisign.emotion.inference import predict_emotion
from sentisign.utils.device import get_device

def combined_prediction(input_data, emotion_model, your_model, device):
    """Example of module integration."""
    # Use emotion detection
    emotion, conf = predict_emotion(emotion_model, input_data, device)

    # Use your module
    your_result = predict(your_model, input_data, device)

    # Combine results
    return emotion, your_result
```

### Using Shared Utilities

Always use shared utilities from `sentisign/utils/`:

```python
from sentisign.utils.device import get_device
from sentisign.utils.video import initialize_camera, detect_faces
```

## Best Practices

### 1. Modular Design

- Keep modules independent
- Use clear interfaces (function signatures)
- Avoid tight coupling between modules

### 2. Error Handling

```python
def load_model(model_path: str, device: torch.device):
    """Load model with proper error handling."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # ... rest of loading logic
```

### 3. Type Hints

Always use type hints for better code clarity:

```python
from typing import List, Tuple, Optional
import torch
import numpy as np

def process_batch(
    inputs: List[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Process batch with type hints."""
    pass
```

### 4. Documentation

- Write comprehensive docstrings
- Include examples in module README
- Keep documentation in sync with code

### 5. Testing

- Test edge cases
- Test with different input sizes
- Test error conditions

## Example: Sign Language Module

Here's how the sign language module should be structured:

```
sentisign/sign_language/
├── __init__.py
├── model.py              # Transformer architecture
├── inference.py          # Hand detection + recognition
├── preprocessing.py      # Hand keypoint extraction
└── README.md             # Module documentation
```

Key files:

**model.py**:
```python
class SignLanguageTransformer(nn.Module):
    """Transformer for sign language recognition."""
    def __init__(self, vocab_size: int, d_model: int = 512):
        super().__init__()
        # Define transformer layers
```

**inference.py**:
```python
def detect_hands(frame: np.ndarray) -> List[np.ndarray]:
    """Detect hands in frame."""
    pass

def recognize_sign(frames: List[np.ndarray], model, device) -> str:
    """Recognize sign from frame sequence."""
    pass
```

**Demo** (`demos/run_sign_to_text.py`):
```python
# Capture frames
frames = capture_frames(duration=2.0)

# Detect and recognize
sign_text = recognize_sign(frames, model, device)

print(f"Recognized: {sign_text}")
```

## Checklist

Before submitting a new module:

- [ ] Module directory created with proper structure
- [ ] Model architecture implemented (if applicable)
- [ ] Inference functions implemented
- [ ] Module README written with examples
- [ ] Demo script created and tested
- [ ] Unit tests added
- [ ] Main README updated
- [ ] ARCHITECTURE.md updated
- [ ] Type hints added throughout
- [ ] Error handling implemented
- [ ] Code follows style guidelines
- [ ] All demos work correctly

## Getting Help

If you're stuck:
- Check existing modules (emotion, tts) for examples
- Ask in GitHub issues
- Request code review early for architectural decisions

Good luck building your module! 🚀
