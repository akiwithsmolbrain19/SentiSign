# Contributing to SentiSign

Thank you for your interest in contributing to SentiSign! This guide will help you get started.

## Development Workflow

### 1. Setup Development Environment

Follow the [SETUP.md](SETUP.md) guide, then install development dependencies:

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
uv pip install -e .
```

### 2. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding tests

### 3. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add docstrings to new functions/classes
- Update relevant documentation

### 4. Test Your Changes

```bash
# Test emotion detection demo
python demos/run_emotion_webcam.py

# Test TTS demo
python demos/run_emotion_tts.py --duration 3

# Run unit tests (when available)
pytest tests/
```

### 5. Commit Your Changes

Follow conventional commit format:

```bash
git add .
git commit -m "type: brief description

Detailed explanation of changes (if needed)
"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```bash
git commit -m "feat: add CLAHE preprocessing option to emotion detection"
git commit -m "fix: resolve camera permission error on Windows"
git commit -m "docs: update installation instructions for macOS"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub:
1. Go to the repository
2. Click "Pull Requests" → "New Pull Request"
3. Select your branch
4. Fill in the PR template
5. Request review from team members

## Code Style Guidelines

### Python Style

Follow PEP 8 with these preferences:

```python
# Use type hints
def predict_emotion(
    model: nn.Module,
    face_roi: np.ndarray,
    device: torch.device,
) -> Tuple[str, float]:
    ...

# Use docstrings (Google style)
def example_function(param1: int, param2: str) -> bool:
    """Brief one-line description.

    Longer description explaining the function's purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When parameter is invalid
    """
    ...

# Use meaningful variable names
emotion_label = "happy"  # Good
e = "happy"              # Bad

# Keep functions focused and small
# Prefer composition over long functions
```

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Third-party imports
import cv2
import numpy as np
import torch
import torch.nn as nn

# Local imports
from sentisign.emotion.model import CNN_NeuralNet
from sentisign.utils.device import get_device
```

### File Organization

```python
# Module docstring at top
"""
Brief module description.

Longer explanation of what this module does.
"""

# Imports

# Constants
CONSTANT_NAME = value

# Classes
class MyClass:
    ...

# Functions
def my_function():
    ...

# Main execution
if __name__ == "__main__":
    main()
```

## Adding New Modules

See [MODULE_GUIDE.md](MODULE_GUIDE.md) for detailed instructions on adding new modules.

Quick checklist:
- [ ] Create module directory in `sentisign/`
- [ ] Add `__init__.py`
- [ ] Implement core functionality
- [ ] Add module README
- [ ] Create demo script in `demos/`
- [ ] Add unit tests in `tests/`
- [ ] Update main README

## Testing Guidelines

### Writing Tests

Use pytest for unit tests:

```python
# tests/test_emotion.py
import pytest
import torch
from sentisign.emotion.model import CNN_NeuralNet

def test_model_forward_pass():
    """Test that model forward pass works correctly."""
    model = CNN_NeuralNet(in_channels=3, num_classes=7)
    input_tensor = torch.randn(1, 3, 44, 44)

    output = model(input_tensor)

    assert output.shape == (1, 7)
    assert not torch.isnan(output).any()

def test_model_with_different_input_size():
    """Test model handles different input sizes."""
    model = CNN_NeuralNet(in_channels=1, num_classes=7)
    input_tensor = torch.randn(1, 1, 44, 44)

    output = model(input_tensor)

    assert output.shape == (1, 7)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_emotion.py

# Run with coverage
pytest --cov=sentisign tests/
```

## Documentation

### Updating Documentation

When adding features, update:
1. Module README (in module directory)
2. Main README (if it's a major feature)
3. ARCHITECTURE.md (if architecture changes)
4. SETUP.md (if setup process changes)

### Writing Good Documentation

- Use clear, concise language
- Include code examples
- Explain the "why", not just the "what"
- Keep it up-to-date with code changes

## Git Workflow

### Before Starting Work

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature
```

### During Development

Commit frequently with meaningful messages:

```bash
git add path/to/changed/files
git commit -m "feat: add X functionality"
```

### Before Pushing

```bash
# Make sure you're up-to-date
git fetch origin
git rebase origin/main

# Run tests
pytest tests/

# Push your branch
git push origin feature/your-feature
```

### Pull Request Process

1. **Create PR** with clear title and description
2. **Link related issues** (e.g., "Closes #123")
3. **Request review** from team members
4. **Address feedback** promptly
5. **Keep PR focused** - one feature/fix per PR
6. **Update PR** if main branch changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Tested locally
- [ ] All demos work
- [ ] Unit tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)
Add screenshots for UI changes
```

## Common Tasks

### Adding a New Emotion

1. Update `EMOTION_LABELS` in `sentisign/emotion/inference.py`
2. Retrain model with new emotion class
3. Update TTS emotion descriptions in `sentisign/tts/emotion_descriptions.py`
4. Update documentation

### Improving Model Performance

1. Document current performance metrics
2. Make changes (architecture, preprocessing, etc.)
3. Compare new vs old performance
4. Update model checkpoint
5. Document improvements in PR

### Adding Command-Line Options

1. Add argument to demo script's `argparse` parser
2. Implement functionality
3. Update demo script docstring
4. Update README with example usage

## Communication

### Asking for Help

- Open an issue for bugs or questions
- Use clear, descriptive titles
- Include error messages and system info
- Provide minimal reproducible example

### Reporting Bugs

Include:
- Python version
- OS and version
- GPU info (if relevant)
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

### Suggesting Features

1. Check if already requested (search issues)
2. Open an issue with:
   - Clear use case
   - Proposed solution
   - Alternatives considered
3. Discuss before implementing large features

## Review Process

### For Reviewers

- Be constructive and respectful
- Focus on code quality, not personal preferences
- Test the changes locally
- Approve when ready, or request changes with clear feedback

### For Authors

- Respond to feedback promptly
- Ask for clarification if needed
- Make requested changes or explain why not
- Thank reviewers for their time

## License

By contributing to SentiSign, you agree that your contributions will be licensed under the project's license.

## Questions?

Feel free to ask questions in:
- GitHub issues
- Pull request comments
- Team chat (if applicable)

Thank you for contributing to SentiSign! 🎉
