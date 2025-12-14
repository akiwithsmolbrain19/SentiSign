# Text-to-Speech Module

Emotion-aware speech synthesis using Parler TTS.

## Overview

This module provides expressive text-to-speech synthesis that adapts the voice
characteristics based on detected emotions. It uses Parler TTS with custom
emotion descriptions to generate natural-sounding speech.

## Features

- Emotion-conditioned speech synthesis
- 7 preset emotion voice profiles
- Custom voice description support
- High-quality audio output (WAV format)

## API Usage

### Basic Emotion-Aware TTS

```python
from sentisign.tts.expressive_tts import synthesize_speech
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthesize speech with happy emotion
output_path = synthesize_speech(
    text="Hello, how are you today?",
    emotion="happy",
    output_path="output.wav",
    device=device
)

print(f"Audio saved to: {output_path}")
```

### Custom Voice Description

```python
from sentisign.tts.expressive_tts import synthesize_with_custom_description

output_path = synthesize_with_custom_description(
    text="This is a custom voice.",
    description="A calm speaker with clear articulation and moderate pace.",
    output_path="custom.wav",
    device=device
)
```

### Getting Emotion Descriptions

```python
from sentisign.tts.emotion_descriptions import get_emotion_description

description = get_emotion_description("happy")
print(description)
# Output: "A cheerful speaker with bright tone, upbeat rhythm, ..."
```

## Emotion Voice Profiles

Each emotion has a carefully crafted voice description:

- **Angry**: Tense tone, firm delivery, moderate speed, raised pitch
- **Disgust**: Constricted articulation, clipped phrases, mid pitch
- **Fear**: Breathy tension, cautious delivery, uneven rhythm
- **Happy**: Bright tone, upbeat rhythm, lively delivery
- **Neutral**: Even tone, steady pacing, clear articulation
- **Sad**: Soft tone, gentle delivery, slower rhythm, lower pitch
- **Surprise**: Sudden emphasis, energetic articulation, dynamic rhythm

## Technical Details

### Model

- Uses `parler-tts/parler-tts-mini-v1` from Hugging Face
- Supports custom voice conditioning
- Outputs 24kHz audio in WAV format

### Performance

- First run downloads ~1GB model from Hugging Face
- GPU recommended for faster synthesis (~2-3 seconds per sentence)
- CPU mode works but is slower (~10-15 seconds per sentence)

### Device Recommendations

- CUDA: Fastest, use for production
- MPS (Apple Silicon): Good performance, but may have memory issues with large models
- CPU: Fallback option, slower but always works

## Files

- `emotion_descriptions.py` - Emotion-to-voice description mappings
- `expressive_tts.py` - Speech synthesis functions
