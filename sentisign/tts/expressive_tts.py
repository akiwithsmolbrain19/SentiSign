"""
Expressive text-to-speech synthesis using Parler TTS.

This module provides emotion-aware speech synthesis by conditioning
the TTS model on emotion descriptions.
"""

from pathlib import Path
from typing import Union

import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

from .emotion_descriptions import get_emotion_description


def synthesize_speech(
    text: str,
    emotion: str,
    output_path: Union[str, Path],
    device: torch.device,
    model_name: str = "parler-tts/parler-tts-mini-v1",
) -> Path:
    """Synthesize speech with emotion-appropriate voice characteristics.

    Args:
        text: Text to synthesize
        emotion: Emotion label (e.g., "happy", "sad", "angry")
        output_path: Path to save the output WAV file
        device: torch.device for inference
        model_name: Hugging Face model ID for Parler TTS

    Returns:
        Path to the generated audio file

    Example:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> output = synthesize_speech(
        ...     "Hello, how are you?",
        ...     "happy",
        ...     "output.wav",
        ...     device
        ... )
    """
    output_path = Path(output_path)

    # Get emotion-specific voice description
    description = get_emotion_description(emotion)

    # Load model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize inputs
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Generate speech
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids
        )

    # Convert to audio array and save
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(str(output_path), audio_arr, model.config.sampling_rate)

    return output_path


def synthesize_with_custom_description(
    text: str,
    description: str,
    output_path: Union[str, Path],
    device: torch.device,
    model_name: str = "parler-tts/parler-tts-mini-v1",
) -> Path:
    """Synthesize speech with a custom voice description.

    Use this when you want full control over the voice characteristics
    instead of using preset emotion descriptions.

    Args:
        text: Text to synthesize
        description: Custom voice description
        output_path: Path to save the output WAV file
        device: torch.device for inference
        model_name: Hugging Face model ID for Parler TTS

    Returns:
        Path to the generated audio file

    Example:
        >>> device = torch.device("cpu")
        >>> output = synthesize_with_custom_description(
        ...     "This is a test.",
        ...     "A calm speaker with clear articulation and moderate pace.",
        ...     "custom_output.wav",
        ...     device
        ... )
    """
    output_path = Path(output_path)

    # Load model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize inputs
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Generate speech
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids
        )

    # Convert to audio array and save
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(str(output_path), audio_arr, model.config.sampling_rate)

    return output_path
