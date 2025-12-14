"""
Emotion-to-voice description mappings for TTS.

These descriptions tell the TTS model how to vocalize text with different emotions.
Each description captures the tone, delivery style, pacing, and pitch characteristic
of that emotion.
"""

from typing import Dict


EMOTION_DESCRIPTIONS: Dict[str, str] = {
    "angry": (
        "An angry speaker with tense tone, firm and sharp delivery,"
        " moderate speed and slightly raised pitch; recording is clean and close."
    ),
    "disgust": (
        "A speaker with a disgusted tone, constricted articulation, and clipped phrases;"
        " moderate pace and mid pitch; high-quality close recording."
    ),
    "fear": (
        "A fearful speaker with breathy tension, cautious delivery, and uneven rhythm;"
        " moderate-slow speed and slightly higher pitch; clean recording."
    ),
    "happy": (
        "A cheerful speaker with bright tone, upbeat rhythm, and lively delivery;"
        " moderate-fast speed and slightly higher pitch; very clear close recording."
    ),
    "neutral": (
        "A neutral speaker with even tone, steady pacing, and clear articulation;"
        " moderate speed and mid pitch; high-quality near-field recording."
    ),
    "sad": (
        "A sad speaker with soft tone, gentle delivery, and slower rhythm;"
        " reduced intensity and slightly lower pitch; clean, close recording."
    ),
    "surprise": (
        "A surprised speaker with sudden emphasis, energetic articulation, and dynamic rhythm;"
        " moderate-fast speed and higher pitch; clear and close recording."
    ),
}


def get_emotion_description(emotion: str) -> str:
    """Get voice description for an emotion.

    Args:
        emotion: Emotion label

    Returns:
        Voice description string for TTS, or neutral if emotion not found
    """
    return EMOTION_DESCRIPTIONS.get(emotion, EMOTION_DESCRIPTIONS["neutral"])
