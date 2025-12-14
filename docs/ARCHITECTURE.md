# SentiSign Architecture

This document describes the system architecture and data flow of the SentiSign project.

## System Overview

SentiSign is a multi-modal system that combines sign language recognition with facial emotion detection to produce emotionally expressive speech output.

## Pipeline Architecture

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │         Webcam Input                │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
    ┌───────────────────────────┐   ┌────────────────────────┐
    │  Sign Language            │   │  Facial Emotion        │
    │  Recognition Module       │   │  Detection Module      │
    │  (Transformer)            │   │  (ResNet)              │
    │                           │   │                        │
    │  Status: In Development   │   │  Status: Complete ✓    │
    └───────────┬───────────────┘   └────────┬───────────────┘
                │                            │
                │ Recognized Text            │ Emotion Label
                │                            │
                └────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │  Context-Aware             │
                │  Integration Module        │
                │  (SLM)                     │
                │                            │
                │  Status: Planned           │
                └────────────┬───────────────┘
                             │
                             │ Contextual Text
                             │
                             ▼
                ┌────────────────────────────┐
                │  Expressive TTS Module     │
                │  (Parler TTS)              │
                │                            │
                │  Status: Complete ✓        │
                └────────────┬───────────────┘
                             │
                             │ Audio Waveform
                             │
                             ▼
                ┌────────────────────────────┐
                │  Speaker Output            │
                └────────────────────────────┘
```

## Module Details

### 1. Webcam Input

**Responsibility**: Capture video frames from camera

**Technology**:
- OpenCV VideoCapture
- Real-time frame processing
- Configurable camera selection and parameters

**Output**:
- BGR video frames (OpenCV format)
- Frame rate: typically 30 FPS

---

### 2. Sign Language Recognition Module

**Status**: 🚧 In Development

**Responsibility**: Recognize sign language gestures and translate to text

**Architecture** (Planned):
- Hand detection and tracking
- Temporal sequence modeling with Transformer
- Gesture-to-text mapping

**Input**:
- Video frame sequence
- Hand keypoints / bounding boxes

**Output**:
- Recognized text string
- Confidence scores

**Technical Details**:
- Model: Transformer encoder
- Input size: TBD
- Vocabulary: Common sign language phrases
- Latency target: < 500ms

---

### 3. Facial Emotion Detection Module

**Status**: ✅ Complete

**Responsibility**: Detect and classify facial emotions in real-time

**Architecture**:
- ResNet-inspired CNN (CNN_NeuralNet)
- 4 convolutional blocks (64 → 128 → 256 → 512)
- 2 residual connections
- 7-class softmax classifier

**Input**:
- Face ROI: 44×44 RGB image
- Normalized with ImageNet statistics

**Output**:
- Emotion label: {angry, disgust, fear, happy, neutral, sad, surprise}
- Confidence score: [0, 1]

**Technical Details**:
- Model parameters: ~25 MB
- Inference time: ~5-10 ms (GPU), ~50 ms (CPU)
- Preprocessing: Haar Cascade face detection
- Optional CLAHE for contrast enhancement

**Files**:
- `sentisign/emotion/model.py` - Model architecture
- `sentisign/emotion/inference.py` - Inference pipeline
- `models/emotion/resnet_emotion.pth` - Pretrained weights

---

### 4. Context-Aware Integration Module

**Status**: 📋 Planned

**Responsibility**: Combine sign language text with emotion context using SLM

**Architecture** (Proposed):
- Small Language Model (SLM) for text generation
- Prompt engineering with emotion context
- Context window management

**Input**:
- Sign language text (from module 2)
- Emotion label (from module 3)
- Conversation history (optional)

**Output**:
- Contextually appropriate text
- Emotion-enhanced phrasing

**Technical Considerations**:
- Model selection: LLaMA, Phi, or similar small LM
- Latency: Target < 200ms
- Prompt template design
- Emotion weighting strategies

**Example Flow**:
```
Input:
  - Sign text: "hello how are you"
  - Emotion: "happy"

SLM Processing:
  - Apply emotion context
  - Natural phrasing

Output:
  - "Hello! How are you doing today?"
```

---

### 5. Expressive TTS Module

**Status**: ✅ Complete

**Responsibility**: Synthesize emotionally expressive speech

**Architecture**:
- Parler TTS (Hugging Face model)
- Emotion-conditioned voice descriptions
- High-quality audio generation

**Input**:
- Text to synthesize
- Emotion label

**Output**:
- Audio waveform (WAV format)
- Sample rate: 24 kHz

**Technical Details**:
- Model: `parler-tts/parler-tts-mini-v1`
- Model size: ~1 GB (downloaded on first run)
- Inference time: ~2-3s per sentence (GPU), ~10-15s (CPU)
- 7 preset emotion voice profiles

**Voice Characteristics by Emotion**:
- **Happy**: Bright, upbeat, lively, faster pace
- **Sad**: Soft, gentle, slower, lower pitch
- **Angry**: Tense, firm, sharp, raised pitch
- **Fear**: Breathy, cautious, uneven rhythm
- **Neutral**: Even, steady, clear, moderate
- **Surprise**: Sudden emphasis, energetic, dynamic
- **Disgust**: Constricted, clipped, mid pitch

**Files**:
- `sentisign/tts/emotion_descriptions.py` - Emotion-to-voice mappings
- `sentisign/tts/expressive_tts.py` - Synthesis functions

---

## Data Flow

### Current Implementation (Emotion + TTS only)

```
1. Camera captures frame
2. Haar Cascade detects face(s)
3. Face ROI(s) extracted and preprocessed
4. CNN model predicts emotion(s)
5. Dominant emotion selected (if multiple faces/frames)
6. Text + emotion passed to TTS
7. Parler TTS generates audio
8. Audio saved/played
```

### Future Full Pipeline

```
1. Camera captures frame
2. Parallel processing:
   a. Hand detection → Sign recognition → Text
   b. Face detection → Emotion detection → Emotion label
3. SLM integration:
   - Input: Sign text + Emotion label
   - Output: Contextual text
4. TTS synthesis with emotion-appropriate voice
5. Audio output to speaker
```

## Shared Utilities

### Device Management (`sentisign/utils/device.py`)
- Auto-detection of CUDA/MPS/CPU
- Device selection and configuration
- Performance optimization

### Video Processing (`sentisign/utils/video.py`)
- Camera initialization and management
- Face detection (Haar Cascade)
- Annotation drawing
- CLAHE preprocessing

## Performance Considerations

### Latency Targets

| Module | Target Latency |
|--------|---------------|
| Face Detection | < 20 ms |
| Emotion Classification | < 20 ms |
| Sign Recognition | < 500 ms |
| SLM Integration | < 200 ms |
| TTS Synthesis | < 3 s |
| **Total Pipeline** | **< 4 s** |

### Hardware Recommendations

**Minimum**:
- CPU: 4 cores, 2.5+ GHz
- RAM: 8 GB
- Webcam: 720p, 30 FPS

**Recommended**:
- CPU: 8+ cores, 3.0+ GHz
- GPU: NVIDIA GTX 1060 or better (6+ GB VRAM)
- RAM: 16 GB
- Webcam: 1080p, 60 FPS

## Scalability

The modular architecture allows for:
- Independent module development and testing
- Easy model upgrades (swap in better models)
- Flexible deployment (use only needed modules)
- Future extensions (e.g., multiple languages, custom emotions)

## Error Handling

Each module implements graceful degradation:
- No face detected → skip emotion (or use "neutral")
- No hands detected → no sign recognition
- Model loading failure → informative error message
- Camera unavailable → clear user guidance

## Future Enhancements

1. **Multi-language Support**: Extend to multiple sign languages
2. **Continuous Recognition**: Real-time streaming instead of windowed
3. **Personalization**: User-specific emotion profiles
4. **Mobile Deployment**: Optimize for mobile devices
5. **Cloud Integration**: Optional cloud-based processing
