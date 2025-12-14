#!/usr/bin/env python3
"""
Emotion detection with text-to-speech synthesis.

This demo captures emotion from webcam for a specified duration, then synthesizes
speech using the dominant detected emotion.

Usage:
    python demos/run_emotion_tts.py
    python demos/run_emotion_tts.py --duration 10 --text "Hello, how are you today?"

Press 'q' or ESC to end capture early.
"""

import argparse
import sys
import time
from pathlib import Path
from collections import Counter

import cv2
import torch

# Add parent directory to path to import sentisign
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentisign.emotion.inference import load_model, predict_emotions_batch, EMOTION_LABELS
from sentisign.utils.device import get_device, print_device_info
from sentisign.utils.video import (
    initialize_camera,
    load_face_detector,
    detect_faces,
    draw_annotations,
    create_clahe,
)
from sentisign.tts.expressive_tts import synthesize_speech


def main():
    parser = argparse.ArgumentParser(
        description="Webcam emotion detection with TTS synthesis"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/emotion/resnet_emotion.pth"),
        help="Path to emotion detection model",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Capture duration in seconds"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hey, how are you doing today?",
        help="Text to synthesize with TTS"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parler_tts_out.wav"),
        help="Output WAV file path"
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        default=True,
        help="Flip frame horizontally"
    )
    parser.add_argument(
        "--no-flip",
        dest="flip",
        action="store_false",
        help="Don't flip frame"
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.3,
        help="Face detection scale factor"
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Face detection minimum neighbors"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=44,
        help="Input image size for model"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device"
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Use CLAHE preprocessing"
    )

    args = parser.parse_args()

    # Setup device
    device = get_device(args.device)
    print_device_info(device)

    # Load model
    print(f"\nLoading emotion model from {args.model}...")
    model = load_model(
        model_path=str(args.model),
        num_classes=len(EMOTION_LABELS),
        in_channels=3,
        linear_in_features=2048,
        device=device,
    )
    print("Model loaded successfully!")

    # Initialize camera and face detector
    print(f"Initializing camera {args.camera}...")
    cap = initialize_camera(args.camera)
    face_detector = load_face_detector()

    # Optional CLAHE
    clahe = create_clahe() if args.clahe else None

    print(f"\nCapturing emotions for {args.duration} seconds...")
    print("Press 'q' or ESC to end early\n")

    # Collect emotion predictions during capture window
    emotion_counts = Counter()
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # Detect faces
            faces = detect_faces(
                frame,
                face_detector,
                scale_factor=args.scale_factor,
                min_neighbors=args.min_neighbors,
            )

            # Predict emotions
            if faces:
                annotations = predict_emotions_batch(
                    model,
                    frame,
                    faces,
                    device,
                    img_size=args.img_size,
                    in_channels=3,
                    imagenet_norm=True,
                    clahe=clahe,
                )

                # Count emotions
                for _, emotion, _ in annotations:
                    emotion_counts[emotion] += 1

                # Draw annotations
                draw_annotations(frame, annotations)

            # Display frame with timer
            elapsed = time.time() - start_time
            remaining = max(0, args.duration - elapsed)
            cv2.putText(
                frame,
                f"Time remaining: {remaining:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("SentiSign - Emotion Capture", frame)

            # Check for timeout or quit
            if elapsed >= args.duration:
                break
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # 'q' or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Flush window events
        for _ in range(5):
            cv2.waitKey(1)

    # Determine dominant emotion
    if emotion_counts:
        dominant_emotion = emotion_counts.most_common(1)[0][0]
    else:
        dominant_emotion = "neutral"
        print("No emotions detected, using neutral")

    # Print emotion summary
    print("\n=== Emotion Summary ===")
    for emotion, count in emotion_counts.most_common():
        print(f"  {emotion:>8s}: {count:>3d}")
    print(f"\nDominant emotion: {dominant_emotion}")

    # Synthesize speech
    print(f"\nSynthesizing speech with '{dominant_emotion}' emotion...")
    print(f"Text: \"{args.text}\"")

    # Use CPU for TTS if GPU memory is limited
    tts_device = torch.device("cpu") if device.type == "mps" else device

    output_path = synthesize_speech(
        text=args.text,
        emotion=dominant_emotion,
        output_path=args.output,
        device=tts_device,
    )

    print(f"\nAudio saved to: {output_path}")
    print("Demo complete!")


if __name__ == "__main__":
    main()
