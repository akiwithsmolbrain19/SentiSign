#!/usr/bin/env python3
"""
Real-time emotion detection from webcam.

This demo captures video from your webcam, detects faces, and displays
the predicted emotion for each face in real-time.

Usage:
    python demos/run_emotion_webcam.py
    python demos/run_emotion_webcam.py --camera 1 --no-flip

Press 'q' to quit the demo.
"""

import argparse
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="Real-time webcam emotion detection"
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
        "--flip",
        action="store_true",
        default=True,
        help="Flip frame horizontally (mirror mode)"
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
        help="Use CLAHE preprocessing for better contrast"
    )

    args = parser.parse_args()

    # Setup device
    device = get_device(args.device)
    print_device_info(device)

    # Load model
    print(f"Loading model from {args.model}...")
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

    print("\nStarting emotion detection...")
    print("Press 'q' to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
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

            # Predict emotions for all detected faces
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
                draw_annotations(frame, annotations)

            # Display frame
            cv2.imshow("SentiSign - Emotion Detection", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended")


if __name__ == "__main__":
    main()
