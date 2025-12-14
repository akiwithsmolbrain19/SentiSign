"""
Video and camera utilities.

Provides functions for camera initialization, face detection, and drawing annotations.
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np


def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Initialize camera capture.

    Args:
        camera_index: Camera device index (typically 0 for default camera)

    Returns:
        OpenCV VideoCapture object

    Raises:
        RuntimeError: If camera cannot be opened
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Check permissions and device availability."
        )
    return cap


def load_face_detector() -> cv2.CascadeClassifier:
    """Load Haar Cascade face detector.

    Returns:
        Loaded face detector

    Raises:
        RuntimeError: If cascade classifier cannot be loaded
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    return face_detector


def detect_faces(
    frame: np.ndarray,
    face_detector: cv2.CascadeClassifier,
    scale_factor: float = 1.3,
    min_neighbors: int = 5,
    min_size: Optional[Tuple[int, int]] = None,
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in a frame.

    Args:
        frame: Input frame (BGR format)
        face_detector: Haar Cascade face detector
        scale_factor: Scale factor for multiscale detection
        min_neighbors: Minimum neighbors for detection
        min_size: Minimum face size (width, height). None for no minimum.

    Returns:
        List of face bounding boxes as (x, y, w, h) tuples
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if min_size is not None and min_size[0] > 0:
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
    else:
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
        )

    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def draw_emotion_label(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """Draw bounding box and emotion label on frame.

    Args:
        frame: Frame to draw on (modified in-place)
        box: Bounding box as (x, y, w, h)
        label: Emotion label text
        confidence: Confidence score (0-1)
        color: BGR color for box and text background
        font_scale: Font scale for label text
        thickness: Line thickness for box
    """
    x, y, w, h = box

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Prepare label text
    text = f"{label}: {confidence * 100:.1f}%"

    # Get text size for background rectangle
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # Draw background rectangle for text
    top_left = (x, y - th - baseline)
    bottom_right = (x + tw, y)
    cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)

    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),  # Black text
        thickness,
        cv2.LINE_AA,
    )


def draw_annotations(
    frame: np.ndarray,
    annotations: List[Tuple[Tuple[int, int, int, int], str, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw multiple emotion annotations on frame.

    Args:
        frame: Frame to draw on (modified in-place)
        annotations: List of (box, label, confidence) tuples
        color: BGR color for annotations
    """
    for box, label, confidence in annotations:
        draw_emotion_label(frame, box, label, confidence, color)


def create_clahe(clip_limit: float = 2.0, grid_size: int = 8) -> cv2.CLAHE:
    """Create CLAHE (Contrast Limited Adaptive Histogram Equalization) object.

    CLAHE can improve face detection and emotion recognition in varying lighting.

    Args:
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE object
    """
    return cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(max(1, grid_size), max(1, grid_size))
    )
