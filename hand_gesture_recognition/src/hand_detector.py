"""utils/hand_detector.py — MediaPipe-based hand detection and landmark extraction."""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class HandLandmarks:
    """Stores normalized and pixel-space landmarks for one detected hand."""
    normalized: np.ndarray          # shape (21, 3) — x, y, z in [0,1]
    pixel_coords: np.ndarray        # shape (21, 2) — pixel x, y
    handedness: str                 # "Left" or "Right"
    confidence: float


class HandDetector:
    """
    Wraps MediaPipe Hands for robust landmark detection.

    MediaPipe returns 21 landmarks per hand (wrist + 4 per finger).
    Each landmark has (x, y) normalized to image size, and z for depth.
    """

    CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """
        Run detection on a BGR frame (OpenCV default).
        Returns a list of HandLandmarks (one per detected hand).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        detections: List[HandLandmarks] = []
        if not results.multi_hand_landmarks:
            return detections

        h, w = frame.shape[:2]
        for hand_lms, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            normalized = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            pixel_coords = (normalized[:, :2] * [w, h]).astype(int)
            confidence = handedness.classification[0].score
            label = handedness.classification[0].label

            detections.append(HandLandmarks(
                normalized=normalized,
                pixel_coords=pixel_coords,
                handedness=label,
                confidence=confidence,
            ))

        return detections

    def get_feature_vector(self, landmarks: HandLandmarks) -> np.ndarray:
        """
        Flatten 21×3 landmarks into a 63-dim feature vector.
        Normalizes relative to wrist position so the gesture is
        translation-invariant.
        """
        coords = landmarks.normalized.copy()
        # Translate: wrist (index 0) becomes origin
        coords -= coords[0]
        # Scale: normalize by the max spread so size-invariant
        scale = np.max(np.abs(coords)) + 1e-6
        coords /= scale
        return coords.flatten()

    def draw_landmarks(
        self,
        frame: np.ndarray,
        detections: List[HandLandmarks],
        draw_connections: bool = True,
    ) -> np.ndarray:
        """Overlay landmarks and skeleton on a copy of the frame."""
        annotated = frame.copy()
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = True

        # Re-run to get MediaPipe native draw objects (cleaner than manual)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated,
                    hand_lms,
                    self.CONNECTIONS if draw_connections else None,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return annotated

    def get_bounding_box(
        self,
        landmarks: HandLandmarks,
        frame_shape: Tuple[int, int],
        padding: int = 20,
    ) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) bounding box with padding."""
        h, w = frame_shape[:2]
        pts = landmarks.pixel_coords
        x1 = max(0, pts[:, 0].min() - padding)
        y1 = max(0, pts[:, 1].min() - padding)
        x2 = min(w, pts[:, 0].max() + padding)
        y2 = min(h, pts[:, 1].max() + padding)
        return x1, y1, x2, y2

    def close(self):
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
