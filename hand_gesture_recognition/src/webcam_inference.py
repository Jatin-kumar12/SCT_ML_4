"""inference/webcam_inference.py — Real-time gesture recognition via webcam."""

import cv2
import torch
import numpy as np
from collections import deque
from typing import List, Optional
import time

from utils.hand_detector import HandDetector
import training.config


class GesturePredictor:
    """
    Wraps a trained model for single-frame or smoothed prediction.
    Supports all three model types (CNN, MLP, LSTM).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5,
        device: str = "cpu",
        model_type: str = "landmark_mlp",
    ):
        self.model = model.eval().to(device)
        self.device = device
        self.class_names = class_names
        self.threshold = confidence_threshold
        self.model_type = model_type
        self.image_size = 224

        # Temporal smoothing: keep a deque of recent predictions
        self._pred_buffer = deque(maxlen=smoothing_window)
        self._prob_buffer = deque(maxlen=smoothing_window)

        from torchvision import transforms
        self._img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict_landmark(self, feature_vec: np.ndarray):
        """Predict from a 63-dim landmark feature vector."""
        x = torch.from_numpy(feature_vec).float().unsqueeze(0).to(self.device)
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return probs

    @torch.no_grad()
    def predict_image(self, roi_bgr: np.ndarray):
        """Predict from a cropped hand ROI (BGR numpy array)."""
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        x = self._img_transform(rgb).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return probs

    def smooth_and_decide(self, probs: np.ndarray):
        """
        Push current probs into buffer, average, return
        (class_name, confidence) if above threshold else (None, conf).
        """
        self._prob_buffer.append(probs)
        avg_probs = np.mean(self._prob_buffer, axis=0)
        best_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[best_idx])

        if confidence >= self.threshold:
            return self.class_names[best_idx], confidence
        return None, confidence


class WebcamGestureRecognizer:
    """
    Full real-time pipeline:
        Camera → Hand detection → Feature extraction → Model → Overlay
    """

    def __init__(self, model, cfg: training.config.InferenceConfig, data_cfg: training.config.DataConfig):
        self.cfg = cfg
        self.predictor = GesturePredictor(
            model=model,
            class_names=data_cfg.class_names,
            confidence_threshold=cfg.confidence_threshold,
            smoothing_window=cfg.smoothing_window,
            model_type=cfg.model_type,
        )
        self.detector = HandDetector(max_hands=1)
        self.fps_counter = deque(maxlen=30)

    def run(self):
        cap = cv2.VideoCapture(self.cfg.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Press Q to quit | Press S to save screenshot")

        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            detections = self.detector.detect(frame)

            label, confidence = None, 0.0

            if detections:
                hand = detections[0]

                if self.cfg.model_type == "landmark_mlp":
                    feat = self.detector.get_feature_vector(hand)
                    probs = self.predictor.predict_landmark(feat)
                else:
                    x1, y1, x2, y2 = self.detector.get_bounding_box(
                        hand, frame.shape
                    )
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        probs = self.predictor.predict_image(roi)
                    else:
                        probs = np.zeros(len(self.predictor.class_names))

                label, confidence = self.predictor.smooth_and_decide(probs)

                if self.cfg.display_landmarks:
                    frame = self.detector.draw_landmarks(frame, detections)

                # Bounding box
                x1, y1, x2, y2 = self.detector.get_bounding_box(
                    hand, frame.shape
                )
                color = (0, 200, 100) if label else (100, 100, 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            self._draw_overlay(frame, label, confidence)

            if self.cfg.display_fps:
                self.fps_counter.append(1.0 / (time.time() - t0 + 1e-6))
                fps = np.mean(self.fps_counter)
                cv2.putText(
                    frame, f"FPS: {fps:.1f}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                )

            cv2.imshow("Hand Gesture Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"screenshot_{int(time.time())}.jpg", frame)
                print("Screenshot saved.")

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()

    @staticmethod
    def _draw_overlay(frame, label: Optional[str], confidence: float):
        h, w = frame.shape[:2]
        # Semi-transparent banner at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if label:
            text = f"{label.upper()}  {confidence:.0%}"
            color = (80, 220, 100)
        else:
            text = f"No gesture  {confidence:.0%}"
            color = (120, 120, 120)

        cv2.putText(
            frame, text, (16, 40),
            cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def run_webcam_demo(
    model_path: str = "checkpoints/best_model.pth",
    model_type: str = "landmark_mlp",
    num_classes: int = 10,
):
    from training.config import DataConfig, InferenceConfig

    data_cfg = training.config.DataConfig(num_classes=num_classes)
    inf_cfg = training.config.InferenceConfig(
        model_path=model_path,
        model_type=model_type,
    )

    if model_type == "landmark_mlp":
        from models.landmark_mlp import GestureLandmarkMLP
        model = GestureLandmarkMLP(num_classes=num_classes)
    elif model_type == "cnn":
        from models.cnn_model import GestureCNN
        model = GestureCNN(num_classes=num_classes)
    else:
        from models.landmark_mlp import GestureLSTM
        model = GestureLSTM(num_classes=num_classes)

    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    recognizer = WebcamGestureRecognizer(model, inf_cfg, data_cfg)
    recognizer.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="checkpoints/best_model.pth")
    parser.add_argument("--model-type", default="landmark_mlp",
                        choices=["landmark_mlp", "cnn", "lstm"])
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    run_webcam_demo(args.model_path, args.model_type, args.num_classes)
