"""training/config.py — Central configuration for all hyperparameters."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    dataset_path: str = "data/gestures"
    image_size: int = 224
    num_classes: int = 10
    class_names: List[str] = field(default_factory=lambda: [
        "thumbs_up", "thumbs_down", "peace", "fist",
        "open_palm", "ok_sign", "pointing", "rock",
        "call_me", "stop"
    ])
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    augmentation: bool = True
    num_landmarks: int = 21          # MediaPipe hand landmarks
    landmark_features: int = 63      # 21 landmarks × 3 (x, y, z)


@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v3"   # mobilenet_v3 | resnet18 | efficientnet_b0
    pretrained: bool = True
    dropout: float = 0.3
    hidden_dim: int = 256
    # LSTM-specific
    lstm_hidden: int = 128
    lstm_layers: int = 2
    sequence_length: int = 30        # frames per gesture sequence


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"        # cosine | step | plateau
    patience: int = 10               # early stopping
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "auto"             # auto | cpu | cuda | mps


@dataclass
class InferenceConfig:
    model_path: str = "checkpoints/best_model.pth"
    model_type: str = "landmark_mlp" # cnn | lstm | landmark_mlp
    confidence_threshold: float = 0.7
    smoothing_window: int = 5        # temporal smoothing frames
    camera_id: int = 0
    display_fps: bool = True
    display_landmarks: bool = True
