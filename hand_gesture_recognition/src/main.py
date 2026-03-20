"""main.py — Entry point: train, evaluate, or run real-time inference."""

import argparse
import torch
from training.config import DataConfig, ModelConfig, TrainConfig, InferenceConfig


def build_model(model_type: str, num_classes: int, model_cfg: ModelConfig):
    if model_type == "cnn":
        from models.cnn_model import GestureCNN
        return GestureCNN(
            num_classes=num_classes,
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
        )
    elif model_type == "lstm":
        from models.landmark_mlp import GestureLSTM
        return GestureLSTM(
            num_classes=num_classes,
            hidden_dim=model_cfg.lstm_hidden,
            num_layers=model_cfg.lstm_layers,
            dropout=model_cfg.dropout,
        )
    elif model_type == "attention_lstm":
        from models.landmark_mlp import AttentionGestureLSTM
        return AttentionGestureLSTM(num_classes=num_classes, dropout=model_cfg.dropout)
    else:  # default: landmark_mlp
        from models.landmark_mlp import GestureLandmarkMLP
        return GestureLandmarkMLP(
            num_classes=num_classes,
            dropout=model_cfg.dropout,
        )


def cmd_train(args):
    data_cfg = DataConfig(dataset_path=args.data_path)
    model_cfg = ModelConfig(backbone=args.backbone)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    from utils.metrics import count_parameters
    from training.trainer import GestureTrainer

    model = build_model(args.model_type, data_cfg.num_classes, model_cfg)
    count_parameters(model)

    if args.model_type == "cnn":
        from data.dataset import build_image_loaders
        train_loader, val_loader, test_loader = build_image_loaders(
            data_cfg.dataset_path,
            batch_size=train_cfg.batch_size,
        )
    else:
        # Landmark data — expects pre-extracted JSON files
        from data.dataset import GestureLandmarkDataset
        from torch.utils.data import DataLoader, random_split
        full_ds = GestureLandmarkDataset(args.data_path)
        n = len(full_ds)
        n_train, n_val = int(n * 0.8), int(n * 0.1)
        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n - n_train - n_val]
        )
        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size)
        test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size)

    trainer = GestureTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=data_cfg.num_classes,
        learning_rate=train_cfg.learning_rate,
        checkpoint_dir=train_cfg.checkpoint_dir,
    )

    history = trainer.train(train_cfg.epochs)

    from utils.metrics import plot_training_history, evaluate_model, plot_confusion_matrix
    plot_training_history(history)
    results = evaluate_model(model, test_loader, data_cfg.class_names)
    plot_confusion_matrix(results["confusion_matrix"], data_cfg.class_names)


def cmd_infer(args):
    from inference.webcam_inference import run_webcam_demo
    run_webcam_demo(
        model_path=args.model_path,
        model_type=args.model_type,
        num_classes=args.num_classes,
    )


def cmd_collect(args):
    """Collect landmark data for a new gesture class via webcam."""
    import cv2
    import json
    import os
    from utils.hand_detector import HandDetector

    os.makedirs(f"data/landmarks/{args.gesture}", exist_ok=True)
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    samples = []

    print(f"Collecting data for gesture: '{args.gesture}'")
    print("Press SPACE to capture a sample | Q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        detections = detector.detect(frame)

        if detections:
            frame = detector.draw_landmarks(frame, detections)
            feat = detector.get_feature_vector(detections[0])
            cv2.putText(
                frame, f"Detected! Samples: {len(samples)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2,
            )

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and detections:
            samples.append({
                "features": feat.tolist(),
                "label": args.label_id,
                "gesture": args.gesture,
            })
            print(f"  Captured sample #{len(samples)}")
        elif key == ord("q"):
            break

    out_path = f"data/landmarks/{args.gesture}/samples.json"
    with open(out_path, "w") as f:
        json.dump(samples, f)
    print(f"Saved {len(samples)} samples to {out_path}")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="Train a gesture model")
    t.add_argument("--data-path", required=True)
    t.add_argument("--model-type", default="landmark_mlp",
                   choices=["cnn", "lstm", "attention_lstm", "landmark_mlp"])
    t.add_argument("--backbone", default="mobilenet_v3")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)

    # infer
    i = sub.add_parser("infer", help="Run real-time webcam inference")
    i.add_argument("--model-path", default="checkpoints/best_model.pth")
    i.add_argument("--model-type", default="landmark_mlp",
                   choices=["cnn", "lstm", "landmark_mlp"])
    i.add_argument("--num-classes", type=int, default=10)

    # collect
    c = sub.add_parser("collect", help="Collect landmark training data")
    c.add_argument("--gesture", required=True, help="Gesture name")
    c.add_argument("--label-id", type=int, required=True, help="Integer class label")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "collect":
        cmd_collect(args)
