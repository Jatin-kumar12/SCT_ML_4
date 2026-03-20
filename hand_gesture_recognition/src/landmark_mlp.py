"""models/landmark_mlp.py — Lightweight MLP on MediaPipe keypoints."""

import torch
import torch.nn as nn


class GestureLandmarkMLP(nn.Module):
    """
    Fast MLP classifier operating on 63-dim landmark feature vectors.

    Why this works well:
    - MediaPipe landmarks are already semantically rich (wrist, fingertips, etc.)
    - After translation + scale normalization the features are pose-invariant
    - Runs at >1000 fps on CPU — ideal for real-time HCI

    Input:  (B, 63)  — 21 landmarks × 3 (x, y, z), normalized
    Output: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 63,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


"""models/lstm_model.py — Bidirectional LSTM for dynamic/temporal gestures."""


class GestureLSTM(nn.Module):
    """
    Bidirectional LSTM that classifies a sequence of landmark frames.
    Suited for dynamic gestures that unfold over time: wave, swipe,
    clockwise circle, etc.

    Input:  (B, T, 63)  — T frames of landmark feature vectors
    Output: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 63,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Project raw features to a richer embedding before the LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 63)
        proj = self.input_proj(x)                   # (B, T, 128)
        out, (h_n, _) = self.lstm(proj)

        if self.bidirectional:
            # Concat forward (h_n[-2]) and backward (h_n[-1]) final states
            final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final = h_n[-1]                         # (B, hidden_dim)

        return self.classifier(final)               # (B, num_classes)


"""models/attention_lstm.py — LSTM with self-attention over time steps."""


class AttentionLayer(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, H)
        scores = self.attn(lstm_out).squeeze(-1)    # (B, T)
        weights = torch.softmax(scores, dim=1)      # (B, T)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B, H)
        return context, weights


class AttentionGestureLSTM(nn.Module):
    """
    LSTM + self-attention: lets the model focus on the most
    discriminative frames in a gesture sequence.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 63,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            128, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        proj = self.input_proj(x)
        lstm_out, _ = self.lstm(proj)               # (B, T, H)
        context, attn_weights = self.attention(lstm_out)
        return self.classifier(context), attn_weights
