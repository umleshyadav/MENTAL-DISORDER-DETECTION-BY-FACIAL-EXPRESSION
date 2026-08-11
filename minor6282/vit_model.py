"""
vit_model.py
============
Defines the ViT-based emotion recognition model used across all scripts
(training, inference, Streamlit app, and the real-time final.py).
"""

import torch
import torch.nn as nn
import timm


EMOTIONS    = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = 7


def build_vit(pretrained: bool = False) -> nn.Module:
    """
    Build ViT-Tiny (patch16, 224) with a custom classification head.

    Args:
        pretrained: If True, loads ImageNet pretrained weights (useful for
                    training from scratch). Set False for inference when you
                    will load best_vit.pth manually.

    Returns:
        model (nn.Module): ViT model ready for training or inference.
    """
    model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def load_vit(weights_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a trained ViT model from a saved state-dict file.

    Args:
        weights_path: Path to the .pth file (e.g. "best_vit.pth").
        device:       "cuda" or "cpu".

    Returns:
        model (nn.Module): Loaded model in eval mode.
    """
    model = build_vit(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


class LSTMModel(nn.Module):
    """
    LSTM model that predicts mental health state from a sequence of
    emotion probability vectors output by the ViT model.

    Input  shape: (batch, seq_len, 7)   — 7 emotion probs per frame
    Output shape: (batch, 3)            — logits for 3 mental states
    """

    STATES = ["Stable", "Depression-like", "Anxiety-like"]

    def __init__(self,
                 input_size:  int = 7,
                 hidden_size: int = 64,
                 num_layers:  int = 2,
                 num_classes: int = 3,
                 dropout:     float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])   # use last time-step
        return self.fc(out)


def load_lstm(weights_path: str, device: str = "cpu") -> "LSTMModel":
    """
    Load a trained LSTMModel from a saved state-dict file.

    Args:
        weights_path: Path to the .pth file (e.g. "lstm_model.pth").
        device:       "cuda" or "cpu".

    Returns:
        model (LSTMModel): Loaded model in eval mode.
    """
    model = LSTMModel()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model
