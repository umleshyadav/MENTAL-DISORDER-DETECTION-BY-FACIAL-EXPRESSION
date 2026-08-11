"""
utils.py
========
Shared utility functions used across training, evaluation, and inference.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import List, Tuple


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
STATES   = ["Stable", "Depression-like", "Anxiety-like"]

# Emoji map for display
EMOTION_EMOJI = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😊",
    "Neutral":  "😐",
    "Sad":      "😢",
    "Surprise": "😲",
}

STATE_COLOR = {
    "Stable":           (0, 220, 100),
    "Depression-like":  (0, 80,  220),
    "Anxiety-like":     (220, 80, 0),
}

# ─────────────────────────────────────────────
# IMAGE TRANSFORMS
# ─────────────────────────────────────────────
def get_inference_transform(img_size: int = 224) -> transforms.Compose:
    """Return the standard inference transform (no augmentation)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def preprocess_frame(frame: np.ndarray,
                     transform: transforms.Compose,
                     device: str = "cpu") -> torch.Tensor:
    """
    Convert a BGR OpenCV frame to a model-ready tensor.

    Args:
        frame:     BGR numpy array from cv2.
        transform: Torchvision transform pipeline.
        device:    "cuda" or "cpu".

    Returns:
        tensor: shape (1, 3, H, W) on the given device.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    return tensor


# ─────────────────────────────────────────────
# FACE DETECTION
# ─────────────────────────────────────────────
def get_face_cascade() -> cv2.CascadeClassifier:
    """Return OpenCV Haar cascade for frontal face detection."""
    xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade  = cv2.CascadeClassifier(xml_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from: {xml_path}")
    return cascade


def detect_faces(gray_frame: np.ndarray,
                 cascade: cv2.CascadeClassifier,
                 scale_factor: float = 1.3,
                 min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a grayscale frame.

    Returns:
        List of (x, y, w, h) tuples for each detected face.
    """
    faces = cascade.detectMultiScale(gray_frame, scale_factor, min_neighbors)
    return list(faces) if len(faces) > 0 else []


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────
def predict_emotion(model: torch.nn.Module,
                    tensor: torch.Tensor,
                    device: str = "cpu") -> Tuple[str, np.ndarray]:
    """
    Run ViT forward pass and return (predicted_emotion, prob_vector).

    Args:
        model:  ViT model (in eval mode).
        tensor: (1, 3, H, W) tensor on device.
        device: "cuda" or "cpu".

    Returns:
        emotion (str), probs (np.ndarray of shape [7])
    """
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    emotion = EMOTIONS[int(np.argmax(probs))]
    return emotion, probs


def predict_mental_state(lstm_model: torch.nn.Module,
                         sequence: np.ndarray,
                         device: str = "cpu") -> str:
    """
    Run LSTM forward pass on a (seq_len, 7) emotion sequence.

    Args:
        lstm_model: Trained LSTMModel.
        sequence:   np.ndarray of shape (seq_len, 7).
        device:     "cuda" or "cpu".

    Returns:
        state (str): One of STATES.
    """
    seq_tensor = torch.tensor(
        sequence, dtype=torch.float32
    ).unsqueeze(0).to(device)                  # (1, seq_len, 7)
    with torch.no_grad():
        out   = lstm_model(seq_tensor)
        idx   = int(torch.argmax(out).item())
    return STATES[idx]


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def draw_emotion_on_frame(frame: np.ndarray,
                          x: int, y: int, w: int, h: int,
                          emotion: str,
                          confidence: float,
                          probs: np.ndarray) -> np.ndarray:
    """
    Draw bounding box, emotion label, confidence, and top-3 bar on a frame.

    Returns:
        Annotated BGR frame.
    """
    color = (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{emotion} ({confidence:.2f})",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # top-3 emotions below the box
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_text = "  ".join(
        [f"{EMOTIONS[i]}:{probs[i]:.2f}" for i in top3_idx]
    )
    cv2.putText(frame, top3_text,
                (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return frame


def draw_state_on_frame(frame: np.ndarray,
                        state: str,
                        smoothed_emotion: str) -> np.ndarray:
    """
    Draw mental state and smoothed emotion on the top of the frame.

    Returns:
        Annotated BGR frame.
    """
    color = STATE_COLOR.get(state, (255, 255, 255))
    cv2.putText(frame, f"Emotion : {smoothed_emotion}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"State   : {state}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame


def plot_emotion_sequence(sequence: np.ndarray,
                          title: str = "Emotion Over Time") -> plt.Figure:
    """
    Plot a (T, 7) emotion probability sequence.

    Args:
        sequence: np.ndarray of shape (T, 7).
        title:    Plot title string.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, emo in enumerate(EMOTIONS):
        ax.plot(sequence[:, i], label=emo)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_emotion_bar(avg_probs: np.ndarray,
                     title: str = "Average Emotion Distribution") -> plt.Figure:
    """
    Plot a bar chart of average emotion probabilities.

    Args:
        avg_probs: np.ndarray of shape (7,).

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(EMOTIONS, avg_probs,
                  color=["#e74c3c","#8e44ad","#3498db",
                         "#f1c40f","#95a5a6","#2c3e50","#1abc9c"])
    ax.set_ylabel("Average Probability")
    ax.set_title(title)
    for bar, val in zip(bars, avg_probs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# DATASET HELPERS
# ─────────────────────────────────────────────
def count_samples_per_class(root_dir: str) -> dict:
    """
    Count images per emotion class in a dataset folder.

    Args:
        root_dir: Path like "datasets/train".

    Returns:
        dict mapping emotion_name → count.
    """
    counts = {}
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            counts[cls] = n
    return counts


def get_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"
