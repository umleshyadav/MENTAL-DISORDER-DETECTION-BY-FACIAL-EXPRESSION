"""
train_lstm.py
=============
Trains an LSTM model to predict mental health states from sequences of
emotion probability vectors produced by the trained ViT model.

Mental states (3 classes):
    0 → Stable
    1 → Depression-like
    2 → Anxiety-like

The LSTM is trained on SYNTHETIC sequences generated from the real dataset
emotion distributions, because no real temporal mental-health labels exist.
The synthetic generation logic mirrors the pattern described in the project
report: prolonged sadness/fear → depression, high anger/fear volatility → anxiety.

Run on Google Colab AFTER train_vit.py has been completed.
best_vit.pth must exist in the same directory.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import timm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VIT_WEIGHTS   = "best_vit.pth"
DATASET_DIR   = "datasets"
TRAIN_DIR     = os.path.join(DATASET_DIR, "train")

EMOTIONS      = ["angry","disgust","fear","happy","neutral","sad","surprise"]
STATES        = ["Stable", "Depression-like", "Anxiety-like"]
NUM_EMOTIONS  = 7
NUM_STATES    = 3
SEQ_LEN       = 20            # frames per sequence window
N_SYNTH_SEQ   = 6000          # total synthetic sequences to generate
BATCH_SIZE    = 64
NUM_EPOCHS    = 50
LR            = 1e-3
SAVE_PATH     = "lstm_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ─────────────────────────────────────────────
# STEP 1 – Extract real emotion distributions
#           from the training images using ViT
# ─────────────────────────────────────────────
print("[INFO] Loading ViT to extract emotion distributions …")

vit = timm.create_model("vit_tiny_patch16_224", pretrained=False)
vit.head = nn.Sequential(
    nn.Linear(vit.head.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_EMOTIONS),
)
vit.load_state_dict(torch.load(VIT_WEIGHTS, map_location=device))
vit.to(device).eval()

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_transforms)
loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

# Collect softmax prob vectors grouped by emotion label
print("[INFO] Running ViT inference on training images …")
probs_by_class = {i: [] for i in range(NUM_EMOTIONS)}

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        out = vit(images)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        for prob, lbl in zip(probs, labels.numpy()):
            probs_by_class[lbl].append(prob)

for k in probs_by_class:
    probs_by_class[k] = np.array(probs_by_class[k])
    print(f"  {EMOTIONS[k]}: {len(probs_by_class[k])} samples")

# ─────────────────────────────────────────────
# STEP 2 – Synthesize labeled sequences
# ─────────────────────────────────────────────
# Label rules (mirroring the report description):
#   Stable         → mostly happy / neutral frames
#   Depression-like → prolonged sad / fear / disgust
#   Anxiety-like   → high fear / angry with variability

STABLE_CLASSES     = [3, 4]          # happy=3, neutral=4
DEPRESSION_CLASSES = [5, 2, 1]       # sad=5, fear=2, disgust=1
ANXIETY_CLASSES    = [2, 0, 6]       # fear=2, angry=0, surprise=6

def sample_sequence(dominant_classes, seq_len=SEQ_LEN, noise=0.25):
    """Sample SEQ_LEN emotion prob vectors, mostly from dominant_classes."""
    seq = []
    for _ in range(seq_len):
        if np.random.rand() < (1 - noise):
            cls = np.random.choice(dominant_classes)
        else:
            cls = np.random.randint(0, NUM_EMOTIONS)
        pool = probs_by_class[cls]
        if len(pool) == 0:
            vec = np.zeros(NUM_EMOTIONS); vec[cls] = 1.0
        else:
            vec = pool[np.random.randint(len(pool))]
        seq.append(vec)
    return np.array(seq, dtype=np.float32)

np.random.seed(42)
sequences, labels = [], []

n_per_class = N_SYNTH_SEQ // NUM_STATES

for _ in range(n_per_class):
    sequences.append(sample_sequence(STABLE_CLASSES,     noise=0.10))
    labels.append(0)

for _ in range(n_per_class):
    sequences.append(sample_sequence(DEPRESSION_CLASSES, noise=0.20))
    labels.append(1)

for _ in range(n_per_class):
    sequences.append(sample_sequence(ANXIETY_CLASSES,    noise=0.30))
    labels.append(2)

X = torch.tensor(np.array(sequences), dtype=torch.float32)   # (N, SEQ_LEN, 7)
y = torch.tensor(labels, dtype=torch.long)                    # (N,)

# Shuffle & split 80/20
perm   = torch.randperm(len(X))
X, y   = X[perm], y[perm]
split  = int(0.8 * len(X))
X_tr, y_tr = X[:split], y[:split]
X_va, y_va = X[split:], y[split:]

train_ds = TensorDataset(X_tr, y_tr)
val_ds   = TensorDataset(X_va, y_va)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Sequences: {len(X_tr)} train | {len(X_va)} val")

# ─────────────────────────────────────────────
# STEP 3 – LSTM Model
# ─────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=NUM_EMOTIONS, hidden_size=64, num_layers=2,
                 num_classes=NUM_STATES, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last time-step
        return self.fc(out)

lstm_model = LSTMModel().to(device)
print(f"[INFO] LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(lstm_model.parameters(), lr=LR)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# ─────────────────────────────────────────────
# STEP 4 – Training Loop
# ─────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    lstm_model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = lstm_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * xb.size(0)
        t_correct += (out.argmax(1) == yb).sum().item()
        t_total   += xb.size(0)

    lstm_model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = lstm_model(xb)
            loss = criterion(out, yb)
            v_loss    += loss.item() * xb.size(0)
            v_correct += (out.argmax(1) == yb).sum().item()
            v_total   += xb.size(0)

    scheduler.step()

    tr_l = t_loss / t_total; tr_a = t_correct / t_total
    vl_l = v_loss / v_total; vl_a = v_correct / v_total
    history["train_loss"].append(tr_l); history["train_acc"].append(tr_a)
    history["val_loss"].append(vl_l);   history["val_acc"].append(vl_a)

    if vl_a > best_val_acc:
        best_val_acc = vl_a
        torch.save(lstm_model.state_dict(), SAVE_PATH)

    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
          f"Train Loss: {tr_l:.4f} Acc: {tr_a*100:.1f}% | "
          f"Val Loss: {vl_l:.4f} Acc: {vl_a*100:.1f}%"
          + (" ← best" if vl_a == best_val_acc else ""))

print(f"\n[INFO] Best Val Accuracy: {best_val_acc*100:.2f}%")
print(f"[INFO] LSTM model saved to: {SAVE_PATH}")

# ─────────────────────────────────────────────
# STEP 5 – Plot
# ─────────────────────────────────────────────
epochs_range = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs_range, history["train_acc"], label="Train Acc")
axes[0].plot(epochs_range, history["val_acc"],   label="Val Acc")
axes[0].set_title("LSTM Accuracy"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(epochs_range, history["train_loss"], label="Train Loss")
axes[1].plot(epochs_range, history["val_loss"],   label="Val Loss")
axes[1].set_title("LSTM Loss"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig("lstm_training_curves.png", dpi=150)
plt.show()
print("[INFO] Curves saved to lstm_training_curves.png")
