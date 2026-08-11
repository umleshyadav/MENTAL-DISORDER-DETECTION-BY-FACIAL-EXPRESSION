"""
train_vit.py
============
Trains a Vision Transformer (ViT-Tiny/16-224) on the FER2013-style dataset
for 7-class facial expression recognition.

Dataset structure expected:
    datasets/
        train/
            angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
        test/
            angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/

Run on Google Colab with GPU runtime.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR   = "datasets"          # path relative to this script
TRAIN_DIR     = os.path.join(DATASET_DIR, "train")
TEST_DIR      = os.path.join(DATASET_DIR, "test")

EMOTIONS      = ["angry","disgust","fear","happy","neutral","sad","surprise"]
NUM_CLASSES   = 7
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15          # 15 % of training set used for validation
SEED          = 42
SAVE_PATH     = "best_vit.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # FER images are grayscale
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5,  0.5,  0.5]),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5,  0.5,  0.5]),
])

# ─────────────────────────────────────────────
# DATASETS & LOADERS
# ─────────────────────────────────────────────
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_dataset       = datasets.ImageFolder(TEST_DIR,  transform=val_transforms)

# class order check
print(f"[INFO] Classes detected: {full_train_dataset.classes}")
assert full_train_dataset.classes == sorted(EMOTIONS), \
    "Class order mismatch! Check folder names."

val_size   = int(len(full_train_dataset) * VAL_SPLIT)
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)
# val subset should use val_transforms (no augmentation)
val_dataset.dataset = copy.deepcopy(full_train_dataset)
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"[INFO] Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")

# ─────────────────────────────────────────────
# MODEL — ViT-Tiny pretrained on ImageNet-21k
# ─────────────────────────────────────────────
def build_vit():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    return model

model = build_vit().to(device)
print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────────
# LOSS, OPTIMIZER, SCHEDULER
# ─────────────────────────────────────────────
# Compute class weights to handle FER2013 imbalance
class_counts = np.zeros(NUM_CLASSES)
for _, label in full_train_dataset:
    class_counts[label] += 1
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"[INFO] Class weights: {class_weights.round(3)}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ──────────────────────────────
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss    += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total           += images.size(0)

    train_loss = running_loss / total
    train_acc  = running_correct / total

    # ── Validate ───────────────────────────
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_sum  += loss.item() * images.size(0)
            val_correct   += (outputs.argmax(1) == labels).sum().item()
            val_total     += images.size(0)

    val_loss = val_loss_sum / val_total
    val_acc  = val_correct  / val_total

    scheduler.step()

    # ── Bookkeeping ────────────────────────
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, SAVE_PATH)

    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc*100:.2f}%"
          + (" ← best" if val_acc == best_val_acc else ""))

print(f"\n[INFO] Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"[INFO] Best model saved to: {SAVE_PATH}")

# ─────────────────────────────────────────────
# FINAL TEST EVALUATION
# ─────────────────────────────────────────────
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()
test_correct, test_total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_correct += (outputs.argmax(1) == labels).sum().item()
        test_total   += images.size(0)

test_acc = test_correct / test_total
print(f"[INFO] Test Accuracy: {test_acc*100:.2f}%")

# ─────────────────────────────────────────────
# PLOT — Accuracy & Loss curves
# ─────────────────────────────────────────────
epochs_range = range(1, NUM_EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, history["train_acc"], label="Train Acc")
axes[0].plot(epochs_range, history["val_acc"],   label="Val Acc")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(epochs_range, history["train_loss"], label="Train Loss")
axes[1].plot(epochs_range, history["val_loss"],   label="Val Loss")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("vit_training_curves.png", dpi=150)
plt.show()
print("[INFO] Training curves saved to vit_training_curves.png")
