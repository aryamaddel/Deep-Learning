#!/usr/bin/env python3
"""Train deep learning models on Fashion MNIST + comparative analysis."""

import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torchvision import datasets

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
EPOCHS = 40

def load_data():
    train_data = datasets.FashionMNIST(root="fashion_mnist_data", train=True, download=False)
    test_data = datasets.FashionMNIST(root="fashion_mnist_data", train=False, download=False)
    X_train = train_data.data.numpy()
    y_train = train_data.targets.numpy()
    X_test = test_data.data.numpy()
    y_test = test_data.targets.numpy()
    class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
                   "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    return X_train, y_train, X_test, y_test, class_names

def save_result(name, acc, prec, rec, f1, train_time, infer_time, cm):
    path = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_').lower()}.json")
    with open(path, "w") as f:
        json.dump({"name": name, "acc": round(acc, 4), "prec": round(prec, 4),
                   "rec": round(rec, 4), "f1": round(f1, 4),
                   "train_time": round(train_time, 2), "infer_time": round(infer_time, 2),
                   "cm": cm.tolist()}, f)
    return path

def print_result(name, acc, f1, train_time, infer_time):
    tqdm.write(f"  {name:<30s}  acc={acc:.4f}  f1={f1:.4f}  train={train_time:.1f}s  infer={infer_time:.1f}s")

def train_torch(model, loader, criterion, optimizer, device, name):
    t0 = time.time()
    epoch_bar = tqdm(total=EPOCHS, desc=f"  {name}", unit="ep", leave=False)
    for epoch in range(EPOCHS):
        batch_bar = tqdm(total=len(loader), desc=f"    ep {epoch+1}/{EPOCHS}", unit="batch", leave=False)
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
            batch_bar.update(1)
        batch_bar.close()
        epoch_bar.update(1)
    epoch_bar.close()
    return time.time() - t0

def evaluate(model, loader, y_test, device, name, t_train):
    model.eval()
    t0 = time.time()
    all_preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            all_preds.append(model(Xb.to(device)).argmax(dim=1).cpu().numpy())
    t_infer = time.time() - t0
    preds = np.concatenate(all_preds)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted")
    cm = confusion_matrix(y_test, preds)
    p = save_result(name, acc, prec, rec, f1, t_train, t_infer, cm)
    print_result(name, acc, f1, t_train, t_infer)
    tqdm.write(f"  Saved → {p}")

# ═══════════════════════════════════════════════════════════
tqdm.write("Loading Fashion MNIST...")
X_train, y_train, X_test, y_test, class_names = load_data()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
tqdm.write(f"Device: {device}")

X_train_t = torch.tensor(X_train.reshape(-1, 784).astype(np.float32) / 255.0)
X_test_t = torch.tensor(X_test.reshape(-1, 784).astype(np.float32) / 255.0)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

X_train_cnn = torch.tensor(X_train / 255.0, dtype=torch.float32).unsqueeze(1)
X_test_cnn = torch.tensor(X_test / 255.0, dtype=torch.float32).unsqueeze(1)

B = 256
loader_flat = DataLoader(TensorDataset(X_train_t, y_train_t), B, shuffle=True)
loader_flat_t = DataLoader(TensorDataset(X_test_t, y_test_t), B)
loader_cnn = DataLoader(TensorDataset(X_train_cnn, y_train_t), B, shuffle=True)
loader_cnn_t = DataLoader(TensorDataset(X_test_cnn, y_test_t), B)

summary = []

# ── 1. Deep CNN ───────────────────────────────────────────
tqdm.write("─" * 60)
m = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10),
).to(device)
t = train_torch(m, loader_cnn, nn.CrossEntropyLoss(), optim.Adam(m.parameters(), lr=1e-3), device, "Deep CNN")
evaluate(m, loader_cnn_t, y_test, device, "1. Deep CNN", t)
summary.append(("1. Deep CNN", t, "SUCCESS"))

# ── 2. Simple CNN ─────────────────────────────────────────
tqdm.write("─" * 60)
m = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10),
).to(device)
t = train_torch(m, loader_cnn, nn.CrossEntropyLoss(), optim.Adam(m.parameters(), lr=1e-3), device, "Simple CNN")
evaluate(m, loader_cnn_t, y_test, device, "2. Simple CNN", t)
summary.append(("2. Simple CNN", t, "SUCCESS"))

# ── 3. Softmax Regression ─────────────────────────────────
tqdm.write("─" * 60)
m = nn.Sequential(nn.Linear(784, 10)).to(device)
t = train_torch(m, loader_flat, nn.CrossEntropyLoss(), optim.Adam(m.parameters(), lr=1e-3), device, "Softmax Reg")
evaluate(m, loader_flat_t, y_test, device, "3. Softmax Regression", t)
summary.append(("3. Softmax Regression", t, "SUCCESS"))

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
tqdm.write("\n" + "=" * 60)
tqdm.write("  SUMMARY")
tqdm.write("=" * 60)
for name, t, status in summary:
    tqdm.write(f"    {name:<30s} {t:>6.1f}s  {status}")

# ═══════════════════════════════════════════════════════════
# COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════
results = []
for fname in sorted(os.listdir(RESULTS_DIR)):
    if fname.endswith(".json"):
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            results.append(json.load(f))
results.sort(key=lambda r: -r["acc"])

n = len(results)
tqdm.write("\n" + "=" * 75)
tqdm.write(f"COMPARATIVE ANALYSIS — {n} Models on Fashion MNIST")
tqdm.write("=" * 75)
tqdm.write(f"{'Rank':<5s} {'Model':<28s} {'Acc':>7s} {'F1':>7s} {'Train(s)':>9s} {'Infer(s)':>9s}")
tqdm.write("-" * 65)
for i, r in enumerate(results, 1):
    tqdm.write(f"{i:<5d} {r['name']:<28s} {r['acc']:.4f}  {r['f1']:.4f}  "
          f"{r['train_time']:>8.1f}  {r['infer_time']:>8.1f}")

fig, ax = plt.subplots(figsize=(12, 5))
names = [r["name"] for r in results]
accs = [r["acc"] for r in results]
colors = ["#2ecc71", "#2ecc71", "#3498db"]
bars = ax.barh(range(n), accs, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(range(n))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Test Accuracy")
ax.set_title("Fashion MNIST — Model Comparison", fontsize=14)
ax.set_xlim(0, 1.0)
for bar, val in zip(bars, accs):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("comparison_accuracy.png", dpi=150)
tqdm.write("\nSaved: comparison_accuracy.png")

fig, ax = plt.subplots(figsize=(10, 6))
for r in results:
    ax.scatter(r["train_time"], r["acc"], c="#2ecc71", marker="s", s=100, zorder=5)
    label = r["name"].split(". ")[1] if ". " in r["name"] else r["name"]
    ax.annotate(label, (r["train_time"], r["acc"]), fontsize=8,
                textcoords="offset points", xytext=(5, 3))
ax.set_xlabel("Training Time (s)")
ax.set_ylabel("Test Accuracy")
ax.set_title("Accuracy vs Training Time", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_accuracy_vs_time.png", dpi=150)
tqdm.write("Saved: comparison_accuracy_vs_time.png")

best = results[0]
cm = np.array(best["cm"])
fig, ax = plt.subplots(figsize=(9, 8))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — {best['name']} (acc={best['acc']:.4f})", fontsize=13)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("comparison_best_cm.png", dpi=150)
tqdm.write(f"Saved: comparison_best_cm.png (best: {best['name']})")

tqdm.write("\n" + "----- PER-CLASS ACCURACY -----")
header = f"{'Class':<15s}" + "".join(f"{r['name'][:14]:>14s}" for r in results)
tqdm.write(header)
tqdm.write("-" * len(header))
for c in range(10):
    line = f"{class_names[c]:<15s}"
    for r in results:
        cm_arr = np.array(r["cm"])
        line += f"{cm_arr[c, c] / cm_arr[c, :].sum():>14.3f}"
    tqdm.write(line)

tqdm.write("\nAll done.")
