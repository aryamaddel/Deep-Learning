import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets

print("Downloading Fashion MNIST...")
train_data = datasets.FashionMNIST(root="fashion_mnist_data", train=True, download=True)
test_data  = datasets.FashionMNIST(root="fashion_mnist_data", train=False, download=True)

train_images = train_data.data.numpy()
train_labels = train_data.targets.numpy()
test_images  = test_data.data.numpy()
test_labels  = test_data.targets.numpy()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ── Basic stats ─────────────────────────────────────────────────────
print("\n===== DATASET OVERVIEW =====")
print(f"Training images : {train_images.shape}  ({train_images.dtype})")
print(f"Training labels : {train_labels.shape}")
print(f"Test images     : {test_images.shape}")
print(f"Test labels     : {test_labels.shape}")
print(f"Pixel range     : [{train_images.min()}, {train_images.max()}]")
print(f"Classes         : {len(class_names)}")

# ── Class distribution ─────────────────────────────────────────────
train_counts = np.bincount(train_labels)
test_counts = np.bincount(test_labels)

print("\n===== CLASS DISTRIBUTION =====")
print(f"{'Class':<15} {'Train':>6} {'Test':>6}")
for i, name in enumerate(class_names):
    print(f"{name:<15} {train_counts[i]:>6} {test_counts[i]:>6}")
print(f"{'TOTAL':<15} {train_counts.sum():>6} {test_counts.sum():>6}")

# ── Sample images ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.ravel()
for i in range(10):
    axes[i].imshow(train_images[i], cmap="gray")
    axes[i].set_title(class_names[train_labels[i]], fontsize=9)
    axes[i].axis("off")
plt.suptitle("Fashion MNIST — Sample Training Images", fontsize=13)
plt.tight_layout()
plt.savefig("sample_images.png", dpi=120)
plt.close()
print("\nSaved: sample_images.png")

# ── Pixel intensity distribution ────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.hist(train_images.ravel(), bins=50, color="steelblue", edgecolor="white", alpha=0.8)
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")
plt.title("Pixel Intensity Distribution (Training Set)")
plt.savefig("pixel_distribution.png", dpi=120)
plt.close()
print("Saved: pixel_distribution.png")

# ── Per-class mean images ───────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.ravel()
for i in range(10):
    class_pixels = train_images[train_labels == i]
    mean_img = class_pixels.astype(np.float32).mean(axis=0)
    axes[i].imshow(mean_img, cmap="gray")
    axes[i].set_title(f"{class_names[i]} (mean)", fontsize=8)
    axes[i].axis("off")
plt.suptitle("Per-Class Mean Images", fontsize=13)
plt.tight_layout()
plt.savefig("class_means.png", dpi=120)
plt.close()
print("Saved: class_means.png")

# ── Correlation between class pairs ─────────────────────────────────
print("\n===== CLASS MEAN CORRELATION MATRIX (TOP 5 PAIRS) =====")
class_means = np.array([train_images[train_labels == i].astype(np.float32).mean(axis=0).ravel() for i in range(10)])
corr = np.corrcoef(class_means)
pairs = []
for i in range(10):
    for j in range(i + 1, 10):
        pairs.append((corr[i, j], i, j))
pairs.sort(reverse=True)
for r, i, j in pairs[:5]:
    print(f"  {class_names[i]:<14} ↔ {class_names[j]:<14}  r = {r:.4f}")

# ── Per-class statistics ────────────────────────────────────────────
print("\n===== PER-CLASS STATISTICS =====")
print(f"{'Class':<15} {'Mean':>7} {'Std':>7} {'Min':>5} {'Max':>5}")
for i, name in enumerate(class_names):
    pixels = train_images[train_labels == i].astype(np.float32)
    print(f"{name:<15} {pixels.mean():>7.2f} {pixels.std():>7.2f} {pixels.min():>5.0f} {pixels.max():>5.0f}")

# ── Image variance per class ────────────────────────────────────────
variances = [train_images[train_labels == i].astype(np.float32).var() for i in range(10)]

plt.figure(figsize=(8, 5))
bars = plt.bar(range(10), variances, color=plt.cm.tab10(np.linspace(0, 1, 10)))
plt.xticks(range(10), class_names, rotation=45, ha="right", fontsize=8)
plt.ylabel("Pixel Variance")
plt.title("Intra-Class Pixel Variance")
plt.tight_layout()
plt.savefig("class_variance.png", dpi=120)
plt.close()
print("Saved: class_variance.png")

print("\nAnalysis complete.")
