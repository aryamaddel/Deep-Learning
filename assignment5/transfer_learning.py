"""Transfer learning with an ImageNet-pretrained ResNet18 on CIFAR-10.

Run from the repository root:
    uv run python assignment5/transfer_learning.py

Quick smoke test:
    uv run python assignment5/transfer_learning.py --quick
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet18 transfer learning on CIFAR-10")
    parser.add_argument("--feature-epochs", type=int, default=1)
    parser.add_argument("--fine-tune-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--val-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--quick", action="store_true", help="small one-epoch smoke test")
    return parser.parse_args()


def run_epoch(model, loader, loss_function, device, optimizer=None):
    """Run one explicit training or evaluation pass."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = loss_function(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, device, phase, epochs, learning_rate,
                history, best_state, best_val_accuracy):
    """Train one phase and keep the checkpoint selected by validation accuracy."""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    start = time.perf_counter()

    print(f"\n{phase}: {epochs} epoch(s), lr={learning_rate}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters):,}")

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_function, device)
        history.append({
            "phase": phase,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f}, "
            f"accuracy {train_accuracy:.4f} | val loss {val_loss:.4f}, "
            f"accuracy {val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, OUT / "best_resnet18_cifar10.pt")

    elapsed = time.perf_counter() - start
    phase_rows = [row for row in history if row["phase"] == phase]
    summary = {
        "phase": phase,
        "best_validation_accuracy": max(row["val_accuracy"] for row in phase_rows),
        "training_time_seconds": elapsed,
        "trainable_parameters": sum(p.numel() for p in trainable_parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    return best_state, best_val_accuracy, summary


def predict(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            actual.extend(labels.numpy())
            predicted.extend(outputs.argmax(dim=1).cpu().numpy())
    return np.array(actual), np.array(predicted)


def main():
    args = parse_args()
    if args.quick:
        args.feature_epochs = 1
        args.fine_tune_epochs = 1
        args.train_samples = 1_000
        args.val_samples = 300
        args.test_samples = 300

    OUT.mkdir(exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ImageNet preprocessing is required because ResNet18 learned from images
    # normalized this way. Augmentation is used only for the training set.
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size + 8),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize(args.image_size + 8),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    training_data = datasets.CIFAR10(DATA, train=True, download=True, transform=train_transform)
    validation_data = datasets.CIFAR10(DATA, train=True, download=False, transform=evaluation_transform)
    test_data = datasets.CIFAR10(DATA, train=False, download=True, transform=evaluation_transform)

    all_indices = np.arange(len(training_data))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=5_000,
        random_state=SEED,
        stratify=training_data.targets,
    )

    # Balanced subsets keep the default CPU experiment comfortably short.
    targets = np.array(training_data.targets)
    if args.train_samples < len(train_indices):
        train_indices, _ = train_test_split(
            train_indices, train_size=args.train_samples, random_state=SEED,
            stratify=targets[train_indices],
        )
    if args.val_samples < len(val_indices):
        val_indices, _ = train_test_split(
            val_indices, train_size=args.val_samples, random_state=SEED,
            stratify=targets[val_indices],
        )

    test_indices = np.arange(len(test_data))
    if args.test_samples < len(test_indices):
        test_indices, _ = train_test_split(
            test_indices, train_size=args.test_samples, random_state=SEED,
            stratify=np.array(test_data.targets),
        )
    test_data = Subset(test_data, test_indices)

    train_loader = DataLoader(
        Subset(training_data, train_indices), batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(validation_data, val_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("=" * 72)
    print("TRANSFER LEARNING: IMAGENET RESNET18 ON CIFAR-10")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Input size: {args.image_size}x{args.image_size} | Batch size: {args.batch_size}")
    print(f"Train: {len(train_loader.dataset):,} | Validation: {len(val_loader.dataset):,} "
          f"| Test: {len(test_loader.dataset):,}")

    # Load ImageNet weights, freeze the entire backbone, and replace the
    # original 1,000-class layer with a new 10-class layer.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)

    history = []
    phase_summaries = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0

    best_state, best_val_accuracy, summary = train_phase(
        model, train_loader, val_loader, device,
        "feature_extraction", args.feature_epochs, 1e-3,
        history, best_state, best_val_accuracy,
    )
    phase_summaries.append(summary)

    # Begin fine-tuning from the best feature-extraction checkpoint rather than
    # from a possibly weaker final epoch.
    model.load_state_dict(best_state)

    # Unfreeze only the final residual block. Earlier general-purpose features
    # remain fixed, while the most task-specific features can adapt.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    best_state, best_val_accuracy, summary = train_phase(
        model, train_loader, val_loader, device,
        "fine_tuning", args.fine_tune_epochs, 1e-4,
        history, best_state, best_val_accuracy,
    )
    phase_summaries.append(summary)

    model.load_state_dict(best_state)
    y_true, y_pred = predict(model, test_loader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    test_metrics = {
        "best_validation_accuracy": best_val_accuracy,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
    pd.DataFrame(phase_summaries).to_csv(OUT / "phase_comparison.csv", index=False)
    with open(OUT / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    history_frame = pd.DataFrame(history)
    x = np.arange(1, len(history_frame) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, history_frame["train_loss"], marker="o", label="Training")
    axes[0].plot(x, history_frame["val_loss"], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch across both phases", ylabel="Cross-entropy")
    axes[1].plot(x, history_frame["train_accuracy"], marker="o", label="Training")
    axes[1].plot(x, history_frame["val_accuracy"], marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch across both phases", ylabel="Accuracy")
    for axis in axes:
        axis.axvline(args.feature_epochs + 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=150)
    plt.close(fig)

    matrix = confusion_matrix(y_true, y_pred)
    fig, axis = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    axis.set_title("ResNet18 CIFAR-10 Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Display unnormalized test images for a clear prediction sample.
    raw_test_data = datasets.CIFAR10(DATA, train=False, download=False)
    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(len(y_true), size=min(12, len(y_true)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for axis, index in zip(axes.flat, sample_indices):
        image, _ = raw_test_data[int(test_indices[index])]
        color = "green" if y_true[index] == y_pred[index] else "red"
        axis.imshow(image)
        axis.set_title(
            f"Pred: {CLASS_NAMES[y_pred[index]]}\nTrue: {CLASS_NAMES[y_true[index]]}",
            color=color, fontsize=9,
        )
        axis.axis("off")
    for axis in axes.flat[len(sample_indices):]:
        axis.axis("off")
    fig.suptitle("Sample Test Predictions")
    fig.tight_layout()
    fig.savefig(OUT / "sample_predictions.png", dpi=150)
    plt.close(fig)

    print("\nFINAL TEST METRICS")
    for name, value in test_metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")
    print(f"\nOutputs saved to: {OUT}")


if __name__ == "__main__":
    main()
