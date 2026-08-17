"""
Assignment 04: Image classification using CNN and ANN on Fashion-MNIST.

Run from the repository root:
    uv run python assignment4/cnn_classification.py

Quick test:
    uv run python assignment4/cnn_classification.py --quick
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
OUT = Path(__file__).resolve().parent / "outputs"
CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST CNN experiment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--quick", action="store_true", help="small 2-epoch test run")
    return parser.parse_args()


def load_data(quick=False):
    """Load images, normalize pixels, add a channel, and create validation data."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize [0, 255] pixels to [0, 1] and reshape (28, 28) -> (28, 28, 1).
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, axis=-1)
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, axis=-1)

    x_val, y_val = x_train[-10_000:], y_train[-10_000:]
    x_train, y_train = x_train[:-10_000], y_train[:-10_000]

    if quick:
        x_train, y_train = x_train[:8_000], y_train[:8_000]
        x_val, y_val = x_val[:2_000], y_val[:2_000]
        x_test, y_test = x_test[:2_000], y_test[:2_000]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_cnn():
    """CNN: convolution extracts features; pooling reduces spatial dimensions."""
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ], name="CNN")


def build_ann():
    """Traditional ANN baseline without convolution or pooling."""
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ], name="ANN")


def plot_curves(history, model_name):
    epochs = range(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history.history["accuracy"], label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set(title=f"{model_name} Accuracy", xlabel="Epoch", ylabel="Accuracy")

    axes[1].plot(epochs, history.history["loss"], label="Training")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set(title=f"{model_name} Loss", xlabel="Epoch", ylabel="Loss")

    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{model_name.lower()}_learning_curves.png", dpi=150)
    plt.close(fig)


def train_model(model, data, args):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n{model.name} architecture:")
    model.summary()
    print(f"Training {model.name}...", end=" ", flush=True)
    start = time.perf_counter()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    plot_curves(history, model.name)
    print(f"test accuracy: {test_accuracy:.4f} | loss: {test_loss:.4f} | {elapsed:.1f}s")

    return {
        "Model": model.name,
        "Best Validation Accuracy": max(history.history["val_accuracy"]),
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss,
        "Training Time (s)": elapsed,
        "Parameters": model.count_params(),
    }


def save_confusion_matrix(model, x_test, y_test):
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    matrix = confusion_matrix(y_test, predictions)

    fig, axis = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    axis.set_title("CNN Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT / "cnn_confusion_matrix.png", dpi=150)
    plt.close(fig)
    return predictions


def save_sample_predictions(x_test, y_test, predictions):
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(x_test), size=12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))

    for axis, index in zip(axes.flat, indices):
        actual = CLASS_NAMES[y_test[index]]
        predicted = CLASS_NAMES[predictions[index]]
        color = "green" if actual == predicted else "red"
        axis.imshow(x_test[index].squeeze(), cmap="gray")
        axis.set_title(f"Pred: {predicted}\nTrue: {actual}", color=color, fontsize=9)
        axis.axis("off")

    fig.suptitle("CNN Predictions on Sample Test Images")
    fig.tight_layout()
    fig.savefig(OUT / "sample_predictions.png", dpi=150)
    plt.close(fig)


def save_model_comparison(results):
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "model_comparison.csv", index=False)

    x = np.arange(len(frame))
    width = 0.35
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(x - width / 2, frame["Best Validation Accuracy"], width, label="Validation")
    axis.bar(x + width / 2, frame["Test Accuracy"], width, label="Test")
    axis.set_xticks(x, frame["Model"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("CNN vs ANN Performance")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cnn_vs_ann.png", dpi=150)
    plt.close(fig)
    return frame


def main():
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)

    OUT.mkdir(exist_ok=True)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    print("=" * 65)
    print("ASSIGNMENT 4: CNN IMAGE CLASSIFICATION")
    print("=" * 65)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | "
          f"Learning rate: {args.learning_rate}")

    data = load_data(args.quick)
    print(f"Train: {data[0][0].shape} | Validation: {data[1][0].shape} | "
          f"Test: {data[2][0].shape}")
    print(f"Normalized pixel range: {data[0][0].min():.1f} to {data[0][0].max():.1f}")

    keras.utils.set_random_seed(SEED)
    cnn = build_cnn()
    cnn_result = train_model(cnn, data, args)

    keras.utils.set_random_seed(SEED)
    ann = build_ann()
    ann_result = train_model(ann, data, args)

    x_test, y_test = data[2]
    predictions = save_confusion_matrix(cnn, x_test, y_test)
    save_sample_predictions(x_test, y_test, predictions)
    comparison = save_model_comparison([cnn_result, ann_result])

    print("\nCNN vs ANN RESULTS")
    print(comparison.to_string(
        index=False,
        formatters={
            "Best Validation Accuracy": "{:.4f}".format,
            "Test Accuracy": "{:.4f}".format,
            "Test Loss": "{:.4f}".format,
            "Training Time (s)": "{:.1f}".format,
        },
    ))
    best = comparison.loc[comparison["Test Accuracy"].idxmax(), "Model"]
    print(f"\nBest model by test accuracy: {best}")
    print(f"All plots and results saved in: {OUT}")


if __name__ == "__main__":
    main()
