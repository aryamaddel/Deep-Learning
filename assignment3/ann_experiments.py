"""
Assignment 03: Experimental study of activation functions, optimizers, and
regularization techniques for an ANN using TensorFlow/Keras.

The program uses Fashion-MNIST and performs controlled experiments:
  1. Baseline ANN
  2. Activation comparison: ReLU, Sigmoid, Tanh, Leaky ReLU
  3. Optimizer comparison: SGD, Adam, RMSprop, Adagrad
  4. Regularization comparison: None, Dropout, L1, L2, Batch Normalization,
     and Early Stopping
  5. A final model combining the best validation choice from each study

Run:
    python assignment3/ann_experiments.py
Quick smoke test:
    python assignment3/ann_experiments.py --quick
"""

import argparse
import json
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
from tensorflow import keras
from tensorflow.keras import layers, regularizers


SEED = 42
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
OUT = Path(__file__).resolve().parent / "outputs"
PLOTS = OUT / "plots"
HISTORIES = OUT / "histories"


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Assignment 3 ANN experiments")
    parser.add_argument("--epochs", type=int, default=15, help="maximum epochs per run")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--quick", action="store_true",
        help="use 10,000 train/2,000 validation/2,000 test samples and 3 epochs",
    )
    return parser.parse_args()


def load_data(quick=False):
    """Load, normalize, flatten, and split Fashion-MNIST."""
    (x_train_all, y_train_all), (x_test, y_test) = (
        keras.datasets.fashion_mnist.load_data()
    )
    x_train_all = x_train_all.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Fixed validation set: 50,000 training + 10,000 validation examples.
    x_train, y_train = x_train_all[:-10_000], y_train_all[:-10_000]
    x_val, y_val = x_train_all[-10_000:], y_train_all[-10_000:]

    if quick:
        x_train, y_train = x_train[:10_000], y_train[:10_000]
        x_val, y_val = x_val[:2_000], y_val[:2_000]
        x_test, y_test = x_test[:2_000], y_test[:2_000]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def activation_layer(name):
    """Return an activation layer; LeakyReLU is a layer, not a string here."""
    if name == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.1)
    return layers.Activation(name)


def build_ann(activation="relu", regularization="none"):
    """Build the same two-hidden-layer ANN for every controlled experiment."""
    kernel_regularizer = None
    if regularization == "l1":
        kernel_regularizer = regularizers.l1(1e-5)
    elif regularization == "l2":
        kernel_regularizer = regularizers.l2(1e-4)

    model = keras.Sequential(name=f"ann_{activation}_{regularization}")
    model.add(layers.Input(shape=(28, 28)))
    model.add(layers.Flatten())

    for units in (256, 128):
        model.add(layers.Dense(units, kernel_regularizer=kernel_regularizer))
        if regularization == "batch_norm":
            model.add(layers.BatchNormalization())
        model.add(activation_layer(activation))
        if regularization == "dropout":
            model.add(layers.Dropout(0.3))

    model.add(layers.Dense(10, activation="softmax"))
    return model


def make_optimizer(name):
    """Use explicit learning rates so the optimizer setup is visible."""
    choices = {
        "sgd": lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "adam": lambda: keras.optimizers.Adam(learning_rate=0.001),
        "rmsprop": lambda: keras.optimizers.RMSprop(learning_rate=0.001),
        "adagrad": lambda: keras.optimizers.Adagrad(learning_rate=0.01),
    }
    return choices[name]()


def safe_name(text):
    return text.lower().replace(" ", "_").replace("/", "_")


def plot_history(history, title, output_path):
    """Plot training/validation accuracy and loss for one experiment."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["accuracy"], label="Training")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set(title=f"{title}: Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(epochs, history["loss"], label="Training")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set(title=f"{title}: Loss", xlabel="Epoch", ylabel="Loss")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_group_comparison(results, study):
    """Plot best validation/test accuracy and training time for a study."""
    group = [row for row in results if row["study"] == study]
    names = [row["configuration"] for row in group]
    x = np.arange(len(names))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - width / 2, [r["best_val_accuracy"] for r in group], width,
                label="Best validation")
    axes[0].bar(x + width / 2, [r["test_accuracy"] for r in group], width,
                label="Test")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].bar(x, [r["training_time_seconds"] for r in group], color="darkorange")
    axes[1].set_ylabel("Training time (seconds)")
    for ax in axes:
        ax.set_xticks(x, names, rotation=20, ha="right")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"{study.title()} Study Comparison")
    fig.tight_layout()
    fig.savefig(PLOTS / f"comparison_{safe_name(study)}.png", dpi=150)
    plt.close(fig)


def run_experiment(study, configuration, data, epochs, batch_size,
                   activation="relu", optimizer="adam", regularization="none"):
    """Train one model, evaluate it, record its history, and save its plot."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)

    model = build_ann(activation, regularization)
    model.compile(
        optimizer=make_optimizer(optimizer),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = []
    if regularization == "early_stopping":
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        )

    print(f"\nRunning {configuration}...", end=" ", flush=True)
    if study == "baseline":
        print("\n")
        model.summary()
        print("Training baseline...", end=" ", flush=True)
    start = time.perf_counter()
    fit = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        # Suppress one line of output per epoch. A compact summary is printed
        # after each experiment instead.
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    history = {key: [float(v) for v in values] for key, values in fit.history.items()}
    run_name = safe_name(f"{study}_{configuration}")
    with open(HISTORIES / f"{run_name}.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    plot_history(history, f"{study.title()} - {configuration}", PLOTS / f"{run_name}.png")

    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    row = {
        "study": study,
        "configuration": configuration,
        "activation": activation,
        "optimizer": optimizer,
        "regularization": regularization,
        "epochs_ran": len(history["loss"]),
        "best_epoch": best_epoch,
        "final_train_accuracy": history["accuracy"][-1],
        "final_train_loss": history["loss"][-1],
        "best_val_accuracy": max(history["val_accuracy"]),
        "best_val_loss": min(history["val_loss"]),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "training_time_seconds": elapsed,
    }
    print(
        f"Done | epochs: {row['epochs_ran']:>2} | "
        f"val acc: {row['best_val_accuracy']:.4f} | "
        f"test acc: {test_accuracy:.4f} | {elapsed:.1f}s"
    )
    return row


def best_choice(results, study, field):
    candidates = [row for row in results if row["study"] == study]
    return max(candidates, key=lambda row: row["best_val_accuracy"])[field]


def main():
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 3)

    OUT.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    HISTORIES.mkdir(exist_ok=True)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    banner("1. ENVIRONMENT AND DATASET")
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Available GPU(s)   : {tf.config.list_physical_devices('GPU')}")
    data = load_data(args.quick)
    print(f"Training shape     : {data[0][0].shape}")
    print(f"Validation shape   : {data[1][0].shape}")
    print(f"Test shape         : {data[2][0].shape}")
    print(f"Pixel range        : {data[0][0].min():.1f} to {data[0][0].max():.1f}")

    results = []
    results.append(run_experiment(
        "baseline", "ReLU + Adam + None", data, args.epochs, args.batch_size
    ))

    banner("2. ACTIVATION FUNCTION EXPERIMENTS")
    for activation in ("relu", "sigmoid", "tanh", "leaky_relu"):
        results.append(run_experiment(
            "activation", activation, data, args.epochs, args.batch_size,
            activation=activation,
        ))

    banner("3. OPTIMIZER EXPERIMENTS")
    for optimizer in ("sgd", "adam", "rmsprop", "adagrad"):
        results.append(run_experiment(
            "optimizer", optimizer, data, args.epochs, args.batch_size,
            optimizer=optimizer,
        ))

    banner("4. REGULARIZATION EXPERIMENTS")
    for technique in ("none", "dropout", "l1", "l2", "batch_norm", "early_stopping"):
        results.append(run_experiment(
            "regularization", technique, data, args.epochs, args.batch_size,
            regularization=technique,
        ))

    # Choose only from validation performance; the test set remains unbiased.
    best_activation = best_choice(results, "activation", "activation")
    best_optimizer = best_choice(results, "optimizer", "optimizer")
    best_regularization = best_choice(results, "regularization", "regularization")

    banner("5. FINAL COMBINED MODEL")
    final_label = f"{best_activation} + {best_optimizer} + {best_regularization}"
    results.append(run_experiment(
        "final", final_label, data, args.epochs, args.batch_size,
        activation=best_activation,
        optimizer=best_optimizer,
        regularization=best_regularization,
    ))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "experiment_results.csv", index=False)
    for study in ("activation", "optimizer", "regularization"):
        plot_group_comparison(results, study)

    summary = {
        "selection_rule": "highest best validation accuracy in each controlled study",
        "best_activation": best_activation,
        "best_optimizer": best_optimizer,
        "best_regularization": best_regularization,
        "final_test_accuracy": results[-1]["test_accuracy"],
        "final_test_loss": results[-1]["test_loss"],
        "final_training_time_seconds": results[-1]["training_time_seconds"],
    }
    with open(OUT / "best_model_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    banner("6. COMPLETE RESULTS")
    # Print narrow, readable tables; the CSV retains every recorded metric.
    for study in ("baseline", "activation", "optimizer", "regularization", "final"):
        group = results_df[results_df["study"] == study].copy()
        group = group[[
            "configuration", "best_val_accuracy", "test_accuracy",
            "training_time_seconds",
        ]]
        group.columns = ["Configuration", "Val Acc", "Test Acc", "Time (s)"]
        print(f"\n{study.upper()}")
        print(group.to_string(
            index=False,
            formatters={
                "Val Acc": "{:.4f}".format,
                "Test Acc": "{:.4f}".format,
                "Time (s)": "{:.1f}".format,
            },
        ))
    print("\nBEST-PERFORMING COMBINATION")
    print(f"Activation     : {best_activation}")
    print(f"Optimizer      : {best_optimizer}")
    print(f"Regularization : {best_regularization}")
    print(f"Final test accuracy: {results[-1]['test_accuracy']:.4f}")
    print(f"\nAll tables, histories, and plots saved in: {OUT}")


if __name__ == "__main__":
    main()
