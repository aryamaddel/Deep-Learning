"""Compare RNN, LSTM, and GRU on sequence prediction and IMDB sentiment.

Run:
    uv run assignment6/rnn_lstm_gru.py

Quick smoke test:
    uv run assignment6/rnn_lstm_gru.py --quick
"""

import argparse
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_NAMES = ["RNN", "LSTM", "GRU"]


def parse_args():
    parser = argparse.ArgumentParser(description="RNN, LSTM, and GRU experiments")
    parser.add_argument("--sequence-epochs", type=int, default=6)
    parser.add_argument("--sentiment-epochs", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="run very small smoke tests")
    return parser.parse_args()


def recurrent_layer(model_name, units):
    if model_name == "RNN":
        return layers.SimpleRNN(units)
    if model_name == "LSTM":
        return layers.LSTM(units)
    return layers.GRU(units)


def make_history_rows(history, task, model_name):
    rows = []
    for epoch in range(len(history.history["loss"])):
        row = {"task": task, "model": model_name, "epoch": epoch + 1}
        for metric, values in history.history.items():
            row[metric] = values[epoch]
        rows.append(row)
    return rows


def run_sequence_experiment(args, history_rows):
    print("\n" + "=" * 76)
    print("PART A: NEXT-STEP SEQUENCE PREDICTION")
    print("=" * 76)

    # Create a repeatable waveform and turn it into input windows of 30 values.
    points = 900 if args.quick else 2_500
    time_axis = np.linspace(0, 40 * np.pi, points, dtype="float32")
    signal = np.sin(time_axis) + 0.3 * np.sin(3 * time_axis)
    window = 30
    x, y = [], []
    for index in range(len(signal) - window):
        x.append(signal[index:index + window])
        y.append(signal[index + window])
    x = np.array(x, dtype="float32").reshape(-1, window, 1)
    y = np.array(y, dtype="float32")

    # Chronological split prevents future windows from leaking into training.
    train_end = int(0.70 * len(x))
    val_end = int(0.85 * len(x))
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]
    print(f"Train: {len(x_train)} | Validation: {len(x_val)} | Test: {len(x_test)}")

    results = []
    predictions = {}
    epochs = 2 if args.quick else args.sequence_epochs

    for model_name in MODEL_NAMES:
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)

        model = keras.Sequential([
            layers.Input(shape=(window, 1)),
            recurrent_layer(model_name, 32),
            layers.Dense(1),
        ], name=f"{model_name}_sequence")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )

        print(f"\nTraining {model_name} for sequence prediction...")
        start = time.perf_counter()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )],
            verbose=0,
        )
        elapsed = time.perf_counter() - start
        predicted = model.predict(x_test, verbose=0).reshape(-1)
        predictions[model_name] = predicted

        mae = mean_absolute_error(y_test, predicted)
        rmse = np.sqrt(mean_squared_error(y_test, predicted))
        r2 = r2_score(y_test, predicted)
        results.append({
            "model": model_name,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "training_time_seconds": elapsed,
            "parameters": model.count_params(),
            "epochs_trained": len(history.history["loss"]),
        })
        history_rows.extend(make_history_rows(history, "sequence", model_name))
        print(f"{model_name}: MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f} | {elapsed:.1f}s")

    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "sequence_results.csv", index=False)

    count = min(200, len(y_test))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(y_test[:count], color="black", linewidth=2, label="Actual")
    for model_name in MODEL_NAMES:
        axis.plot(predictions[model_name][:count], label=model_name, alpha=0.8)
    axis.set(title="Next-step Sequence Predictions", xlabel="Test time step", ylabel="Value")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sequence_predictions.png", dpi=150)
    plt.close(fig)
    return frame


def run_sentiment_experiment(args, history_rows):
    print("\n" + "=" * 76)
    print("PART B: IMDB SENTIMENT ANALYSIS")
    print("=" * 76)

    vocabulary_size = 10_000
    maximum_length = 100
    (x_train_all, y_train_all), (x_test_all, y_test_all) = (
        keras.datasets.imdb.load_data(num_words=vocabulary_size)
    )
    y_train_all = np.asarray(y_train_all)
    y_test_all = np.asarray(y_test_all)

    # Select fixed, balanced subsets to keep all three CPU runs below 10 minutes.
    selected_size = 1_300 if args.quick else 7_500
    selected_indices, _ = train_test_split(
        np.arange(len(x_train_all)), train_size=selected_size,
        random_state=SEED, stratify=y_train_all,
    )
    validation_size = 300 if args.quick else 1_500
    train_indices, val_indices = train_test_split(
        selected_indices, test_size=validation_size,
        random_state=SEED, stratify=y_train_all[selected_indices],
    )
    test_size = 500 if args.quick else 2_000
    test_indices, _ = train_test_split(
        np.arange(len(x_test_all)), train_size=test_size,
        random_state=SEED, stratify=y_test_all,
    )

    x_train = keras.utils.pad_sequences(
        [x_train_all[index] for index in train_indices], maxlen=maximum_length
    )
    x_val = keras.utils.pad_sequences(
        [x_train_all[index] for index in val_indices], maxlen=maximum_length
    )
    x_test = keras.utils.pad_sequences(
        [x_test_all[index] for index in test_indices], maxlen=maximum_length
    )
    y_train = y_train_all[train_indices]
    y_val = y_train_all[val_indices]
    y_test = y_test_all[test_indices]
    print(f"Train: {len(x_train)} | Validation: {len(x_val)} | Test: {len(x_test)}")
    print(f"Padded review shape: {x_train.shape} | Vocabulary: {vocabulary_size:,}")

    results = []
    histories = {}
    matrices = {}
    epochs = 1 if args.quick else args.sentiment_epochs

    for model_name in MODEL_NAMES:
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)

        model = keras.Sequential([
            layers.Input(shape=(maximum_length,), dtype="int32"),
            layers.Embedding(vocabulary_size, 32, mask_zero=True),
            recurrent_layer(model_name, 32),
            layers.Dropout(0.30),
            layers.Dense(1, activation="sigmoid"),
        ], name=f"{model_name}_sentiment")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        print(f"\nTraining {model_name} for sentiment analysis...")
        start = time.perf_counter()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=128,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=1, restore_best_weights=True
            )],
            verbose=0,
        )
        elapsed = time.perf_counter() - start
        probabilities = model.predict(x_test, batch_size=128, verbose=0).reshape(-1)
        predicted = (probabilities >= 0.5).astype("int32")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predicted, average="binary", zero_division=0
        )
        accuracy = accuracy_score(y_test, predicted)
        auc = roc_auc_score(y_test, probabilities)
        matrices[model_name] = confusion_matrix(y_test, predicted)
        histories[model_name] = history.history
        results.append({
            "model": model_name,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_auc": auc,
            "best_validation_accuracy": max(history.history["val_accuracy"]),
            "training_time_seconds": elapsed,
            "parameters": model.count_params(),
            "epochs_trained": len(history.history["loss"]),
        })
        history_rows.extend(make_history_rows(history, "sentiment", model_name))
        print(
            f"{model_name}: accuracy={accuracy:.4f} | precision={precision:.4f} | "
            f"recall={recall:.4f} | F1={f1:.4f} | AUC={auc:.4f} | {elapsed:.1f}s"
        )

    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "sentiment_results.csv", index=False)

    for metric, filename, title in [
        ("accuracy", "sentiment_accuracy.png", "Sentiment Classification Accuracy"),
        ("loss", "sentiment_loss.png", "Sentiment Classification Loss"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for axis, model_name in zip(axes, MODEL_NAMES):
            values = histories[model_name]
            epoch_axis = range(1, len(values[metric]) + 1)
            axis.plot(epoch_axis, values[metric], marker="o", label="Training")
            axis.plot(epoch_axis, values[f"val_{metric}"], marker="o", label="Validation")
            axis.set(title=model_name, xlabel="Epoch", ylabel=metric.title())
            axis.set_xticks(list(epoch_axis))
            axis.grid(alpha=0.3)
            axis.legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, model_name in zip(axes, MODEL_NAMES):
        ConfusionMatrixDisplay(
            matrices[model_name], display_labels=["Negative", "Positive"]
        ).plot(ax=axis, cmap="Blues", colorbar=False)
        axis.set_title(model_name)
    fig.suptitle("IMDB Test Confusion Matrices")
    fig.tight_layout()
    fig.savefig(OUT / "sentiment_confusion_matrices.png", dpi=150)
    plt.close(fig)
    return frame


def main():
    args = parse_args()
    OUT.mkdir(exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("=" * 76)
    print("ASSIGNMENT 6: RNN, LSTM, AND GRU")
    print("=" * 76)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

    history_rows = []
    sequence_results = run_sequence_experiment(args, history_rows)
    sentiment_results = run_sentiment_experiment(args, history_rows)
    pd.DataFrame(history_rows).to_csv(OUT / "training_history.csv", index=False)

    print("\nSEQUENCE PREDICTION RESULTS")
    print(sequence_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSENTIMENT ANALYSIS RESULTS")
    print(sentiment_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_sequence = sequence_results.loc[sequence_results["test_rmse"].idxmin(), "model"]
    best_sentiment = sentiment_results.loc[sentiment_results["test_f1"].idxmax(), "model"]
    print(f"\nLowest sequence RMSE: {best_sequence}")
    print(f"Highest sentiment F1: {best_sentiment}")
    print(f"Outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
