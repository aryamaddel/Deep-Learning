"""
Assignment 02: Implementation of Perceptron and Artificial Neural Network (ANN)
models using TensorFlow/Keras.

Aim:
  - Study the architecture and working of Perceptron and ANN.
  - Implement a Single Layer Perceptron using TensorFlow/Keras.
  - Implement a Multi-Layer ANN for a classification problem.
  - Train, evaluate, and compare the performance of the implemented models.
  - Analyze the effect of activation functions and hidden layers.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)


def banner(title):
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


# ---------------------------------------------------------
# 1. Versions + dataset loading and preprocessing
# ---------------------------------------------------------
banner("1. LIBRARY VERSIONS")
import tensorflow as tf
print(f"TensorFlow : {tf.__version__}")
print(f"Keras      : {keras.__version__}")

banner("2. LOAD & PREPROCESS IRIS DATASET")
iris = load_iris()
X, y = iris.data, iris.target
print(f"Dataset shape : features={X.shape}, labels={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train_ohe = keras.utils.to_categorical(y_train, num_classes=3)
y_test_ohe = keras.utils.to_categorical(y_test, num_classes=3)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


# ---------------------------------------------------------
# 2. Single Layer Perceptron (no hidden layer => linear model)
# ---------------------------------------------------------
banner("3. SINGLE LAYER PERCEPTRON (SLP)")

slp = models.Sequential(
    [layers.Input(shape=(4,)), layers.Dense(3, activation="softmax")]
)
slp.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
slp.summary()

slp_hist = slp.fit(
    X_train, y_train_ohe,
    epochs=100, batch_size=16, verbose=0,
    validation_data=(X_test, y_test_ohe),
)

slp_train_acc = slp_hist.history["accuracy"][-1]
slp_val_acc = slp_hist.history["val_accuracy"][-1]
print(f"SLP  -> Train acc: {slp_train_acc:.4f} | Val acc: {slp_val_acc:.4f}")


# ---------------------------------------------------------
# 3. Multi-Layer ANN (with hidden layers => non-linear)
# ---------------------------------------------------------
banner("4. MULTI-LAYER ANN (MLP) WITH HIDDEN LAYERS")
ann = models.Sequential(
    [
        layers.Input(shape=(4,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ]
)
ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
ann.summary()

ann_hist = ann.fit(
    X_train, y_train_ohe,
    epochs=100, batch_size=16, verbose=0,
    validation_data=(X_test, y_test_ohe),
)

ann_train_acc = ann_hist.history["accuracy"][-1]
ann_val_acc = ann_hist.history["val_accuracy"][-1]
print(f"ANN  -> Train acc: {ann_train_acc:.4f} | Val acc: {ann_val_acc:.4f}")


# ---------------------------------------------------------
# 4. Compare performance (accuracy + loss curves)
# ---------------------------------------------------------
banner("5. COMPARE SLP vs ANN (ACCURACY & LOSS)")


def plot_history(h, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.history["accuracy"], label="Train")
    ax[0].plot(h.history["val_accuracy"], label="Validation")
    ax[0].set_title(f"{title} - Accuracy")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(h.history["loss"], label="Train")
    ax[1].plot(h.history["val_loss"], label="Validation")
    ax[1].set_title(f"{title} - Loss")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, path), dpi=120)
    print(f"Saved plot -> {path}")
    plt.close(fig)


plot_history(slp_hist, "Single Layer Perceptron", "slp_curves.png")
plot_history(ann_hist, "Multi-Layer ANN", "ann_curves.png")

fig, ax = plt.subplots(figsize=(7, 5))
methods = ["SLP", "ANN"]
train_acc = [slp_train_acc, ann_train_acc]
val_acc = [slp_val_acc, ann_val_acc]
x = np.arange(len(methods))
w = 0.35
ax.bar(x - w / 2, train_acc, w, label="Train")
ax.bar(x + w / 2, val_acc, w, label="Validation")
ax.set_xticks(x, methods)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("SLP vs ANN - Final Accuracy")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "slp_vs_ann.png"), dpi=120)
print("Saved plot -> slp_vs_ann.png")
plt.close(fig)


# ---------------------------------------------------------
# 5. Analyzing effect of hidden layers on performance
# ---------------------------------------------------------
banner("6. EFFECT OF NUMBER OF HIDDEN LAYERS")
configs = {
    "0 hidden (1 Dense)": [3],
    "1 hidden (8)": [8, 3],
    "2 hidden (16,16)": [16, 16, 3],
    "3 hidden (32,16,8)": [32, 16, 8, 3],
}

results = {}
for name, units in configs.items():
    m = models.Sequential([layers.Input(shape=(4,))])
    for u in units[:-1]:
        m.add(layers.Dense(u, activation="relu"))
    m.add(layers.Dense(units[-1], activation="softmax"))
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(X_train, y_train_ohe, epochs=100, batch_size=16, verbose=0,
              validation_data=(X_test, y_test_ohe))
    results[name] = (h.history["val_accuracy"][-1], h.history["val_loss"][-1])
    print(f"{name:<22} Val acc: {results[name][0]:.4f} | Val loss: {results[name][1]:.4f}")

names = list(results.keys())
accs = [results[n][0] for n in names]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(names, accs, color="steelblue")
ax.set_ylim(0, 1)
ax.set_ylabel("Validation Accuracy")
ax.set_title("Effect of Number of Hidden Layers")
ax.tick_params(axis="x", rotation=15)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hidden_layers_effect.png"), dpi=120)
print("Saved plot -> hidden_layers_effect.png")
plt.close(fig)


# ---------------------------------------------------------
# 6. Analyzing effect of activation functions
# ---------------------------------------------------------
banner("7. EFFECT OF ACTIVATION FUNCTION (hidden layer)")
activations = ["relu", "sigmoid", "tanh"]
act_results = {}
for act in activations:
    m = models.Sequential(
        [
            layers.Input(shape=(4,)),
            layers.Dense(16, activation=act),
            layers.Dense(16, activation=act),
            layers.Dense(3, activation="softmax"),
        ]
    )
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(X_train, y_train_ohe, epochs=100, batch_size=16, verbose=0,
              validation_data=(X_test, y_test_ohe))
    act_results[act] = (h.history["val_accuracy"][-1], h.history["val_loss"][-1])
    print(f"{act:<10} Val acc: {act_results[act][0]:.4f} | Val loss: {act_results[act][1]:.4f}")


# ---------------------------------------------------------
# 7. Final classification report (best model = ANN relu)
# ---------------------------------------------------------
banner("8. FINAL EVALUATION ON ANN")
y_pred = np.argmax(ann.predict(X_test, verbose=0), axis=1)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n" + "=" * 72)
print("  ALL STEPS DONE.")
print("=" * 72)
