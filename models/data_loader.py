import numpy as np
import json, os, time
from torchvision import datasets

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    train_data = datasets.FashionMNIST(root="fashion_mnist_data", train=True, download=False)
    test_data  = datasets.FashionMNIST(root="fashion_mnist_data", train=False, download=False)
    X_train = train_data.data.numpy()
    y_train = train_data.targets.numpy()
    X_test  = test_data.data.numpy()
    y_test  = test_data.targets.numpy()
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    return X_train, y_train, X_test, y_test, class_names


def flatten_normalise(X):
    return X.reshape(len(X), -1).astype(np.float32) / 255.0


def save_result(name, acc, prec, rec, f1, train_time, infer_time, cm):
    path = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_').lower()}.json")
    with open(path, "w") as f:
        json.dump({
            "name": name, "acc": round(acc, 4), "prec": round(prec, 4),
            "rec": round(rec, 4), "f1": round(f1, 4),
            "train_time": round(train_time, 2), "infer_time": round(infer_time, 2),
            "cm": cm.tolist(),
        }, f)
    return path


def print_results(name, acc, f1, train_time, infer_time):
    print(f"  {name:<30s}  acc={acc:.4f}  f1={f1:.4f}  "
          f"train={train_time:.1f}s  infer={infer_time:.1f}s")
