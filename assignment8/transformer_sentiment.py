"""Assignment 8: a small Transformer encoder for sentiment classification.

Run from the repository root:
    uv run assignment8/transformer_sentiment.py

The experiment uses a deterministic, built-in sentence corpus and PyTorch, so
it does not download a dataset or pretrained model. It demonstrates token and
position embeddings, query/key/value self-attention, classification metrics,
and an attention heatmap.
"""

import argparse
import random
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PAD, UNK = "<pad>", "<unk>"

# A balanced, local corpus keeps this introductory experiment reproducible and
# runnable offline. The test split is held out from training and model selection.
EXAMPLES = [
    ("I loved the brilliant story and warm performances", 1),
    ("The film was delightful, funny, and beautifully made", 1),
    ("A wonderful movie with a genuinely moving ending", 1),
    ("The cast gave excellent performances throughout", 1),
    ("This charming story kept me smiling", 1),
    ("The direction was confident and the images were beautiful", 1),
    ("I enjoyed every minute of this clever film", 1),
    ("A heartfelt and entertaining movie", 1),
    ("The characters felt real and easy to care about", 1),
    ("A funny script and a terrific cast", 1),
    ("The soundtrack and acting were both excellent", 1),
    ("It was a fresh, thoughtful, and satisfying story", 1),
    ("An engaging film that I would happily watch again", 1),
    ("The lead actor was fantastic", 1),
    ("A joyful comedy with smart dialogue", 1),
    ("The movie is moving without being overly sentimental", 1),
    ("Strong writing makes this an enjoyable experience", 1),
    ("A polished production with a great ending", 1),
    ("I found the film imaginative and absorbing", 1),
    ("The performances bring this lovely story to life", 1),
    ("I disliked the dull story and wooden performances", 0),
    ("The film was tedious, confusing, and poorly made", 0),
    ("A terrible movie with a predictable ending", 0),
    ("The cast gave weak performances throughout", 0),
    ("This lifeless story left me bored", 0),
    ("The direction was clumsy and the images looked cheap", 0),
    ("I regretted every minute of this messy film", 0),
    ("A joyless and forgettable movie", 0),
    ("The characters felt fake and hard to care about", 0),
    ("A bad script and a disappointing cast", 0),
    ("The soundtrack and acting were both awful", 0),
    ("It was a stale, careless, and frustrating story", 0),
    ("A tedious film that I would never watch again", 0),
    ("The lead actor was dreadful", 0),
    ("A tiresome comedy with clumsy dialogue", 0),
    ("The movie is dull and overly sentimental", 0),
    ("Weak writing makes this an unpleasant experience", 0),
    ("A sloppy production with a disappointing ending", 0),
    ("I found the film unimaginative and tedious", 0),
    ("The performances fail to bring this story to life", 0),
]


def tokenize(text):
    return re.findall(r"[a-z]+|[^\w\s]", text.lower())


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReviewDataset(Dataset):
    def __init__(self, rows, vocab, max_length):
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text, label = self.rows[index]
        tokens = tokenize(text)[: self.max_length]
        ids = [self.vocab.get(token, self.vocab[UNK]) for token in tokens]
        mask = [1] * len(ids)
        ids += [self.vocab[PAD]] * (self.max_length - len(ids))
        mask += [0] * (self.max_length - len(mask))
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(label, dtype=torch.long),
        )


class TransformerBlock(nn.Module):
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask, return_attention=False):
        attended, weights = self.attention(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return (x, weights) if return_attention else (x, None)


class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, max_length, dimension=48, heads=4, dropout=0.15):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, dimension)
        self.block = TransformerBlock(dimension, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension, 2)
        )

    def forward(self, token_ids, attention_mask, return_attention=False):
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x, weights = self.block(x, ~attention_mask, return_attention)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        logits = self.classifier(pooled)
        return (logits, weights) if return_attention else logits


def evaluate(model, loader, device, loss_fn):
    model.eval()
    losses, labels, predictions = [], [], []
    with torch.no_grad():
        for token_ids, mask, target in loader:
            token_ids, mask, target = token_ids.to(device), mask.to(device), target.to(device)
            logits = model(token_ids, mask)
            losses.append(loss_fn(logits, target).item())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.cpu().tolist())
    return float(np.mean(losses)), np.asarray(labels), np.asarray(predictions)


def plot_attention(model, vocab, text, device):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab[UNK]) for token in tokens]
    mask = torch.ones((1, len(ids)), dtype=torch.bool, device=device)
    with torch.no_grad():
        _, weights = model(
            torch.tensor([ids], dtype=torch.long, device=device), mask, return_attention=True
        )
    matrix = weights[0].mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(7, len(tokens) * 0.55), 6))
    sns.heatmap(matrix, xticklabels=tokens, yticklabels=tokens, cmap="mako", ax=ax)
    ax.set_title("Mean attention across heads for one held-out review")
    ax.set_xlabel("Key token"); ax.set_ylabel("Query token")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "attention_heatmap.png", dpi=160)
    plt.close(fig)


def run(args):
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    train_rows, heldout = train_test_split(
        EXAMPLES, test_size=0.30, random_state=SEED,
        stratify=[label for _, label in EXAMPLES]
    )
    val_rows, test_rows = train_test_split(
        heldout, test_size=0.50, random_state=SEED,
        stratify=[label for _, label in heldout]
    )
    max_length = 16
    vocab = {PAD: 0, UNK: 1}
    for text, _ in train_rows:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    batch_size = 8
    datasets = [ReviewDataset(rows, vocab, max_length) for rows in (train_rows, val_rows, test_rows)]
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=batch_size)
    test_loader = DataLoader(datasets[2], batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentTransformer(len(vocab), max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 2 if args.quick else args.epochs
    history, best_state, best_val, wait = [], None, float("inf"), 0
    start = time.perf_counter()

    print(f"Device: {device} | examples: {len(train_rows)} train / {len(val_rows)} validation / {len(test_rows)} test")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for token_ids, mask, target in train_loader:
            token_ids, mask, target = token_ids.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(token_ids, mask), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        val_loss, val_y, val_pred = evaluate(model, val_loader, device, loss_fn)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": val_loss,
            "validation_accuracy": accuracy_score(val_y, val_pred),
        })
        print(f"Epoch {epoch + 1:02d}/{epochs}: train loss={history[-1]['train_loss']:.4f}, "
              f"validation loss={val_loss:.4f}, validation accuracy={history[-1]['validation_accuracy']:.3f}")
        if val_loss < best_val:
            best_val, best_state, wait = val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 5:
                break

    model.load_state_dict(best_state)
    test_loss, y_true, y_pred = evaluate(model, test_loader, device, loss_fn)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    elapsed = time.perf_counter() - start
    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "training_seconds": elapsed,
        "device": str(device),
        "best_validation_loss": best_val,
        "test_examples": len(test_rows),
    }
    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT / "test_metrics.csv", index=False)
    (OUT / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=["negative", "positive"], zero_division=0),
        encoding="utf-8",
    )
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["negative", "positive"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Held-out test confusion matrix")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=160); plt.close()
    fig, ax = plt.subplots(figsize=(7, 4))
    hist = pd.DataFrame(history)
    ax.plot(hist["epoch"], hist["train_loss"], marker="o", label="training")
    ax.plot(hist["epoch"], hist["validation_loss"], marker="o", label="validation")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and validation loss")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=160); plt.close(fig)
    plot_attention(model, vocab, test_rows[0][0], device)
    print("\nHeld-out test results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    print(f"\nOutputs saved in: {OUT}")


def main():
    parser = argparse.ArgumentParser(description="Transformer encoder sentiment experiment")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="run a two-epoch code check")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
