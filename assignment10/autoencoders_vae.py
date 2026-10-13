"""Assignment 10: compare a dense Autoencoder and Variational Autoencoder.

Run from the repository root:
    uv run python assignment10/autoencoders_vae.py

The experiment uses scikit-learn's built-in 8x8 digits dataset and PyTorch.
It runs offline and saves training curves, latent-space plots, reconstructions,
generated samples, and summary metrics under assignment10/outputs/.
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
INPUT_DIM = 64
LATENT_DIM = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Autoencoder and VAE on digits")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Autoencoder(nn.Module):
    """Dense encoder-decoder with a two-dimensional deterministic code."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, INPUT_DIM), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class VariationalAutoencoder(nn.Module):
    """VAE whose encoder predicts a Gaussian posterior for each input."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
        )
        self.mu = nn.Linear(64, LATENT_DIM)
        self.logvar = nn.Linear(64, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, INPUT_DIM), nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        # Sampling is stochastic, while this form keeps gradients flowing to
        # the encoder's mean and variance parameters.
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(reconstruction, x, mu, logvar):
    # Sum per-pixel BCE and per-latent-dimension KL, then average over examples.
    reconstruction_loss = nn.functional.binary_cross_entropy(
        reconstruction, x, reduction="none"
    ).sum(dim=1)
    kl_per_example = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    )
    total = (reconstruction_loss + kl_per_example).mean()
    return total, reconstruction_loss.mean(), kl_per_example.mean()


def make_data(seed):
    digits = load_digits()
    # Images have integer values from 0 to 16; scaling to [0, 1] suits sigmoid
    # outputs and binary cross-entropy reconstruction loss.
    x = (digits.data.astype(np.float32) / 16.0)
    labels = digits.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train, test, x_train, x_test, y_train, y_test


def train_model(name, model, train_loader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for epoch in range(epochs):
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
        count = 0
        for x, _labels in train_loader:
            optimizer.zero_grad()
            if name == "VAE":
                reconstruction, mu, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss(reconstruction, x, mu, logvar)
            else:
                reconstruction, _ = model(x)
                recon_loss = nn.functional.binary_cross_entropy(
                    reconstruction, x, reduction="sum"
                ) / len(x)
                kl_loss = torch.zeros(())
                loss = recon_loss
            loss.backward()
            optimizer.step()
            batch_count = len(x)
            totals["loss"] += loss.item() * batch_count
            totals["reconstruction"] += recon_loss.item() * batch_count
            totals["kl"] += kl_loss.item() * batch_count
            count += batch_count
        row = {"model": name, "epoch": epoch + 1}
        row.update({key: value / count for key, value in totals.items()})
        history.append(row)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"{name} epoch {epoch + 1:>3}/{epochs}: "
                f"loss={row['loss']:.4f}, recon={row['reconstruction']:.4f}, "
                f"KL={row['kl']:.4f}"
            )
    return pd.DataFrame(history)


@torch.no_grad()
def evaluate(model, x_test, name):
    model.eval()
    x = torch.from_numpy(x_test)
    if name == "VAE":
        mu, logvar = model.encode(x)
        reconstruction = model.decode(mu)
        total, recon, kl = vae_loss(reconstruction, x, mu, logvar)
        latent = mu
    else:
        reconstruction, latent = model(x)
        recon = nn.functional.binary_cross_entropy(reconstruction, x, reduction="sum") / len(x)
        total, kl = recon, torch.tensor(0.0)
    return {
        "model": name,
        "test_objective_per_example": total.item(),
        "test_reconstruction_bce_per_pixel": nn.functional.binary_cross_entropy(
            reconstruction, x
        ).item(),
        "test_kl_per_example": kl.item(),
    }, reconstruction.numpy(), latent.numpy()


def save_training_plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in history.groupby("model"):
        axes[0].plot(rows["epoch"], rows["loss"], label=name)
        axes[1].plot(rows["epoch"], rows["reconstruction"], label=f"{name} reconstruction")
        if name == "VAE":
            axes[1].plot(rows["epoch"], rows["kl"], label="VAE KL")
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Loss per example")
    axes[1].set(title="VAE loss components", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=160)
    plt.close(fig)


def save_latent_plot(latent_by_model, labels):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, (name, latent) in zip(axes, latent_by_model.items()):
        points = axis.scatter(latent[:, 0], latent[:, 1], c=labels, cmap="tab10", s=13, alpha=0.8)
        axis.set(title=f"{name} latent space (test set)", xlabel="z₁", ylabel="z₂")
        axis.grid(alpha=0.2)
    fig.subplots_adjust(right=0.88, wspace=0.28)
    colorbar_axis = fig.add_axes([0.91, 0.2, 0.018, 0.6])
    fig.colorbar(points, cax=colorbar_axis, ticks=range(10), label="Digit class")
    fig.savefig(OUT / "latent_spaces.png", dpi=160)
    plt.close(fig)


def save_reconstruction_plot(x_test, reconstructions):
    count = 10
    fig, axes = plt.subplots(3, count, figsize=(12, 4.2))
    rows = [("Original", x_test), ("Autoencoder", reconstructions["Autoencoder"]),
            ("VAE", reconstructions["VAE"])]
    for row_index, (label, images) in enumerate(rows):
        for column in range(count):
            axes[row_index, column].imshow(images[column].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(label, rotation=0, labelpad=40, va="center")
    fig.suptitle("Held-out digit reconstructions", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "reconstructions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_generated_grid(vae):
    vae.eval()
    # Use a fixed grid of prior points to make generated samples repeatable.
    axis = torch.linspace(-2.0, 2.0, 10)
    z = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), dim=-1).reshape(-1, 2)
    generated = vae.decode(z).numpy()
    fig, axes = plt.subplots(10, 10, figsize=(7, 7))
    for index, axis in enumerate(axes.flat):
        axis.imshow(generated[index].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        axis.axis("off")
    fig.suptitle("VAE samples decoded from a 2D standard-normal prior")
    fig.tight_layout()
    fig.savefig(OUT / "vae_generated_grid.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs and batch size must be positive; learning rate must be positive")
    set_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    train, _, _, x_test, _, y_test = make_data(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)

    models = {"Autoencoder": Autoencoder(), "VAE": VariationalAutoencoder()}
    histories, metrics, reconstructions, latent_by_model = [], [], {}, {}
    for name, model in models.items():
        print(f"\nTraining {name} on {len(train)} training images")
        history = train_model(name, model, train_loader, args.epochs, args.learning_rate)
        histories.append(history)
        row, reconstruction, latent = evaluate(model, x_test, name)
        metrics.append(row)
        reconstructions[name] = reconstruction
        latent_by_model[name] = latent

    history = pd.concat(histories, ignore_index=True)
    summary = pd.DataFrame(metrics)
    history.to_csv(OUT / "training_history.csv", index=False)
    summary.to_csv(OUT / "evaluation_metrics.csv", index=False)
    save_training_plot(history)
    save_latent_plot(latent_by_model, y_test)
    save_reconstruction_plot(x_test, reconstructions)
    save_generated_grid(models["VAE"])
    print("\nHeld-out evaluation:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved experiment outputs to {OUT}")


if __name__ == "__main__":
    main()
