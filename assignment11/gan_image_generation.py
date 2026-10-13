"""Assignment 11: train a vanilla GAN to generate handwritten digits.

Run from the repository root:
    uv run python assignment11/gan_image_generation.py

The experiment uses scikit-learn's built-in 8x8 digits dataset and PyTorch.
It runs offline and saves training history, loss curves, and generated samples
under assignment11/outputs/.
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
IMAGE_DIM = 64
NOISE_DIM = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Train a vanilla GAN on handwritten digits")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=NOISE_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Generator(nn.Module):
    """Map a latent noise vector to a flattened 8x8 grayscale image."""

    def __init__(self, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, IMAGE_DIM),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):
    """Return one real/fake logit for each flattened image."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(IMAGE_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, image):
        return self.network(image)


def make_data(seed, batch_size):
    digits = load_digits()
    # The real images are [0, 16]; scale to [-1, 1] to match Generator.tanh.
    images = (digits.images.astype(np.float32) / 8.0) - 1.0
    images = images.reshape(-1, IMAGE_DIM)
    x_train, x_test = train_test_split(
        images, test_size=0.20, random_state=seed, stratify=digits.target
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    return loader, torch.from_numpy(x_test)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.latent_dim < 1:
        raise ValueError("epochs, batch size, and latent dimension must be positive")
    set_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, held_out = make_data(args.seed, args.batch_size)
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        sum_d = sum_g = count = 0.0
        for (real,) in loader:
            real = real.to(device)
            batch = real.shape[0]

            # D learns to assign high logits to real data and low logits to G(z).
            optimizer_d.zero_grad(set_to_none=True)
            noise = torch.randn(batch, args.latent_dim, device=device)
            fake = generator(noise)
            real_logits = discriminator(real)
            fake_logits = discriminator(fake.detach())
            loss_d = criterion(real_logits, torch.ones_like(real_logits))
            loss_d += criterion(fake_logits, torch.zeros_like(fake_logits))
            loss_d.backward()
            optimizer_d.step()

            # G learns to make D classify its generated images as real.
            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch, args.latent_dim, device=device)
            generated_logits = discriminator(generator(noise))
            loss_g = criterion(generated_logits, torch.ones_like(generated_logits))
            loss_g.backward()
            optimizer_g.step()

            sum_d += loss_d.item() * batch
            sum_g += loss_g.item() * batch
            count += batch

        history.append({
            "epoch": epoch,
            "discriminator_loss": sum_d / count,
            "generator_loss": sum_g / count,
        })
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            row = history[-1]
            print(
                f"Epoch {epoch:>3}/{args.epochs} | "
                f"D loss {row['discriminator_loss']:.4f} | "
                f"G loss {row['generator_loss']:.4f}"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT / "training_history.csv", index=False)
    save_loss_plot(history_df)
    save_samples(generator, args.latent_dim, device, args.seed)
    save_evaluation(discriminator, generator, held_out, args.latent_dim, device)
    print(f"Device: {device}")
    print(f"Saved experiment outputs to {OUT}")


@torch.no_grad()
def save_samples(generator, latent_dim, device, seed):
    generator.eval()
    rng = torch.Generator(device=device).manual_seed(seed + 1)
    noise = torch.randn(64, latent_dim, generator=rng, device=device)
    images = generator(noise).cpu().reshape(-1, 8, 8).numpy()
    # Convert from [-1, 1] back to [0, 1] for display.
    images = (images + 1.0) / 2.0
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, image in zip(axes.flat, images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("GAN-generated handwritten digits (8×8)")
    fig.tight_layout()
    fig.savefig(OUT / "generated_samples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_evaluation(discriminator, generator, held_out, latent_dim, device):
    """Save exploratory D scores; these are not a standalone image-quality metric."""
    generator.eval()
    discriminator.eval()
    real = held_out.to(device)
    noise = torch.randn(len(real), latent_dim, device=device)
    fake = generator(noise)
    scores = torch.sigmoid(torch.cat((discriminator(real), discriminator(fake)))).cpu()
    summary = pd.DataFrame({"kind": ["real", "generated"], "count": [len(real), len(fake)]})
    summary["mean_discriminator_real_probability"] = [
        scores[: len(real)].mean().item(), scores[len(real) :].mean().item()
    ]
    summary.to_csv(OUT / "evaluation_metrics.csv", index=False)


def save_loss_plot(history):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["discriminator_loss"], label="Discriminator")
    ax.plot(history["epoch"], history["generator_loss"], label="Generator")
    ax.set(title="GAN training losses", xlabel="Epoch", ylabel="BCE-with-logits loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    train(parse_args())
