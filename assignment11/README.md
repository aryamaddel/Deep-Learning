# Assignment 11 — Generative Adversarial Networks

## Title

Design and implementation of a Generative Adversarial Network (GAN) for
handwritten digit generation.

## Aim

- To generate synthetic handwritten digits using a GAN.
- To understand the roles of the generator and discriminator.
- To implement the vanilla GAN adversarial objective with binary cross-entropy.
- To inspect training losses and generated samples.

## Theory

A **Generative Adversarial Network (GAN)** is a generative model made of two
neural networks trained together through a competitive objective. The
**generator** maps a random latent vector to a synthetic example. The
**discriminator** receives a real or generated example and predicts whether it
comes from the training data. The generator improves when it produces samples
that the discriminator mistakes for real; the discriminator improves by
distinguishing the two sources.

### Generator and Discriminator

Let $z$ be a random noise vector sampled from a simple prior such as
$p_z(z)=\mathcal{N}(0,I)$. The generator $G(z;\theta_G)$ transforms it into an
image. The discriminator $D(x;\theta_D)$ returns the probability (or, in this
implementation, a logit that can be converted to a probability) that $x$ is a
real training example.

This experiment uses fully connected networks for the small 8×8 grayscale
images. The generator has two ReLU hidden layers and a `tanh` output, which
matches images scaled to $[-1,1]$. The discriminator has two Leaky ReLU hidden
layers and one output logit. `BCEWithLogitsLoss` combines the sigmoid and binary
cross-entropy computation in a numerically stable form.

### Vanilla GAN Objective and Loss Functions

The original minimax objective is:

$$\min_G\max_D V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)]+
\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

The discriminator maximizes this objective: it should predict 1 for real data
and 0 for generated data. Equivalently, its binary cross-entropy loss is
$-\mathbb{E}_{x}[\log D(x)]-\mathbb{E}_{z}[\log(1-D(G(z)))]$. The generator's
original minimax loss is $\mathbb{E}_{z}[\log(1-D(G(z)))]$. In practice,
vanilla GANs commonly use the **non-saturating generator loss**
$-\mathbb{E}_{z}[\log D(G(z))]$, which gives stronger gradients when the
discriminator confidently rejects early generated samples. The script uses
this non-saturating form with binary cross-entropy targets of 1 for generated
images in the generator update.

The networks are optimized in alternating steps. During a discriminator step,
generated images are detached so gradients update only the discriminator.
During a generator step, gradients pass through the discriminator's output to
update the generator. Separate optimizers keep each update associated with its
own network parameters.

### Step-by-step Process

1. Load grayscale handwritten digit images and scale pixel values to $[-1,1]$.
2. Sample a minibatch of real images and latent noise vectors.
3. Generate synthetic images from the noise.
4. Update the discriminator using real images as target 1 and generated images
   as target 0.
5. Draw a fresh batch of noise and update the generator so its generated
   images receive target 1 from the discriminator.
6. Repeat these alternating updates for each minibatch and epoch.
7. Save the loss history, a loss plot, generated image grid, and exploratory
   discriminator scores for held-out real and generated images.

### Architecture

```text
Latent noise z (32 values)
          │
          ▼
Generator: Dense(128) → ReLU → Dense(256) → ReLU → Dense(64) → tanh
          │
          ▼
Generated 8×8 grayscale digit ─────┐
                                    ▼
Real 8×8 grayscale digit ───────► Discriminator:
                                    Dense(256) → LeakyReLU(0.2)
                                    → Dense(128) → LeakyReLU(0.2)
                                    → Dense(1) logit
```

The generator maps 32 random values to 64 pixel values. The discriminator
classifies the flattened 64-value image. Both networks are dense because the
selected dataset images are only 8×8; convolutional layers are more suitable
for larger images with spatial structure.

## Experiment Design

### Task and data

The script learns to generate grayscale handwritten digits using scikit-learn's
built-in `load_digits` dataset. It contains 1,797 labeled 8×8 images with pixel
intensities from 0 to 16. Pixels are normalized to $[-1,1]$. Labels are used
only to make a stratified held-out split; the GAN receives no labels and is
trained unsupervised on the training images. The data ships with scikit-learn,
so no download is required. These low-resolution samples are for learning the
GAN training process and are not comparable to modern high-resolution image
generation.

### Model and settings

| Setting | Default |
|---|---:|
| Generator hidden dimensions | 128, 256 |
| Discriminator hidden dimensions | 256, 128 |
| Image dimensions | 8×8 grayscale (64 values) |
| Latent noise dimension | 32 |
| Generator output activation | `tanh` |
| Discriminator output | One logit; no final sigmoid layer |
| Loss | Binary cross-entropy with logits |
| Optimizers | Separate Adam optimizers |
| Learning rate | 0.0002 |
| Adam betas | (0.5, 0.999) |
| Batch size | 64 |
| Epochs | 100 |
| Random seed | 42 |

The discriminator loss is the sum of the mean real-image and fake-image
binary cross-entropies. The generator uses the non-saturating objective, which
labels its generated images as real for its update. GAN losses are coupled and
do not decrease monotonically; loss values alone are not a reliable measure
of image quality. Inspect the fixed-seed generated grid along with the loss
curves. The saved discriminator probabilities are diagnostic scores, not an
independent or definitive quality metric.

### Run the experiment

From the repository root:

```powershell
uv run python assignment11/gan_image_generation.py
```

Optional parameters can be changed from the command line:

```powershell
uv run python assignment11/gan_image_generation.py --epochs 150 --batch-size 32 --learning-rate 0.0001 --latent-dim 48 --seed 7
```

The program uses the PyTorch, NumPy, pandas, scikit-learn, and matplotlib
dependencies already declared in the repository. It automatically uses CUDA
when available and otherwise runs on CPU.

### Generated outputs

- `outputs/training_history.csv`: generator and discriminator mean losses by
  epoch.
- `outputs/training_curves.png`: generator and discriminator loss curves.
- `outputs/generated_samples.png`: fixed-seed grid of 64 generated digits.
- `outputs/evaluation_metrics.csv`: mean discriminator real-probability for
  held-out real images and generated images.

The experiment does not report fabricated results. Run the script to obtain
metrics and sample images for the current environment and settings. A single
GAN run can vary substantially with random seed and hyperparameters, and the
held-out discriminator scores should not be interpreted as a general image
quality score.

## Conclusion

This experiment implements a vanilla GAN with a generator that maps random
noise to 8×8 grayscale images and a discriminator that learns to distinguish
generated images from real handwritten digits. Alternating updates train the
discriminator to classify correctly and the generator to fool it. The saved
sample grid gives a qualitative view of the generator, while the training
curves show the changing adversarial losses. GAN training can be unstable, so
neither loss curves nor discriminator scores alone establish sample quality;
generated images should be inspected and results interpreted in light of this
small, low-resolution dataset.

## FAQs

### 1. Differentiate a generative model from a discriminative model.

A generative model learns a representation of how data is produced and can
sample new examples, such as a GAN generator producing digit images. A
discriminative model learns to distinguish classes or estimate labels given
input data; the GAN discriminator distinguishes real images from generated
ones. A discriminator's real/fake output is not itself a generated example.

### 2. Write the objective function of a vanilla GAN.

The minimax objective is
$\min_G\max_D \mathbb{E}_{x\sim p_{data}}[\log D(x)] +
\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$. The discriminator maximizes the
objective, while the generator minimizes it. In practice the generator often
uses the non-saturating loss $-\mathbb{E}_{z}[\log D(G(z))]$ for stronger
gradients early in training; this is the generator loss used by this script.

### 3. What happens if the discriminator becomes too strong early in training?

It can confidently reject generated images, leaving the original minimax
generator objective with very small gradients. The generator then learns
slowly or stalls, and the two networks stop making balanced progress. The
non-saturating generator loss helps with this gradient issue, while learning
rate and update balance may also need adjustment.

### 4. Why is random noise supplied to the generator?

Noise provides a latent input that the generator can transform into varied
outputs. Different noise vectors can produce different digits, allowing the
generator to model a distribution rather than reproduce one fixed image. A
trained generator maps regions of this latent space to patterns resembling
training examples.

### 5. What is the difference between a GAN and a Variational Autoencoder (VAE)?

A GAN learns through adversarial competition between a generator and a
discriminator; it does not require an encoder or direct per-example
reconstruction loss. A VAE has an encoder and decoder and optimizes a
reconstruction term plus a KL-divergence regularizer on the latent
distribution. GANs can produce sharp samples but may be difficult to train and
can suffer mode collapse. VAEs have an explicit probabilistic latent model and
often train more predictably, though their outputs can be smoother.

### 6. Explain mode collapse and one possible mitigation strategy.

Mode collapse occurs when the generator produces a narrow set of outputs even
though the training data contains many different patterns. One mitigation is
to use minibatch discrimination, which lets the discriminator detect a lack
of variety across samples in a batch. Other approaches include feature
matching, Wasserstein GAN objectives with gradient penalty, or tuning the
relative learning rates and update frequency. No one method guarantees that
collapse is eliminated.

### 7. Why are separate optimizers used for the generator and discriminator?

The generator and discriminator have different parameters and objectives, so
each needs its own optimizer state and update step. Alternating the optimizers
lets each loss update the intended network; when training the generator,
gradients pass through the discriminator but only generator parameters are
stepped.

### 8. How would you adapt this experiment to generate RGB images?

Use three-channel images and a convolutional generator and discriminator. The
generator can project noise to a small feature map, upsample through
transposed convolutions or interpolation plus convolutions, and output three
channels with `tanh` when pixels are scaled to $[-1,1]$. The discriminator can
use strided convolutions to reduce spatial dimensions before its real/fake
logit. Increase data and model capacity, apply suitable normalization and
regularization, and monitor sample diversity and quality as well as losses.
