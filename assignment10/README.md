# Assignment 10 — Autoencoders and Variational Autoencoders

## Title

Implementation of Autoencoders and Variational Autoencoders (VAE).

## Aim

- To study the architecture and working principles of Autoencoders and VAEs.
- To learn compact latent representations by reconstructing input images.
- To understand the reparameterization trick and KL-divergence regularization.
- To compare deterministic reconstruction with probabilistic generation.
- To visualize learned latent spaces and samples decoded from a prior.

## Theory

An **Autoencoder (AE)** is a neural network trained without class labels to
compress an input into a lower-dimensional representation and reconstruct the
input from it. It has an **encoder**, a **latent representation** (also called
the code), and a **decoder**. The model learns useful structure because the
code has limited capacity and the training objective penalizes reconstruction
errors. An Autoencoder does not necessarily learn a useful or smooth latent
space; its behavior depends on its architecture, objective, and data.

### Autoencoder Architecture

For input $x$, the encoder $f_\theta$ produces a code $z$, and the decoder
$g_\phi$ produces a reconstruction $\hat{x}$:

$$z=f_\theta(x), \qquad \hat{x}=g_\phi(z).$$

Training minimizes a reconstruction loss such as mean squared error (MSE) for
continuous values or binary cross-entropy (BCE) for values scaled to $[0,1]$.
This experiment uses BCE. A low reconstruction loss means that the output
matches the input under that loss; it does not guarantee that the code is
meaningful for every downstream task.

### Variational Autoencoder (VAE)

A **VAE** is a latent-variable generative model. Instead of mapping an input to
one fixed code, its encoder predicts the parameters of a distribution over
codes, typically a diagonal Gaussian:

$$q_\phi(z\mid x)=\mathcal{N}(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))).$$

The decoder models the data given a latent point. Training balances
reconstruction quality against a regularization term that encourages the
encoded distributions to stay near a chosen prior, commonly
$p(z)=\mathcal{N}(0,I)$. The negative evidence lower bound (negative ELBO) is:

$$\mathcal{L}(x)=\mathbb{E}_{q_\phi(z\mid x)}[-\log p_\theta(x\mid z)] + D_{KL}(q_\phi(z\mid x)\Vert p(z)).$$

The first term rewards faithful reconstruction. The **KL divergence** term
measures the difference between the encoded distribution and the prior and
regularizes the latent space. If the KL term is too influential, the model may
ignore the latent code (posterior collapse); if it is omitted, arbitrary prior
samples may decode poorly.

### Reparameterization Trick

Directly sampling $z\sim\mathcal{N}(\mu,\sigma^2)$ interrupts ordinary
backpropagation through the sampling operation. The reparameterization trick
rewrites the sample as:

$$z=\mu+\sigma\odot\epsilon, \qquad \epsilon\sim\mathcal{N}(0,I).$$

Randomness is isolated in $\epsilon$, so gradients can flow through $\mu$ and
$\sigma$. The network predicts log variance for numerical convenience, with
$\sigma=\exp(\tfrac12\log\sigma^2)$.

### AE and VAE Compared

| Property | Autoencoder | Variational Autoencoder |
|---|---|---|
| Encoder output | One deterministic code | Mean and log variance of a distribution |
| Training objective | Reconstruction loss | Reconstruction loss plus KL divergence |
| Latent space | Not constrained to a prior | Regularized toward a selected prior |
| Generation | Decode a chosen code; arbitrary codes may be invalid | Sample from the prior and decode |
| Main trade-off | Can emphasize reconstruction fidelity | Balances fidelity with a structured, sampleable space |

## Experiment Design

### Task and data

The program trains a dense AE and VAE to reconstruct grayscale handwritten
digits. It uses scikit-learn's built-in `load_digits` dataset: 1,797 labeled
8×8 images with pixel intensities from 0 to 16. Images are flattened to 64
features and scaled to $[0,1]$. Labels are used only to color the latent-space
visualizations; neither model receives labels during training. A stratified
80/20 split leaves 1,437 training images and 360 held-out images. The dataset
is included with scikit-learn, so no download is needed.

### Model and settings

Both networks use a two-dimensional bottleneck, allowing a direct 2D plot. The
encoder and decoder are fully connected networks with ReLU hidden layers and a
sigmoid output. The AE encoder returns a single two-value code. The VAE encoder
returns two two-value vectors, $\mu$ and $\log\sigma^2$; the decoder receives
a reparameterized sample while training, and the mean code for held-out
reconstructions.

| Setting | Default |
|---|---:|
| Input | 64 scaled pixel values per image |
| Hidden dimensions | 128, 64 |
| Latent dimension | 2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 40 |
| Random seed | 42 |
| Reconstruction objective | Binary cross-entropy |
| VAE regularizer | KL divergence to standard normal prior |

The reported VAE objective is the sum of per-image BCE over the 64 input
features plus KL divergence, averaged over examples. The AE trains on the same
per-image summed BCE without a KL term. Its `test_reconstruction_bce_per_pixel`
metric divides the summed BCE by 64 for easier interpretation. Latent plots
show held-out codes colored by the known digit class. The sample grid decodes a
fixed grid of points in the VAE's two-dimensional standard-normal prior.

### Run the experiment

From the repository root:

```powershell
uv run python assignment10/autoencoders_vae.py
```

Optional settings can be changed from the command line:

```powershell
uv run python assignment10/autoencoders_vae.py --epochs 60 --batch-size 32 --learning-rate 0.0008 --seed 7
```

The script uses the PyTorch, NumPy, pandas, scikit-learn, and matplotlib
dependencies already declared by the repository. It trains on CPU by default
and saves its generated files under `assignment10/outputs/`.

### Generated outputs

- `outputs/training_history.csv`: objective, reconstruction, and KL values by
  epoch.
- `outputs/evaluation_metrics.csv`: held-out objective, per-pixel BCE, and KL.
- `outputs/training_curves.png`: training objective and VAE loss components.
- `outputs/reconstructions.png`: original and reconstructed held-out digits.
- `outputs/latent_spaces.png`: AE and VAE 2D encodings, colored by digit label.
- `outputs/vae_generated_grid.png`: images decoded from prior points.

The loss scales are specific to the chosen BCE reduction and input dimensions.
Reconstruction plots and per-pixel loss help interpret quality, while latent
plots provide a qualitative view of organization. Results on this small
8×8 dataset should not be treated as a benchmark for larger images or other
data distributions.

### Observed results

The default 40-epoch run with seed 42 produced the following held-out metrics:

| Model | Objective per example | Reconstruction BCE per pixel | KL per example |
|---|---:|---:|---:|
| Autoencoder | 21.1325 | 0.3302 | 0.0000 |
| VAE | 24.2852 | 0.3470 | 2.0750 |

On this run, the deterministic Autoencoder achieved a lower reconstruction
error. The VAE's objective also includes KL divergence, so its total should
not be interpreted as a like-for-like reconstruction score. The prior grid
shows a gradual range of decoded digit-like forms, though the low-resolution
images and two-dimensional latent space limit sample detail. Metrics can vary
with training settings and library versions; the generated CSV files contain
the run's values.

## Conclusion

The experiment implements a deterministic Autoencoder and a probabilistic VAE
with the same basic encoder-decoder structure. The Autoencoder learns a compact
code by minimizing reconstruction error. The VAE additionally learns a mean
and variance for each input, uses the reparameterization trick to train through
sampling, and uses KL divergence to regularize codes toward a standard-normal
prior. This regularization makes it possible to generate images by sampling
latent points and decoding them. The saved reconstructions, loss curves, and
latent plots illustrate the trade-off between faithful reconstruction and a
structured latent space.

## FAQs

### 1. What are the main components of an Autoencoder?

An Autoencoder has an encoder, a latent representation (code), and a decoder.
The encoder compresses the input, and the decoder maps the code back to a
reconstruction.

### 2. What is a latent representation?

It is the compact vector produced by the encoder. It is intended to retain
information needed to reconstruct the input. In this experiment it has two
values so it can be plotted, although a two-value code limits how much detail
can be represented.

### 3. What is the role of the Decoder?

The decoder maps a latent code to an output in the input's space. During
training it learns to reproduce the original input from the code; in a VAE it
can also map prior samples to generated examples.

### 4. What is reconstruction loss?

It measures the difference between the input and its reconstruction. Common
choices include MSE and BCE. This assignment scales pixels to $[0,1]$ and uses
BCE with sigmoid decoder outputs.

### 5. How does a VAE differ from a traditional Autoencoder?

A traditional Autoencoder maps each input to a deterministic code and usually
optimizes reconstruction loss. A VAE predicts a probability distribution over
codes and adds a KL-divergence term to encourage that distribution to resemble
a prior, supporting sampling and generation.

### 6. What are mean and variance in a VAE?

The encoder outputs a mean vector $\mu$ and a log-variance vector
$\log\sigma^2$. Together they parameterize the approximate posterior
distribution for the input's latent code. The model predicts log variance
rather than variance directly for numerical convenience.

### 7. What is the reparameterization trick?

It expresses a latent sample as $z=\mu+\sigma\odot\epsilon$, where
$\epsilon$ is standard normal noise. This keeps sampling stochastic while
allowing gradients to reach the encoder parameters.

### 8. What is KL divergence in a VAE?

KL divergence measures how one probability distribution differs from another.
In a VAE it measures how far the encoder's approximate posterior for an input
is from the selected latent prior, usually a standard normal distribution.

### 9. Why is KL divergence used as a regularization term?

It encourages encoded distributions to occupy a latent space with a known
structure. When the prior is standard normal, prior samples are more likely to
fall in regions the decoder learned to use. Excessive regularization can,
however, reduce reconstruction detail or cause posterior collapse.

### 10. How can a VAE generate new data?

Sample a latent vector from the prior (for example, $z\sim\mathcal{N}(0,I)$)
and pass it through the decoder. The output is a new example from the decoder's
learned data distribution. Sample quality depends on training and on how well
the model uses its latent variables.

### 11. What is latent-space visualization?

It is a plot of encoded examples in a low-dimensional space. Here the code is
already two-dimensional, so held-out digits can be plotted directly and colored
by their labels. The label colors aid interpretation but are not used to train
the unsupervised models.

### 12. How can Autoencoders be used for anomaly detection?

Train an Autoencoder on representative normal examples, then compare each
example's reconstruction error with a threshold selected on validation data.
Examples with unusually high error may be anomalous. This method can miss
anomalies that reconstruct well, and thresholding must account for normal data
variation; high reconstruction error alone is not proof of an anomaly.
