# Assignment 3 — ANN Experiment Design

This assignment compares activation functions, optimization algorithms, and
regularization techniques using a TensorFlow/Keras ANN on Fashion-MNIST.

## Experiment setup

- Dataset: Fashion-MNIST (60,000 training and 10,000 test grayscale images)
- Split: 50,000 training, 10,000 validation, 10,000 test
- Baseline: Flatten → Dense(256) → Dense(128) → Dense(10, softmax)
- Activations: ReLU, Sigmoid, Tanh, Leaky ReLU
- Optimizers: SGD, Adam, RMSprop, Adagrad
- Regularization: Dropout, L1, L2, Batch Normalization, Early Stopping
- Metrics: training/validation accuracy and loss, test accuracy and loss, time
- Selection: highest validation accuracy; the test set is used only for evaluation

Each comparison changes one factor while keeping the others at the baseline
(ReLU + Adam + no regularization). The final experiment combines the best
activation, optimizer, and regularization found in the controlled studies.

## Run

From the repository root:

```powershell
uv run python assignment3/ann_experiments.py
```

The full run trains 16 models for up to 15 epochs. For a fast code check:

```powershell
uv run python assignment3/ann_experiments.py --quick
```

You can also select the maximum epochs and batch size:

```powershell
uv run python assignment3/ann_experiments.py --epochs 20 --batch-size 128
```

## Screenshot checklist

Capture these relevant sections for the report:

1. Dataset loading/preprocessing code and the `ENVIRONMENT AND DATASET` output.
2. `build_ann`, `activation_layer`, and the baseline `model.summary()` output.
3. `make_optimizer` and the optimizer-results console output.
4. The regularization branches in `build_ann` and EarlyStopping callback.
5. The final `COMPLETE RESULTS` table and `BEST-PERFORMING COMBINATION` output.
6. Comparison plots and selected per-run accuracy/loss plots from `outputs/plots`.

## Generated outputs

- `outputs/experiment_results.csv`: all metrics and training times
- `outputs/best_model_summary.json`: selected combination and final test results
- `outputs/histories/*.json`: epoch-wise accuracy and loss for every model
- `outputs/plots/*.png`: individual learning curves and comparison charts

`--quick` is intended only to verify the program. Use the full run for the
results and screenshots submitted in the assignment.
