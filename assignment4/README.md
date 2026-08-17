# Assignment 4 — CNN Image Classification

This experiment trains a Convolutional Neural Network on Fashion-MNIST and
compares it with a traditional Artificial Neural Network.

## Run

```powershell
uv run python assignment4/cnn_classification.py
```

Quick two-epoch test:

```powershell
uv run python assignment4/cnn_classification.py --quick
```

Custom hyperparameters:

```powershell
uv run python assignment4/cnn_classification.py --epochs 15 --batch-size 128 --learning-rate 0.001
```

## Generated outputs

- `cnn_learning_curves.png`: CNN training/validation accuracy and loss
- `ann_learning_curves.png`: ANN training/validation accuracy and loss
- `cnn_confusion_matrix.png`: CNN classification confusion matrix
- `sample_predictions.png`: sample images with predicted and true classes
- `cnn_vs_ann.png`: CNN and ANN accuracy comparison
- `model_comparison.csv`: accuracy, loss, time, and parameter results

## Suggested screenshots

1. Dataset loading, normalization, reshaping code and dataset-shape output.
2. `build_cnn()` code and the CNN model summary.
3. Training hyperparameters in `train_model()` and the evaluation output.
4. CNN accuracy/loss learning curves.
5. Confusion matrix.
6. Sample-image predictions.
7. Final CNN-versus-ANN results table and comparison graph.

Use the full run for submitted results; `--quick` is only a program check.
