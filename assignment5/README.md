# Assignment 5 — Transfer Learning with a Pre-trained CNN

## Introduction

**Transfer learning** is a deep-learning technique in which knowledge learned
by a model on a large dataset is reused for a different but related task.
Instead of training a convolutional neural network (CNN) from scratch, a
pre-trained model is used as the starting point for a new classification
problem.

Pre-trained CNNs are commonly trained on **ImageNet**, which contains more than
one million images from 1,000 categories. Their early layers learn general
visual features such as edges, colors, textures, shapes, and patterns. Deeper
layers combine these into features that are more specific to the original
classification task. Because many visual tasks share low-level features, an
ImageNet model can often learn a new task with less data and training time.

This experiment adapts **ResNet18**, pre-trained on ImageNet, to classify the
10 classes in **CIFAR-10**.

## Main Components of a Transfer Learning Model

1. **Source model:** ResNet18 supplies weights learned from ImageNet. ResNet's
   residual (skip) connections let information and gradients bypass blocks,
   making deep networks easier to optimize.
2. **Feature extractor:** The convolutional layers convert an image into a
   compact feature representation. During feature extraction, these layers are
   frozen so their weights do not change.
3. **New classification head:** ResNet18's original 1,000-class fully connected
   layer is replaced with a new 10-output layer for CIFAR-10.
4. **Input preprocessing:** CIFAR-10 images are resized from 32×32 to 64×64
   and normalized with the mean and standard deviation used for ImageNet.
   Random crop and horizontal flip are applied only to training images.
5. **Fine-tuning stage:** The last residual block (`layer4`) is unfrozen and
   trained with a small learning rate. This adapts high-level features without
   rapidly destroying useful pre-trained weights.

## Performance Metrics

- **Training and validation loss:** show how well optimization is progressing
  and help reveal overfitting.
- **Training and validation accuracy:** measure the percentage of correctly
  classified images during training.
- **Test accuracy:** measures final generalization on data never used to update
  or select the model.
- **Precision:** of the samples predicted as a class, the proportion that are
  correct.
- **Recall:** of the actual samples in a class, the proportion found by the
  model.
- **F1-score:** harmonic mean of precision and recall. Macro averaging gives
  each class equal importance.
- **Confusion matrix:** shows which classes the model confuses.
- **Training time and trainable parameters:** indicate computational cost.

Accuracy is suitable because CIFAR-10 is balanced, while macro precision,
recall, and F1 make class-level weaknesses more visible.

## Experiment Design

### Dataset and split

- Dataset: CIFAR-10, containing 60,000 RGB images in 10 classes
- Training data: balanced subset of 10,000 images
- Validation data: balanced subset of 2,000 images
- Test data: balanced subset of 2,000 images
- Random seed: 42, with a fixed stratified training/validation split

### Controlled stages

| Stage              | Trainable layers   | Epochs | Learning rate | Purpose                                         |
| ------------------ | ------------------ | -----: | ------------: | ----------------------------------------------- |
| Feature extraction | New `fc` head only |      1 |         0.001 | Learn a classifier from fixed ImageNet features |
| Fine-tuning        | `layer4` and `fc`  |      1 |        0.0001 | Adapt high-level features carefully to CIFAR-10 |

Both stages use Adam, cross-entropy loss, the same data split, batch size, and
evaluation procedure. Fine-tuning starts from the feature-extraction model, so
it is a second training phase rather than an independent model. The checkpoint
with the best validation accuracy across both phases is used once on the test
set. This avoids using test performance for model selection.

### Run the experiment

From the repository root:

```powershell
uv run python assignment5/transfer_learning.py
```

The default configuration uses 64×64 images, a batch size of 128, and one epoch
per phase. It is designed to finish within approximately 10 minutes on an
i5-13400F CPU after downloads have completed. Runtime can still vary with power
settings and other running programs. The first run downloads CIFAR-10 and the
official ImageNet ResNet18 weights.
For a small two-phase code check, use:

```powershell
uv run python assignment5/transfer_learning.py --quick
```

Custom settings can be supplied directly:

```powershell
uv run python assignment5/transfer_learning.py --feature-epochs 5 --fine-tune-epochs 5 --batch-size 64 --image-size 224 --train-samples 45000 --val-samples 5000 --test-samples 10000
```

### Generated outputs

- `outputs/best_resnet18_cifar10.pt`: best validation checkpoint
- `outputs/training_history.csv`: metrics for every epoch and phase
- `outputs/phase_comparison.csv`: best validation result, time, and parameter
  counts for each phase
- `outputs/test_metrics.json`: final accuracy, macro precision, recall, and F1
- `outputs/learning_curves.png`: loss and accuracy curves
- `outputs/confusion_matrix.png`: test-set confusion matrix
- `outputs/sample_predictions.png`: example predictions

The default run is suitable for the CPU-limited assignment experiment.
`--quick` reduces the subsets further and is intended only to verify the code.

## Conclusion

Transfer learning reuses robust visual features and therefore usually reaches
useful accuracy faster than training a comparable CNN from random weights. In
this experiment, feature extraction first trains a lightweight classification
head. Fine-tuning then allows the last ResNet block to specialize its high-level
features for CIFAR-10. The actual conclusion should be based on the generated
CSV and JSON files: compare the best validation accuracy before and after
unfreezing, then report the final test metrics and the main confusions visible
in the confusion matrix. Fine-tuning is beneficial only if it improves held-out
validation performance, not merely training accuracy.

## FAQs

### 1. Why is transfer learning useful in Deep Learning?

It reduces the amount of labelled data, computation, and training time needed
for a new task. The model begins with useful visual features instead of random
weights and often generalizes better on small datasets.

### 2. What is a pre-trained CNN model?

It is a CNN whose parameters have already been learned from a large dataset.
Examples include ResNet, VGG, DenseNet, EfficientNet, and MobileNet models
trained on ImageNet.

### 3. What is the difference between feature extraction and fine-tuning?

In feature extraction, the pre-trained backbone is frozen and only a new output
head is trained. In fine-tuning, some or all backbone layers are unfrozen and
updated on the new dataset.

### 4. Why are the initial layers of a pre-trained CNN usually frozen?

They contain general-purpose features such as edges and textures that are useful
across many image tasks. Freezing them preserves that knowledge, lowers memory
and computation requirements, and reduces overfitting.

### 5. What is the significance of skip connections in ResNet?

A skip connection adds a block's input to its output. It gives gradients a
shorter route through the network, reduces degradation and vanishing-gradient
problems, and makes much deeper CNNs practical to train.

### 6. How can a pre-trained CNN be adapted to a new classification problem?

Match the expected input preprocessing, replace the original classifier with a
layer having one output per new class, freeze the backbone, train the new head,
and optionally unfreeze later layers for fine-tuning.

### 7. Why should a small learning rate be used during fine-tuning?

Large updates can overwrite useful pre-trained weights, a problem known as
catastrophic forgetting. A small learning rate makes controlled adjustments to
the learned representation.

### 8. What are the advantages over training a CNN from scratch?

Transfer learning normally converges faster, requires fewer labelled examples,
uses less computation, and can improve generalization—especially when the new
dataset is small or moderately sized.

### 9. What are the limitations of transfer learning?

It can perform poorly when the source and target domains are very different.
The inherited model may be large, biased, or unsuitable for deployment, and
aggressive fine-tuning may cause overfitting or catastrophic forgetting.

### 10. How can performance be evaluated?

Use a separate validation set for tuning and a test set for final evaluation.
Report loss, accuracy, class-aware precision, recall, F1-score, a confusion
matrix, learning curves, and computational cost where relevant.

### 11. When should layers of a pre-trained model be unfrozen?

Unfreeze them after the new classification head has learned stable weights.
Fine-tuning is most useful when enough labelled data is available, the target
domain differs somewhat from the source domain, or frozen features have reached
a validation-performance plateau.
