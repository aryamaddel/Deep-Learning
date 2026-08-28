# Assignment 6 — RNN, LSTM, and GRU

## Title

Implementation of Recurrent Neural Networks (RNN), Long Short-Term Memory
(LSTM), and Gated Recurrent Unit (GRU) for sequence prediction and sentiment
analysis.

## Aim

- To study the architecture and working principles of RNN, LSTM, and GRU.
- To implement RNN, LSTM, and GRU models using TensorFlow/Keras.
- To apply recurrent neural networks to next-step sequence prediction.
- To perform sentiment analysis on movie reviews.
- To compare RNN, LSTM, and GRU using suitable evaluation metrics.

## Theory

Recurrent Neural Networks are designed to process sequential and time-dependent
data. Unlike a feed-forward network, a recurrent layer passes a hidden state
from one time step to the next. The output at time *t* can therefore depend on
the current input and information from earlier time steps.

RNNs are commonly used for time-series prediction, text classification,
sentiment analysis, language modelling, speech recognition, and machine
translation. A standard RNN is effective for short dependencies, but repeated
multiplication during backpropagation can make gradients vanish or explode.
LSTM and GRU introduce gates that control information flow and improve learning
over longer sequences.

## RNN, LSTM, and GRU Architecture

### Standard RNN

A standard RNN combines the current input with its previous hidden state:

`h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)`

The hidden state acts as the network's memory. Its simple structure gives an
RNN relatively few parameters and fast inference, but information from distant
time steps is difficult to retain.

### LSTM

An LSTM maintains both a hidden state and a cell state. Three principal gates
control the memory:

- **Forget gate:** decides what old cell-state information should be removed.
- **Input gate:** decides what new information should be stored.
- **Output gate:** decides which cell-state information becomes the hidden
  state and output.

The cell state provides a controlled path through time, reducing the vanishing
gradient problem. This capability comes with a larger parameter count and more
computation than a standard RNN.

### GRU

A GRU is a streamlined gated architecture with no separate cell state:

- **Update gate:** balances retained past information and new information.
- **Reset gate:** controls how much past information contributes to the new
  candidate hidden state.

GRUs generally have fewer parameters than comparable LSTMs and can train more
quickly, although neither architecture is universally superior.

## Sequence Prediction

Sequence prediction estimates a future value from an ordered window of earlier
values. This experiment creates a deterministic waveform from two sine waves.
Every sample contains 30 consecutive values, and its target is the immediately
following value. The samples are split chronologically into training,
validation, and test sets so future observations do not leak into training.

RNN, LSTM, and GRU models use the same 32-unit recurrent layer, dense output
layer, optimizer, batch size, and data split. Mean squared error is the training
loss. Final models are compared using:

- **MAE:** average absolute prediction error.
- **RMSE:** penalizes large errors more strongly than MAE.
- **R²:** proportion of target variance explained by the predictions.

Lower MAE and RMSE and higher R² indicate better sequence predictions.

## Sentiment Analysis

Sentiment analysis identifies the opinion expressed in text. The IMDB dataset
contains movie reviews labelled as positive or negative. The processing steps
are:

1. Keep the 10,000 most frequent word IDs.
2. Select balanced training, validation, and test subsets.
3. Truncate or pad every review to 100 tokens.
4. Convert word IDs into trainable 32-dimensional embedding vectors.
5. Process the embedded sequence with an RNN, LSTM, or GRU layer.
6. Use a sigmoid output to predict the probability of positive sentiment.

Padding provides a uniform tensor shape for batching. The embedding layer
learns dense numerical representations in which words useful for the task can
acquire similar features.

Sentiment models are evaluated with accuracy, precision, recall, F1-score,
ROC-AUC, and a confusion matrix. Because the selected subsets are balanced,
accuracy is interpretable; the additional metrics still reveal asymmetric
classification behaviour.

## Experiment Design

### Controlled comparison

Only the recurrent cell changes between RNN, LSTM, and GRU. Within each task,
all models receive the same data, random seed, hidden units, optimizer, batch
size, epoch limit, and evaluation procedure. Validation data selects the best
epoch through early stopping; test data is used only for final evaluation.

| Setting | Sequence prediction | Sentiment analysis |
|---|---:|---:|
| Recurrent units | 32 | 32 |
| Sequence length | 30 values | 100 tokens |
| Training samples | About 1,700 windows | 6,000 reviews |
| Validation samples | About 370 windows | 1,500 reviews |
| Test samples | About 370 windows | 2,000 reviews |
| Epoch limit | 6 | 2 |
| Batch size | 64 | 128 |
| Optimizer | Adam | Adam |
| Loss | Mean squared error | Binary cross-entropy |

The defaults are deliberately CPU-friendly and should normally finish within
10 minutes on the i5-13400F used for this repository. Runtime depends on system
load. The first run also downloads the compressed IMDB dataset.

### Run

From the repository root:

```powershell
uv run assignment6/rnn_lstm_gru.py
```

Small smoke test:

```powershell
uv run assignment6/rnn_lstm_gru.py --quick
```

Custom epoch limits:

```powershell
uv run assignment6/rnn_lstm_gru.py --sequence-epochs 8 --sentiment-epochs 3
```

### Generated outputs

- `outputs/sequence_results.csv`: MAE, RMSE, R², time, and parameters.
- `outputs/sentiment_results.csv`: classification metrics, time, and parameters.
- `outputs/training_history.csv`: epoch-wise metrics for all six runs.
- `outputs/sequence_predictions.png`: actual and predicted waveform values.
- `outputs/sentiment_accuracy.png`: sentiment training/validation accuracy.
- `outputs/sentiment_loss.png`: sentiment training/validation loss.
- `outputs/sentiment_confusion_matrices.png`: RNN, LSTM, and GRU test matrices.

Use the default run for submitted results; `--quick` is only a program check.
The measured CSV values should be quoted in the observation and conclusion.

## Observed Results

The verified default run on the repository's i5-13400F CPU produced:

| Sequence model | Test MAE | Test RMSE | Test R² | Training time |
|---|---:|---:|---:|---:|
| RNN | 0.0227 | 0.0263 | 0.9987 | 2.7 s |
| LSTM | 0.0519 | 0.0645 | 0.9924 | 4.0 s |
| GRU | 0.0573 | 0.0730 | 0.9903 | 4.0 s |

| Sentiment model | Accuracy | Precision | Recall | F1 | AUC | Training time |
|---|---:|---:|---:|---:|---:|---:|
| RNN | 0.5425 | 0.5400 | 0.5740 | 0.5565 | 0.5553 | 4.3 s |
| LSTM | 0.8165 | 0.8763 | 0.7370 | 0.8007 | 0.9054 | 6.5 s |
| GRU | 0.8065 | 0.8511 | 0.7430 | 0.7934 | 0.8864 | 6.7 s |

The six measured training stages took about 28 seconds in total. TensorFlow
startup, preprocessing, evaluation, and plot generation add some wall-clock
time, but the complete cached-data run remained well below 10 minutes. Small
metric differences are possible on later runs because neural-network operations
and hardware execution are not always perfectly deterministic.

## Conclusion

RNN, LSTM, and GRU all learned the short, regular waveform successfully. The
standard RNN performed best on this task with an RMSE of 0.0263 and R² of
0.9987; its simpler memory was sufficient for the 30-step deterministic input.
For sentiment analysis, LSTM performed best with 81.65% accuracy, an F1-score
of 0.8007, and an AUC of 0.9054. GRU was close behind, while the standard RNN
reached only 54.25% accuracy. These results illustrate that gating becomes
valuable for longer and more complex token relationships. GRU used fewer
parameters than LSTM, but LSTM gave the strongest classification results in
this run. Thus, architecture choice should depend on sequence complexity,
accuracy requirements, and computational cost rather than a universal ranking.

## FAQs

### 1. Why are RNNs suitable for sequential data?

They process elements in order and pass a hidden state between time steps, so a
current prediction can use information from earlier elements.

### 2. What is the vanishing gradient problem in RNNs?

During backpropagation through many time steps, gradients can repeatedly shrink
toward zero. Early time steps then receive almost no learning signal, making
long-range dependencies difficult to learn.

### 3. What are the main gates in an LSTM?

The forget gate removes unneeded memory, the input gate controls new memory,
and the output gate controls what the cell exposes as its hidden state.

### 4. How does GRU differ from LSTM?

A GRU uses reset and update gates and combines memory with the hidden state. An
LSTM has input, forget, and output gates plus a separate cell state. GRU is
usually smaller, while LSTM offers more explicit memory control.

### 5. Why are LSTM and GRU preferred for long sequences?

Their gates create controlled paths for information and gradients, allowing
important signals to persist longer than in a standard RNN.

### 6. What is sequence prediction?

It is the prediction of the next value, values, or category from an ordered
history, such as forecasting the next measurement in a time series.

### 7. What is sentiment analysis?

It is an NLP classification task that identifies an opinion or emotional
polarity, commonly positive, negative, or neutral, from text.

### 8. What is tokenization in Natural Language Processing?

Tokenization divides text into units such as words, subwords, or characters and
maps those units to numerical IDs that a model can process.

### 9. Why is padding required for text sequences?

Reviews have different lengths. Padding makes them equal length so multiple
reviews can be placed into one fixed-shaped training batch.

### 10. What is an embedding layer?

It maps each discrete token ID to a trainable dense vector. These vectors let
the network learn useful semantic and task-specific relationships among words.

### 11. What is the role of the hidden state in an RNN?

It summarizes information processed so far and carries that context to the next
time step.

### 12. Which metrics are suitable for sentiment analysis?

Accuracy, precision, recall, F1-score, ROC-AUC, and the confusion matrix are
useful. For imbalanced data, F1, precision-recall measures, and per-class
results are especially important.

### 13. How can overfitting be reduced in LSTM and GRU models?

Use more representative data, dropout, weight regularization, smaller models,
early stopping, shorter sequences where appropriate, and validation-based
hyperparameter selection.
