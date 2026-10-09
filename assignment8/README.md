# Assignment 8 — Transformer and Self-Attention

## Title

Implementation of a Transformer encoder for NLP sentiment classification.

## Aim

- To study the main components of the Transformer architecture.
- To understand Query, Key, and Value vectors and scaled dot-product self-attention.
- To examine how token embeddings and positional information represent text.
- To implement a small Transformer encoder and apply it to sentiment classification.
- To evaluate the model with held-out data and standard classification metrics.

## Theory

Transformers are neural network architectures for sequence data. Unlike RNNs,
they do not process tokens one at a time. Self-attention lets each token combine
information from other tokens in the same input, while position information
preserves word order. This allows Transformer training to process sequence
positions in parallel and helps models represent long-range relationships.

### Main Components of a Transformer

1. **Tokenizer:** Splits text into tokens (often subwords) and converts them to
   integer IDs. A vocabulary maps each token to its ID; padding and unknown-token
   IDs support batching and unseen words.
2. **Token embeddings:** Map token IDs to trainable dense vectors.
3. **Positional information:** Encodes token order. It can be added to token
   embeddings using sinusoidal encodings or learned position embeddings.
4. **Self-attention:** Builds a contextual representation for each token by
   comparing its Query vector with the Key vectors of tokens in the sequence,
   then combining their Value vectors.
5. **Feed-forward network:** Applies the same small multilayer network to each
   token position after attention.
6. **Residual connections and layer normalization:** Stabilize optimization and
   help information and gradients flow through the network.
7. **Task head:** Converts the encoder output into task predictions. This
   experiment uses a two-class sentiment classifier.

### Self-Attention

For input representation matrix $X$, learned projection matrices form queries,
keys, and values:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V$$

Scaled dot-product attention is:

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The dot products measure how strongly each query relates to each key. Dividing
by $\sqrt{d_k}$ keeps scores at a useful scale before softmax turns them into
weights. Each output is a weighted sum of the Values. In **multi-head attention**,
several attention operations run in parallel, allowing the model to represent
different relationships between tokens.

### Positional Encoding

Self-attention alone does not encode token order: permuting input tokens also
permutes the outputs. A position vector is therefore added to each token
embedding. This assignment uses learned position embeddings. The original
Transformer used sinusoidal position encodings, which provide a deterministic
signal at each sequence position.

### Popular Pre-trained Transformer Models

- **BERT:** Bidirectional encoder model commonly used for classification,
  extraction, and language understanding.
- **DistilBERT:** A smaller, faster model distilled from BERT.
- **RoBERTa:** A BERT-style encoder trained with a revised training recipe.
- **GPT family:** Decoder-only models commonly used for text generation.
- **T5:** An encoder-decoder model that frames many NLP tasks as text-to-text.
- **BART:** An encoder-decoder model useful for generation and summarization.

These models are commonly adapted with task-specific fine-tuning. The code in
this assignment implements and trains a small Transformer from random
initialization; it does not download or fine-tune a pre-trained model.

### Transformer vs RNN and LSTM

An RNN reads tokens sequentially and carries a hidden state from one time step
to the next. An LSTM is a gated RNN with a cell state that helps preserve
information across longer sequences. Transformers use attention to connect
tokens directly and can process token positions in parallel during training.
They often need more memory for long inputs, and attention computation grows
approximately quadratically with sequence length. RNNs and LSTMs can be a
better fit for small or streaming sequence workloads.

## Experiment Design

### Task and data

The script classifies short movie-review sentences as positive or negative. It
uses a small balanced corpus embedded in the Python file, so the experiment is
offline and repeatable. A fixed random seed creates stratified train,
validation, and test splits. The validation set selects the checkpoint with
the lowest validation loss; the test set is evaluated after model selection.
Because the corpus is small and illustrative, results should not be treated as
a benchmark for real-world sentiment systems.

### Model and settings

| Setting | Configuration |
|---|---:|
| Encoder blocks | 1 |
| Hidden dimension | 48 |
| Attention heads | 4 |
| Feed-forward dimension | 96 |
| Maximum sequence length | 16 tokens |
| Optimizer | AdamW |
| Learning rate | 0.003 |
| Batch size | 8 |
| Epoch limit | 30 (early stopping after 5 non-improving validation epochs) |
| Random seed | 42 |
| Loss | Cross-entropy |

The classifier mean-pools the non-padding token representations and predicts a
positive or negative label. The script reports accuracy, macro precision,
macro recall, and macro F1, and saves a confusion matrix, loss plot, and one
held-out example's mean attention map. For imbalanced or high-stakes datasets,
per-class precision and recall and a representative external test set should
also be considered.

### Run the experiment

From the repository root:

```powershell
uv run assignment8/transformer_sentiment.py
```

For a brief two-epoch code check:

```powershell
uv run assignment8/transformer_sentiment.py --quick
```

The script uses packages already listed in the repository's `pyproject.toml`.
It requires no external dataset or pre-trained checkpoint download.

### Generated outputs

- `outputs/test_metrics.csv`: held-out test metrics and run information.
- `outputs/training_history.csv`: per-epoch training and validation metrics.
- `outputs/classification_report.txt`: per-class precision, recall, and F1.
- `outputs/confusion_matrix.png`: held-out predictions by true class.
- `outputs/learning_curves.png`: training and validation loss over epochs.
- `outputs/attention_heatmap.png`: mean attention weights across heads for one
  held-out review. Attention weights show token-to-token weighting; they are
  not, by themselves, a complete explanation of a prediction.

## Conclusion

The experiment demonstrates how token embeddings, learned positional
embeddings, self-attention, a feed-forward block, and a task head combine in a
Transformer encoder. The generated metrics indicate how this model performed
on the fixed held-out portion of the small review corpus. Since the corpus is
small and contains simple examples, the results illustrate the workflow rather
than establish general performance. A stronger study would use a larger,
representative dataset, multiple random seeds, and a comparison with a
pre-trained model such as DistilBERT. Report the observed test scores and
confusions from the generated files when submitting the experiment.

## FAQs

### 1. What is the self-attention mechanism?

Self-attention lets each token use information from other tokens in its input.
It computes query-key similarity scores, normalizes those scores, and combines
the corresponding value vectors into contextual token representations.

### 2. What are Query, Key, and Value in self-attention?

They are learned projections of token representations. A Query represents what
a token is looking for, Keys describe what each token can match, and Values
contain the information that is combined using the attention weights.

### 3. What is positional encoding?

It is information added to token representations to indicate token order.
Without positional information, self-attention has no built-in way to tell
which token came first. This implementation uses learned position embeddings.

### 4. How does a Transformer differ from an RNN?

An RNN processes a sequence one step at a time and carries a hidden state.
A Transformer uses attention to connect tokens directly and can process token
positions in parallel during training. Attention may use more memory on long
sequences.

### 5. How does a Transformer differ from an LSTM?

An LSTM is a gated recurrent network with a cell state and sequential time-step
updates. A Transformer uses self-attention and position information instead of
recurrent state updates. LSTMs can be useful for small or streaming sequences;
Transformers are effective when broad token-to-token context and parallel
training are valuable.

### 6. What is the purpose of tokenization in Transformer models?

Tokenization divides text into units such as subwords and maps each unit to an
integer ID. The model uses those IDs to look up embeddings. Subword tokenizers
can represent many rare words using familiar word pieces.

### 7. What is fine-tuning in Transformer-based models?

Fine-tuning starts with a pre-trained model and continues training some or all
of its parameters on labelled examples for a target task. It adapts general
language representations to that task, usually with a new task-specific output
head and a smaller learning rate.

### 8. What is the Hugging Face Transformers library?

It is an open-source Python library that provides model architectures,
pre-trained checkpoints, tokenizers, and utilities for training and inference
across many Transformer models. This assignment uses PyTorch directly to make
the encoder components visible and does not use the library API.

### 9. What NLP applications can be developed using Transformers?

Applications include text classification, sentiment analysis, named-entity
recognition, question answering, translation, summarization, information
extraction, conversational systems, and text generation.

### 10. What are the limitations of Transformer models?

They can require substantial training data, accelerator memory, and compute.
Standard full self-attention has quadratic time and memory cost in sequence
length. Models can also reproduce data biases, hallucinate in generative
settings, and be difficult to interpret. Performance depends on the data and
evaluation setting.

### 11. How can the performance of a Transformer-based NLP application be evaluated?

Choose metrics that match the task. For classification, use a held-out test
set and report accuracy, precision, recall, F1, and a confusion matrix. For
generation or extraction, use task-appropriate metrics and human review where
needed. Keep test data separate from training and model selection, and report
dataset size, class balance, and relevant error examples.
