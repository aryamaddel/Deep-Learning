# Assignment 12 — Chatbot / Question-Answering System Using Large Language Models

## Title

Development of a chatbot or question-answering system using Large Language Models (LLMs).

## Aim

- To study the architecture and working principles of an LLM.
- To understand pretraining and fine-tuning.
- To build a retrieval-grounded question-answering chatbot on top of an LLM API.
- To compare grounded and ungrounded answers, including behavior on questions the knowledge base cannot answer.

## Theory

### LLM

A Large Language Model is a neural network, almost always a **transformer**,
trained on very large text corpora to model the probability of the next token
given the preceding tokens. With billions of parameters, it learns grammar,
facts, and reasoning patterns from data, and it can follow instructions
written in natural language.

Core components:

1. **Tokenizer:** splits text into subword tokens (BPE, WordPiece) and maps
   each to an integer id.
2. **Embedding layer:** maps each token id to a dense vector. Positional
   information (sinusoidal, learned, or rotary) is added because attention
   itself is order-agnostic.
3. **Transformer blocks (stacked N times):** each has a (masked)
   multi-head self-attention layer and a position-wise feed-forward network,
   joined by residual connections and layer normalization.
4. **Output head:** a linear layer plus softmax gives a probability
   distribution over the vocabulary for the next token.

Common architectures: **decoder-only** (GPT, Llama; generate text),
**encoder-only** (BERT; understand text), and **encoder-decoder** (T5;
translation, summarization). Chatbots are normally decoder-only models.

**Working principle (generation):** the prompt is tokenized and embedded; the
transformer produces a distribution for the next token; a decoding rule
(greedy, or sampling with temperature/top-p) picks one token; the token is
appended and the process repeats until an end token or length limit. The model
only sees a limited number of tokens, its **context window**.

### Pretraining and Finetuning

- **Pretraining:** self-supervised training on a huge unlabeled corpus.
  Decoder-only models minimize cross-entropy for next-token prediction;
  BERT-style models predict masked tokens. This is the most expensive stage and
  produces a general-purpose *base model*.
- **Fine-tuning:** adapts the base model with additional training.
  - *Supervised fine-tuning (SFT / instruction tuning)* on prompt-response
    pairs teaches the model to follow instructions and chat.
  - *RLHF / preference optimization* aligns outputs with human preferences
    (helpfulness, safety).
  - *Parameter-efficient fine-tuning (LoRA, adapters)* trains only a small set
    of added weights, cutting memory and cost.
- **Prompting / RAG** needs no weight updates: knowledge is supplied at
  inference time in the prompt.

### Steps Involved in LLM Model Design

1. **Define the objective** (general chatbot, domain QA, summarizer) and
   constraints (latency, cost, privacy).
2. **Collect and clean data:** deduplicate, filter low-quality or unsafe text,
   remove personal data.
3. **Tokenizer design:** train BPE/WordPiece, fix vocabulary size.
4. **Choose the architecture:** number of layers, hidden size, attention
   heads, context length, positional encoding.
5. **Pretrain:** next-token prediction using AdamW, learning-rate warm-up and
   decay, mixed precision, and distributed training.
6. **Fine-tune and align:** SFT on instructions, then RLHF or preference tuning.
7. **Evaluate:** benchmarks, task accuracy, human review, safety and bias tests.
8. **Deploy and monitor:** quantization, serving, guardrails, retrieval (RAG),
   logging, and feedback loops.

### Question-Answering System Design Used Here

Training an LLM from scratch is impractical, so this assignment uses a
pretrained LLM through an API and adds **retrieval-augmented generation (RAG)**:

```
question -> TF-IDF retriever -> top-k passages -> prompt (context + question) -> LLM -> answer
```

The system prompt instructs the model to answer using only the retrieved
context and to reply "I don't know based on the provided context." when the
context lacks the answer. This grounds the response and reduces hallucination,
though it does not eliminate it.

## Experiment Design

### Knowledge base and questions

The script contains a 12-passage knowledge base on LLM concepts (attention,
tokenization, pretraining, fine-tuning, RAG, embeddings, decoding, and so on)
and 12 test questions: 10 answerable from the passages and 2 deliberately
unanswerable (e.g. the parameter count of a specific model). No dataset
download is needed.

### Conditions

| Condition | Description |
|---|---|
| `no_retrieval` | The question is sent to the LLM alone (relies on memorized knowledge). |
| `rag` | The top-2 TF-IDF passages are included and the model must answer from them. |

### Settings and measures

| Setting | Configuration |
|---|---:|
| Retriever | TF-IDF (unigrams + bigrams), cosine similarity |
| Passages per question | 2 (`--top-k`) |
| API calls | 24 (12 per condition) |
| Temperature | 0 by default; configurable |
| API format | OpenAI-compatible Chat Completions |
| Accuracy | Answerable: all expected keywords present. Unanswerable: model abstains. |
| Additional metrics | Abstain rate on unanswerable questions, mean latency |

Keyword matching is a crude automatic proxy for answer quality; read the saved
answers in `qa_results.csv` as well. The question set is small and
illustrative, not a benchmark.

### Run the experiment

Set the credentials in the environment (never commit the key):

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "kimi-k3"
$env:OPENAI_BASE_URL = "https://code.in2peta.com"
```

Evaluate both conditions:

```powershell
uv run python assignment12/llm_chatbot.py
```

Chat interactively with the grounded bot (prints the source passages):

```powershell
uv run python assignment12/llm_chatbot.py --chat
```

Inspect the prompts without any API call:

```powershell
uv run python assignment12/llm_chatbot.py --dry-run
```

### Generated outputs

- `outputs/qa_results.csv`: per-question answers, correctness, retrieval score, and latency.
- `outputs/qa_summary.json`: accuracy, abstain rate, and mean latency per condition.

### Observed results

Not yet recorded: this experiment requires a live API key. After running it,
report the model, endpoint, date, temperature, and the values from
`outputs/qa_summary.json` here. Results should not be invented from the dry run.

## Conclusion

An LLM is a transformer trained by next-token prediction on large corpora
(pretraining) and then adapted to follow instructions and human preferences
(fine-tuning). A practical question-answering chatbot can be built without
training a model by combining a pretrained LLM with retrieval, so answers are
grounded in supplied passages and the system can decline when the evidence is
missing. The experiment compares ungrounded and retrieval-grounded answering
on answerable and unanswerable questions. Conclusions should come from the
recorded run, and the small question set and keyword scoring limit how far
they generalize.

## FAQs

### 1. How to update parameters in LLM?

Parameters are updated by gradient-based training. A forward pass computes the
loss (cross-entropy between predicted and actual next token); backpropagation
computes the gradient of the loss with respect to every weight; and an
optimizer such as AdamW applies the update,
`w <- w - lr * optimizer_step(grad)`, usually with learning-rate warm-up and
decay, gradient clipping, and mixed precision. This happens during
pretraining and fine-tuning. Parameter-efficient methods (LoRA, adapters)
freeze the original weights and update only small added matrices. At inference
time, parameters are not updated; knowledge is changed through the prompt or
retrieval instead.

### 2. What is hallucination.

Hallucination is when an LLM produces fluent, confident text that is false,
fabricated, or not supported by the supplied source. It occurs because the
model optimizes for plausible next tokens, not verified truth. Mitigations
include retrieval-augmented generation, instructing the model to say "I don't
know", lower temperature, citations, fine-tuning on factual data, and human or
automated verification. They reduce the problem but do not eliminate it.

### 3. What is the attention mechanism? Explain self-attention and multi-head attention.

Attention lets a model weigh which parts of the input matter most when
producing each output. **Self-attention** applies this within one sequence:
every token is projected into a query (Q), key (K), and value (V) vector, and

`Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`

The dot product of a token's query with all keys gives similarity scores;
scaling by `sqrt(d_k)` keeps them stable; softmax turns them into weights; and
the output is the weighted sum of the values. Each token thereby gathers
information from all relevant tokens. Decoder models use a causal mask so a
token cannot see future tokens. **Multi-head attention** runs `h` attention
operations in parallel with different learned projections, then concatenates
and linearly projects the results. Different heads can capture different
relationships (syntax, coreference, nearby versus distant context).

### 4. What is tokenization in LLMs?

Tokenization converts raw text into a sequence of tokens, the units the model
processes, and maps each to an integer id in a fixed vocabulary. LLMs use
subword schemes such as BPE, WordPiece, or SentencePiece: common words stay
whole while rare words break into pieces, so the vocabulary stays manageable
and unseen words can still be represented. Token counts determine context
usage, speed, and API cost, and the tokenizer used for training must also be
used at inference.

### 5. Explain word embeddings vs contextual embeddings.

A **word embedding** (Word2Vec, GloVe) assigns each word one fixed vector
regardless of context, so "bank" has a single representation for both a river
bank and a financial bank. A **contextual embedding** (BERT, GPT) is computed
by the whole network from the surrounding sentence, so the same word gets
different vectors in different contexts and polysemy is handled. Static
embeddings are small and cheap; contextual embeddings are richer and are what
LLMs use internally.
