# Assignment 7 — Text Pre-processing and Word Embedding Implementation

## Title

Text pre-processing and word embedding implementation using Word2Vec, GloVe, or FastText.

## Aim

- To implement and demonstrate text preprocessing techniques (lowercasing, cleaning, tokenization, stop word removal, stemming, and lemmatization).
- To construct dense vector representations of text using Word2Vec (CBOW and Skip-gram), GloVe, and FastText.
- To compare word vector similarities using Cosine Similarity.
- To evaluate Out-Of-Vocabulary (OOV) word handling across different embedding models.
- To project and visualize high-dimensional word vectors in 2D space using Principal Component Analysis (PCA).

## Theory

### Natural Language Processing (NLP) & Text Preprocessing
Natural Language Processing (NLP) enables computers to process, understand, and extract meaning from human language. Because machine learning and deep learning algorithms operate on numerical tensors rather than raw strings, unstructured text must undergo preprocessing before being vectorized.

1. **Lowercasing:** Normalizes characters to lowercase so words like `"Text"` and `"text"` map to the same representation.
2. **Noise Removal:** Eliminates punctuation, special characters, and formatting clutter that do not contribute to semantic meaning.
3. **Tokenization:** Breaks continuous text sequences into atomic processing units called tokens (e.g., words or subwords).
4. **Stop Word Removal:** Filters out high-frequency functional words (e.g., `"the"`, `"is"`, `"at"`) that carry little domain-specific semantic signal.
5. **Stemming:** Heuristically chops off prefixes/suffixes to reduce words to their base stem (e.g., `"processing"` $\rightarrow$ `"process"` using Porter Stemmer). It is fast but can produce non-dictionary words (over-stemming).
6. **Lemmatization:** Uses vocabulary lookup and morphological rules to reduce words to their valid dictionary root lemma (e.g., `"computers"` $\rightarrow$ `"computer"` using WordNet Lemmatizer).

---

### Word Embeddings

Word embeddings map discrete text tokens to dense, continuous numerical vectors $\mathbb{R}^d$ (typically $d \in [50, 300]$). Unlike sparse one-hot encoding, dense word embeddings capture semantic relationships and contextual similarity: words appearing in similar contexts are positioned close to each other in the vector space.

```
       One-Hot Encoding                     Dense Embedding Vector
"king"   -> [1, 0, 0, 0, 0, ...]    -> [ 0.42, -0.11,  0.89,  0.25, ...]
"queen"  -> [0, 1, 0, 0, 0, ...]    -> [ 0.45, -0.09,  0.87,  0.22, ...]
```

---

### Word2Vec Architecture (CBOW vs Skip-gram)

Introduced by Mikolov et al. (2013), Word2Vec uses a shallow two-layer neural network to learn word representations.

* **Continuous Bag-of-Words (CBOW):** Predicts a target center word given its surrounding context words $w_{t-k}, \dots, w_{t+k}$. CBOW averages context representations and trains faster on large corpora.
* **Skip-gram:** Predicts surrounding context words $w_{t-k}, \dots, w_{t+k}$ given a single center word $w_t$. Skip-gram gives more weight to infrequent words and performs exceptionally well on small/medium domain corpora.

$$\text{CBOW: } P(w_t \mid w_{t-k}, \dots, w_{t+k}) \quad \quad \text{Skip-gram: } P(w_{t-k}, \dots, w_{t+k} \mid w_t)$$

---

### GloVe (Global Vectors for Word Representation)

Developed by Pennington et al. (2014) at Stanford, GloVe combines the advantages of global matrix factorization and local context window methods. Rather than scanning local context windows sequentially, GloVe constructs a global word co-occurrence matrix $X$, where $X_{ij}$ counts how often word $i$ appears in the context of word $j$.

The model minimizes a weighted least-squares objective function:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

where $f(X_{ij}) = \min \left(1, \left(\frac{X_{ij}}{x_{\max}}\right)^\alpha \right)$ prevents frequent co-occurrences from dominating learning.

---

### FastText

Developed by Facebook AI Research (Bojanowski et al., 2017), FastText extends Skip-gram by treating each word as a bag of character $n$-grams. For instance, with $n=3$, the word `"where"` is represented by `<wh`, `whe`, `her`, `ere`, `res>`, and the special boundary token `<where>`.

The vector representation of word $w$ is the sum of its character $n$-gram vectors:

$$v_w = \sum_{g \in G_w} z_g$$

**Key Advantage:** FastText can compute meaningful vectors for **Out-Of-Vocabulary (OOV)**, misspelled, or rare words by aggregating vectors of its constituent character $n$-grams.

---

## Experiment Design

### Text Preprocessing & Embedding Pipeline Setup

The experiment tests text preprocessing and trains four distinct embedding models on a domain corpus:

| Setting | Configuration / Hyperparameter |
|---|---|
| Text Preprocessing | Lowercasing $\rightarrow$ Noise Removal $\rightarrow$ Tokenization $\rightarrow$ Stop Word Filter $\rightarrow$ Stemming/Lemmatization |
| Vector Dimension | $d = 50$ |
| Context Window Size | 5 |
| Minimum Count | 1 |
| Models Implemented | Word2Vec CBOW, Word2Vec Skip-gram, FastText, GloVe |
| Optimization / Epochs | 100 Epochs |
| Evaluation Metrics | Cosine Similarity, OOV Support, Top-K Similarity, 2D PCA Projections |

---

### Execution

Run the script from the repository root using `uv`:

```powershell
uv run assignment7/word_embeddings.py
```

### Generated Outputs

- `outputs/cosine_similarity_results.csv`: Cosine similarity scores across word pairs for all 4 models.
- `outputs/oov_handling_results.csv`: Out-of-vocabulary handling status and nearest word retrieval for unseen tokens.
- `outputs/word_embeddings_2d.png`: 2D PCA projection scatter plot visualizing semantic spatial clustering.

---

## Observed Results

### 1. Step-by-Step Preprocessing Demonstration

```
Original Text       : Natural language processing (NLP) enables computers to process, understand, and analyze human language.
1. Lowercased       : natural language processing (nlp) enables computers to process, understand, and analyze human language.
2. Cleaned          : natural language processing nlp enables computers to process understand and analyze human language
3. Tokenized        : ['natural', 'language', 'processing', 'nlp', 'enables', 'computers', 'to', 'process', 'understand', 'and', 'analyze', 'human', 'language']
4. Stopwords Filter : ['natural', 'language', 'processing', 'nlp', 'enables', 'computers', 'process', 'understand', 'analyze', 'human', 'language']
5. Stemmed          : ['natur', 'languag', 'process', 'nlp', 'enabl', 'comput', 'process', 'understand', 'analyz', 'human', 'languag']
6. Lemmatized       : ['natural', 'language', 'processing', 'nlp', 'enables', 'computer', 'process', 'understand', 'analyze', 'human', 'language']
```

---

### 2. Cosine Similarity Comparison Across Models

The quantitative evaluation on the target word pairs produced the following cosine similarity scores:

| Word 1 | Word 2 | Word2Vec CBOW | Word2Vec Skip-gram | FastText | GloVe |
|---|---|---:|---:|---:|---:|
| `king` | `queen` | 0.5662 | 0.9903 | 0.9535 | 0.8950 |
| `man` | `woman` | 0.3194 | 0.9834 | 0.9391 | 0.7056 |
| `language` | `processing` | 0.7395 | 0.9938 | 0.9689 | 0.3410 |
| `vector` | `embedding` | 0.6923 | 0.9936 | 0.9646 | 0.3769 |
| `stemming` | `lemmatization` | 0.7574 | 0.9948 | 0.9811 | 0.2067 |
| `king` | `computer` *(unrelated control)* | 0.6790 | 0.9920 | 0.9785 | **-0.0310** |

---

### 3. Out-Of-Vocabulary (OOV) Handling Analysis

When tested with an unseen compound word (`"preprocessingtechnique"`):

| Model | OOV Handling Result |
|---|---|
| **Word2Vec CBOW** | `KeyError` (Cannot generate vector for unseen token) |
| **Word2Vec Skip-gram** | `KeyError` (Cannot generate vector for unseen token) |
| **GloVe** | `KeyError` (Cannot generate vector for unseen token) |
| **FastText** | **Supported via Subwords** $\rightarrow$ Nearest: `preprocessing` ($0.997$), `processing` ($0.995$), `representation` ($0.993$) |

---

### 4. 2D Embedding Scatter Plot

![2D Word Embeddings Scatter Plot](outputs/word_embeddings_2d.png)

---

## Conclusion

1. **Text Preprocessing Efficiency:** Preprocessing successfully reduced noisy text to clean, normalized tokens. Lemmatization proved superior to stemming for downstream tasks as it preserves grammatically valid roots (`"computer"` vs `"comput"`).
2. **Embedding Comparison:**
   - **Word2Vec Skip-gram** and **FastText** captured high semantic similarity across domain word pairs.
   - **GloVe** effectively discriminated semantically distinct words, assigning a negative cosine similarity ($-0.0310$) to the unrelated pair (`king`, `computer`), whereas local context models exhibited higher baseline similarities due to corpus scale.
3. **OOV Robustness:** **FastText** demonstrated superior capability by generating meaningful vector representations for out-of-vocabulary words (`"preprocessingtechnique"`) using subword character $n$-grams, whereas Word2Vec and GloVe failed with lookup errors.
4. **Spatial Clustering:** 2D PCA projections confirmed that semantically associated terms (e.g., `stemming` & `lemmatization`, `vector` & `embedding`) form distinct spatial clusters in vector space.

---

## FAQs

### 1. Why is text preprocessing required before training an NLP model?
Raw text contains noise, inconsistent capitalization, punctuation, and uninformative high-frequency words. Preprocessing cleans, standardizes, and reduces the vocabulary space, ensuring models learn meaningful semantic patterns rather than noise.

### 2. What is tokenization?
Tokenization is the process of breaking a continuous text stream or document into smaller discrete units (tokens), such as words, subwords, or characters, which can be mapped to numerical IDs.

### 3. What is the difference between stemming and lemmatization?
Stemming uses crude heuristic rules to trim word endings (often resulting in non-words like `"comput"`), whereas lemmatization uses morphological analysis and dictionary lookups to return the valid root form (lemma, like `"computer"`).

### 4. What are stop words?
Stop words are common language words (e.g., `"and"`, `"the"`, `"in"`) that appear frequently across documents but carry little domain-specific semantic value. Filtering them reduces noise and dimensionality.

### 5. Why are words represented as vectors?
Vectors map non-numerical text tokens into continuous mathematical vector spaces $\mathbb{R}^d$, allowing mathematical operations (like distance, cosine similarity, and matrix multiplication) to compute semantic relationships.

### 6. What is the difference between CBOW and Skip-gram?
CBOW predicts the target center word given its surrounding context words (faster, better for frequent words). Skip-gram predicts context words given a center target word (slower, better for rare words and small datasets).

### 7. How does GloVe differ from Word2Vec?
Word2Vec trains via predictive neural windows scanning text locally. GloVe leverages global co-occurrence statistics across the entire corpus by factorizing a global log-co-occurrence matrix.

### 8. How does FastText handle rare or unseen words?
FastText represents words as sums of character $n$-gram vectors. For unseen/OOV words, it builds a vector by summing vectors of its constituent $n$-grams.

### 9. What is cosine similarity between word vectors?
Cosine similarity measures the cosine of the angle between two multi-dimensional vectors:
$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
Values close to $1.0$ indicate high semantic similarity regardless of vector magnitude.

### 10. How can similar words be identified using word embeddings?
By computing the cosine similarity between a target word vector and all vectors in the vocabulary, ranking them in descending order to retrieve the nearest neighbors.

### 11. How can word embeddings be visualized in two-dimensional space?
High-dimensional embedding vectors ($d \ge 50$) can be projected onto a 2D plane using dimensionality reduction techniques such as Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE).
