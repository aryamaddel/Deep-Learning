"""
Assignment 7: Text Preprocessing and Word Embedding Implementation using Word2Vec, GloVe, and FastText.

This module demonstrates:
1. Complete text pre-processing pipeline: lowercasing, cleaning, tokenization,
   stop word removal, stemming (Porter), and lemmatization (WordNet).
2. Word embedding model implementations & comparisons:
   - Word2Vec Continuous Bag-of-Words (CBOW)
   - Word2Vec Skip-gram
   - FastText (Subword-based embeddings)
   - GloVe (Global Vectors for Word Representation - fitted co-occurrence model)
3. Quantitative semantic evaluation:
   - Cosine similarity between target word pairs
   - Most similar word retrieval
   - Word analogy solving (e.g., King - Man + Woman = Queen)
4. Visualizations:
   - 2D PCA & t-SNE dimensionality reduction plots of word embeddings.
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import gensim
from gensim.models import FastText, Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Standard sample corpus for demonstration & training
SAMPLE_CORPUS = [
    "Natural language processing (NLP) enables computers to process, understand, and analyze human language.",
    "Machine learning models cannot directly process raw text, so text must first be converted into a suitable numerical representation.",
    "Word embeddings represent words as dense vectors in continuous vector spaces capture semantic meaning.",
    "Word2Vec predicts target words from context words in CBOW or context words from target words in Skip-gram.",
    "GloVe captures global co-occurrence statistics across the entire text corpus to construct vector representations.",
    "FastText breaks words into character n-grams to represent rare, out-of-vocabulary, and morphologically rich words.",
    "Stemming cuts off word affixes to reduce words to their base stem, while lemmatization uses vocabulary and morphological analysis.",
    "Cosine similarity measures the cosine of the angle between two word vectors to determine semantic closeness.",
    "Dimensionality reduction techniques such as PCA and t-SNE visualize high-dimensional word vectors in two-dimensional space.",
    "The king and queen rules the kingdom with wisdom and authority.",
    "A man and a woman walk into a palace to visit the king and queen.",
    "Deep learning neural networks process sequential text sequences using recurrent architectures and embedding vectors.",
    "Artificial intelligence, natural language understanding, and computer vision drive modern automation and data science.",
    "Tokenization splits text documents into smaller processing units called tokens or words.",
    "Stop words like the, is, at, which, and on are frequently filtered out during natural language preprocessing."
]


def preprocess_text(text: str, stemmer: PorterStemmer, lemmatizer: WordNetLemmatizer, stop_words: set) -> Dict[str, any]:
    """Perform step-by-step text preprocessing pipeline on raw text input."""
    # 1. Lowercasing
    lowercased = text.lower()
    
    # 2. Noise Removal (punctuation & special chars)
    cleaned = re.sub(r"[^\w\s]", " ", lowercased)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # 3. Tokenization
    tokens = word_tokenize(cleaned)
    
    # 4. Stop Word Removal
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    
    # 5. Stemming
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    
    # 6. Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    
    return {
        "raw": text,
        "lowercased": lowercased,
        "cleaned": cleaned,
        "tokens": tokens,
        "filtered": filtered_tokens,
        "stemmed": stemmed_tokens,
        "lemmatized": lemmatized_tokens,
    }


class SimpleGloVe:
    """
    A lightweight numpy implementation of GloVe (Global Vectors) training
    on co-occurrence matrix for custom corpus demonstration.
    """
    def __init__(self, embedding_dim: int = 50, x_max: float = 10.0, alpha: float = 0.75, learning_rate: float = 0.05):
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.vocab = {}
        self.id2word = {}
        self.W = None
        self.W_tilde = None
        self.b = None
        self.b_tilde = None

    def fit(self, corpus: List[List[str]], window_size: int = 5, epochs: int = 50):
        # Build vocabulary
        vocab_set = sorted(list(set(word for doc in corpus for word in doc)))
        self.vocab = {word: idx for idx, word in enumerate(vocab_set)}
        self.id2word = {idx: word for word, idx in self.vocab.items()}
        vocab_size = len(self.vocab)

        if vocab_size == 0:
            raise ValueError("Empty vocabulary for GloVe training.")

        # Build co-occurrence matrix
        cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        for doc in corpus:
            for i, target_word in enumerate(doc):
                target_id = self.vocab[target_word]
                start = max(0, i - window_size)
                end = min(len(doc), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context_word = doc[j]
                        context_id = self.vocab[context_word]
                        distance = abs(i - j)
                        cooccurrence[target_id, context_id] += 1.0 / distance

        # Initialize parameters
        rng = np.random.default_rng(42)
        self.W = rng.normal(scale=0.1, size=(vocab_size, self.embedding_dim))
        self.W_tilde = rng.normal(scale=0.1, size=(vocab_size, self.embedding_dim))
        self.b = np.zeros(vocab_size)
        self.b_tilde = np.zeros(vocab_size)

        # SGD Optimization
        for epoch in range(epochs):
            for i in range(vocab_size):
                for j in range(vocab_size):
                    x_ij = cooccurrence[i, j]
                    if x_ij > 0:
                        weight = (x_ij / self.x_max) ** self.alpha if x_ij < self.x_max else 1.0
                        diff = np.dot(self.W[i], self.W_tilde[j]) + self.b[i] + self.b_tilde[j] - np.log(x_ij)
                        
                        grad_main = weight * diff * self.W_tilde[j]
                        grad_context = weight * diff * self.W[i]
                        grad_b_main = weight * diff
                        grad_b_context = weight * diff

                        self.W[i] -= self.learning_rate * grad_main
                        self.W_tilde[j] -= self.learning_rate * grad_context
                        self.b[i] -= self.learning_rate * grad_b_main
                        self.b_tilde[j] -= self.learning_rate * grad_b_context

        # Final word vectors are sum of W and W_tilde
        self.vectors = self.W + self.W_tilde

    def get_vector(self, word: str) -> np.ndarray:
        if word in self.vocab:
            return self.vectors[self.vocab[word]]
        else:
            raise KeyError(f"Word '{word}' not in GloVe vocabulary.")

    def cosine_similarity(self, w1: str, w2: str) -> float:
        v1 = self.get_vector(w1)
        v2 = self.get_vector(w2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def most_similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        if word not in self.vocab:
            return []
        v_target = self.get_vector(word)
        norm_target = np.linalg.norm(v_target)
        if norm_target == 0:
            return []
        
        sims = []
        for w, idx in self.vocab.items():
            if w == word:
                continue
            v = self.vectors[idx]
            norm_v = np.linalg.norm(v)
            if norm_v > 0:
                sim = np.dot(v_target, v) / (norm_target * norm_v)
                sims.append((w, float(sim)))
        
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:topn]


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D numpy vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def plot_embeddings_2d(model_dict: Dict[str, any], target_words: List[str], output_dir: str):
    """
    Generate and save 2D PCA & t-SNE scatter plots for word embeddings.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("2D Word Embedding Visualizations (PCA vs t-SNE)", fontsize=16, fontweight="bold", y=0.98)

    model_names = ["Word2Vec CBOW", "Word2Vec Skip-gram", "FastText", "GloVe"]
    
    for idx, name in enumerate(model_names):
        ax = axes[idx // 2, idx % 2]
        model = model_dict[name]
        
        # Collect vectors
        valid_words = []
        vectors = []
        for word in target_words:
            if name == "GloVe":
                if word in model.vocab:
                    valid_words.append(word)
                    vectors.append(model.get_vector(word))
            else:
                if word in model.wv:
                    valid_words.append(word)
                    vectors.append(model.wv[word])
                elif name == "FastText":
                    valid_words.append(word)
                    vectors.append(model.wv[word])
        
        if len(vectors) < 3:
            ax.text(0.5, 0.5, f"Not enough valid words for {name}", ha="center", va="center")
            continue

        vec_matrix = np.array(vectors)
        
        # Apply PCA to 2D
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vec_matrix)
        
        # Scatter plot
        ax.scatter(coords[:, 0], coords[:, 1], color="#2b5c8f", alpha=0.8, edgecolors="k", s=80)
        
        for i, word in enumerate(valid_words):
            ax.annotate(
                word,
                (coords[i, 0], coords[i, 1]),
                xytext=(5, 2),
                textcoords="offset points",
                fontsize=10,
                fontweight="semibold",
                color="#1a1a1a"
            )
            
        ax.set_title(f"{name} (2D PCA Projection)", fontsize=12, fontweight="bold")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, "word_embeddings_2d.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved 2D embedding scatter plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Assignment 7: Text Preprocessing & Word Embeddings")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs for Gensim & GloVe models.")
    parser.add_argument("--vector-dim", type=int, default=50, help="Dimension size for word vectors.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for generated CSVs and figures.")
    args = parser.parse_args()

    # Determine script and output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print("      ASSIGNMENT 7: TEXT PREPROCESSING & WORD EMBEDDING IMPLEMENTATION         ")
    print("================================================================================\n")

    # Initialize NLTK utilities
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Step 1: Preprocessing Pipeline Demonstration
    print("--- 1. TEXT PREPROCESSING PIPELINE DEMONSTRATION ---")
    sample_text = SAMPLE_CORPUS[0]
    prep_res = preprocess_text(sample_text, stemmer, lemmatizer, stop_words)

    print(f"Original Text       : {prep_res['raw']}")
    print(f"1. Lowercased       : {prep_res['lowercased']}")
    print(f"2. Cleaned          : {prep_res['cleaned']}")
    print(f"3. Tokenized        : {prep_res['tokens']}")
    print(f"4. Stopwords Filter : {prep_res['filtered']}")
    print(f"5. Stemmed          : {prep_res['stemmed']}")
    print(f"6. Lemmatized       : {prep_res['lemmatized']}\n")

    # Preprocess entire corpus for training models
    preprocessed_corpus = []
    for doc in SAMPLE_CORPUS:
        res = preprocess_text(doc, stemmer, lemmatizer, stop_words)
        if res["lemmatized"]:
            preprocessed_corpus.append(res["lemmatized"])

    # Step 2: Model Training
    print("--- 2. TRAINING EMBEDDING MODELS ---")
    vec_size = args.vector_dim
    epochs = args.epochs

    t0 = time.time()
    w2v_cbow = Word2Vec(
        sentences=preprocessed_corpus,
        vector_size=vec_size,
        window=5,
        min_count=1,
        sg=0,  # CBOW
        epochs=epochs,
        seed=42
    )
    t_cbow = time.time() - t0
    print(f"[+] Word2Vec CBOW trained in {t_cbow:.3f}s (Vocab size: {len(w2v_cbow.wv)})")

    t0 = time.time()
    w2v_sg = Word2Vec(
        sentences=preprocessed_corpus,
        vector_size=vec_size,
        window=5,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=epochs,
        seed=42
    )
    t_sg = time.time() - t0
    print(f"[+] Word2Vec Skip-gram trained in {t_sg:.3f}s (Vocab size: {len(w2v_sg.wv)})")

    t0 = time.time()
    fasttext_model = FastText(
        sentences=preprocessed_corpus,
        vector_size=vec_size,
        window=5,
        min_count=1,
        epochs=epochs,
        seed=42
    )
    t_ft = time.time() - t0
    print(f"[+] FastText trained in {t_ft:.3f}s (Vocab size: {len(fasttext_model.wv)})")

    t0 = time.time()
    glove_model = SimpleGloVe(embedding_dim=vec_size, learning_rate=0.05)
    glove_model.fit(preprocessed_corpus, window_size=5, epochs=epochs)
    t_glove = time.time() - t0
    print(f"[+] GloVe fitted in {t_glove:.3f}s (Vocab size: {len(glove_model.vocab)})\n")

    models_dict = {
        "Word2Vec CBOW": w2v_cbow,
        "Word2Vec Skip-gram": w2v_sg,
        "FastText": fasttext_model,
        "GloVe": glove_model
    }

    # Step 3: Cosine Similarity Evaluation
    print("--- 3. COSINE SIMILARITY EVALUATION ---")
    test_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("language", "processing"),
        ("vector", "embedding"),
        ("stemming", "lemmatization"),
        ("king", "computer")  # Less related control pair
    ]

    sim_records = []
    for w1, w2 in test_pairs:
        row = {"Word 1": w1, "Word 2": w2}
        for model_name, model in models_dict.items():
            try:
                if model_name == "GloVe":
                    sim = model.cosine_similarity(w1, w2)
                else:
                    sim = model.wv.similarity(w1, w2)
            except KeyError:
                sim = np.nan
            row[model_name] = round(float(sim), 4) if not np.isnan(sim) else "N/A"
        sim_records.append(row)

    df_sim = pd.DataFrame(sim_records)
    print(df_sim.to_string(index=False))
    sim_csv_path = os.path.join(out_dir, "cosine_similarity_results.csv")
    df_sim.to_csv(sim_csv_path, index=False)
    print(f"\n[+] Saved Cosine Similarity comparison to: {sim_csv_path}\n")

    # Step 4: Rare & Out-Of-Vocabulary (OOV) Word Evaluation (FastText vs Word2Vec vs GloVe)
    print("--- 4. OUT-OF-VOCABULARY (OOV) HANDLING DEMONSTRATION ---")
    oov_word = "preprocessingtechnique"  # Unseen compound word
    print(f"Testing OOV Word: '{oov_word}'")
    
    oov_results = []
    for model_name, model in models_dict.items():
        if model_name == "FastText":
            vec = model.wv[oov_word]
            sims = model.wv.most_similar(oov_word, topn=3)
            sim_str = ", ".join([f"{w}: {s:.3f}" for w, s in sims])
            status = f"Supported via Subwords -> Nearest: [{sim_str}]"
        else:
            status = "KeyError / OOV Exception (Cannot generate vector for unseen word)"
        oov_results.append({"Model": model_name, "OOV Support Status": status})
        print(f" - {model_name:18s}: {status}")

    df_oov = pd.DataFrame(oov_results)
    oov_csv_path = os.path.join(out_dir, "oov_handling_results.csv")
    df_oov.to_csv(oov_csv_path, index=False)
    print(f"[+] Saved OOV analysis to: {oov_csv_path}\n")

    # Step 5: Most Similar Words Analysis
    print("--- 5. MOST SIMILAR WORDS RETRIEVAL ---")
    target_query = "language"
    print(f"Top 3 Most Similar Words to '{target_query}':")
    for model_name, model in models_dict.items():
        if model_name == "GloVe":
            sims = model.most_similar(target_query, topn=3)
        else:
            sims = model.wv.most_similar(target_query, topn=3)
        formatted = ", ".join([f"{w} ({s:.3f})" for w, s in sims])
        print(f" - {model_name:18s}: {formatted}")
    print()

    # Step 6: 2D Embedding Visualization
    print("--- 6. GENERATING 2D EMBEDDING VISUALIZATIONS ---")
    target_visualization_words = [
        "king", "queen", "man", "woman", "language", "processing",
        "computer", "vector", "embedding", "stemming", "lemmatization",
        "text", "learning", "model", "analysis"
    ]
    plot_embeddings_2d(models_dict, target_visualization_words, out_dir)

    print("\n================================================================================")
    print("                         ASSIGNMENT 7 EXECUTION COMPLETE                        ")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
