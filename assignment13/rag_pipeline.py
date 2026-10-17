"""Assignment 13: a retrieval-augmented generation pipeline with open-source models.

Run from the repository root:
    uv run python assignment13/rag_pipeline.py

The script chunks a small fictional lab handbook, retrieves passages with
TF-IDF and with a MiniLM bi-encoder, and generates answers with Flan-T5.
The same questions are also asked with no retrieved context. The first run
downloads both models from the Hugging Face Hub.
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

SEED = 42
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-small"
MODES = ("closed_book", "tfidf", "dense")
MODE_LABELS = {
    "closed_book": "Closed book",
    "tfidf": "TF-IDF RAG",
    "dense": "Dense RAG",
}

# Fictional handbook. Facts are invented so a closed-book model cannot recall
# them from pretraining. The answer sentence is first; later sentences are
# distractors that should not outrank it.
DOCUMENTS = [
    {
        "doc_id": "founding",
        "title": "Department founding",
        "text": (
            "The Horizon Institute Department of Neural Systems was founded in March 2014 by Dr. Leena Voss "
            "to study learning algorithms for small devices. The group occupies floors 3 and 4 of Building C "
            "on the riverside campus. Orientation for newcomers is held in the atrium, and the original "
            "laboratory notebook is stored in archive cabinet Voss-01. Visitors sign in at the ground-floor "
            "desk. Floor maps are posted beside the elevator. Quiet hours apply after 20:00 on both floors. "
            "Seminar posters must be removed within a week of the event."
        ),
    },
    {
        "doc_id": "aurora",
        "title": "Aurora GPU cluster",
        "text": (
            "The department GPU cluster, named Aurora, contains 16 NVIDIA A100 80GB cards and is housed in "
            "Building C, room 214. Students may reserve the machine for at most 8 hours per week through the "
            "portal at labs.horizon.edu/aurora, and a booking must be cancelled at least 2 hours in advance. "
            "Jobs left running past a reservation are stopped by the scheduler. Scratch space is mounted at "
            "/scratch/aurora and is wiped every Sunday at 02:00. Home directories are stored elsewhere and "
            "are not wiped. Account questions go to the systems administrator."
        ),
    },
    {
        "doc_id": "course",
        "title": "Course NS-410",
        "text": (
            "The required textbook for course NS-410 is Search and Memory by Idris Hale. The class meets "
            "on Tuesdays and Thursdays at 10:15 in lecture hall C-310. Grading uses 40 percent for the "
            "project, 30 percent for the midterm, and 30 percent for the final. Students bring the book "
            "to the first Thursday meeting. The project is due on the last Friday of the term. Late "
            "submissions lose 5 points per day. Slides are posted the evening after each lecture."
        ),
    },
    {
        "doc_id": "library",
        "title": "Lab library loans",
        "text": (
            "The lab library lends embedding model weights for 14 days, and the librarian on duty during "
            "the fall term is Mira Chen at desk L2. A student may borrow at most 2 checkpoints at a time. "
            "Overdue checkpoints incur a fine of 2 lab credits per day. Weights must be returned on a USB "
            "drive labeled with the student id. A network copy does not count as a return. Desk L2 is "
            "closed on public holidays. Renewal requests are filed before noon."
        ),
    },
    {
        "doc_id": "safety",
        "title": "Sensor lab safety",
        "text": (
            "The emergency cutoff switch for the Aurora cluster is the red panel immediately left of the "
            "door in room 214. Pressing the panel stops all jobs and powers down the cards. Liquid nitrogen "
            "for the sensor lab is stored in cabinet S-7, and only students who completed the cryogen "
            "briefing may open it. The after-hours emergency contact is telephone extension 4419. Daytime "
            "incidents are reported to the front desk of Building C. Spill kits hang beside cabinet S-7."
        ),
    },
    {
        "doc_id": "seminar",
        "title": "Friday Signals seminar",
        "text": (
            "The spring 2026 opening talk of the Friday Signals seminar is by Dr. Omar Rahman on sparse "
            "retrieval. The series starts at 16:00 in room C-118. Attendance counts for 5 lab credits when "
            "the student submits a 200-word note by the following Monday at 09:00. Notes are uploaded to "
            "the course box labeled Signals-Notes. Food is not allowed in the room during a talk. The "
            "series coordinator is Nia Kapoor. Seats are first come, first served."
        ),
    },
    {
        "doc_id": "nimbusqa",
        "title": "NimbusQA benchmark",
        "text": (
            "The internal benchmark NimbusQA contains 1200 questions, and the best dense retriever on it "
            "reaches a recall at 5 of 0.81. The questions were written over 80 lab manuals and are stored "
            "on the shared drive at /data/nimbusqa. The set is not released outside the department. The "
            "recall figure was measured in January 2026 by the evaluation group. Each item has one supporting "
            "manual section. Students may not copy benchmark text into a public repository."
        ),
    },
    {
        "doc_id": "leadership",
        "title": "Department leadership",
        "text": (
            "Professor Hana Ibarra leads the department as its head, and students may visit her on Wednesday "
            "from 14:00 to 16:00 in office C-402. Appointments are not required during those hours. The "
            "systems administrator is Pavel Ortiz. He handles account and storage requests and can be reached "
            "at pav.ortiz@horizon.edu. Requests submitted after 15:00 are processed on the next weekday. "
            "The office door is kept open during visiting hours."
        ),
    },
    {
        "doc_id": "budget",
        "title": "2026 equipment budget",
        "text": (
            "The 2026 equipment budget reserves 12000 lab credits for annotation of NimbusQA. The full "
            "budget is 74000 lab credits, of which 45000 are reserved for replacing two A100 cards. The "
            "remaining 17000 lab credits are discretionary. Any discretionary purchase above 500 lab credits "
            "requires written approval from the department head. Receipts are filed with the department "
            "office within 10 days of a purchase. Quotes older than 30 days must be refreshed."
        ),
    },
    {
        "doc_id": "assistant",
        "title": "Assistant refusal rule",
        "text": (
            "Lab note NS-RAG-7 requires the department assistant to refuse an answer when the top cosine "
            "similarity is below 0.35. Passages are split into chunks of about 40 words with 10 words of "
            "overlap, and the assistant retrieves 3 chunks for each question. A refusal uses the exact "
            "sentence: I cannot answer from the lab notes. The assistant cites the chunk title in brackets. "
            "The note was approved by the faculty on 12 January 2026. The threshold is reviewed each term."
        ),
    },
]

# (question, gold document id or None, keywords that a correct answer should contain)
# One factual span per question. Flan-T5-small copies a short answer more reliably
# than a two-part question, so the score measures grounding rather than multi-hop skill.
QUESTIONS = [
    ("Who founded the Department of Neural Systems?", "founding", ["leena voss"]),
    ("How many NVIDIA A100 cards does the Aurora cluster contain?", "aurora", ["16"]),
    ("What textbook is required for course NS-410?", "course", ["search and memory"]),
    ("Who is the fall-term librarian?", "library", ["mira chen"]),
    ("What color is the emergency cutoff panel for the Aurora cluster?", "safety", ["red"]),
    ("Who gives the spring 2026 opening talk of Friday Signals?", "seminar", ["omar rahman"]),
    ("What recall at 5 did the best dense retriever reach on NimbusQA?", "nimbusqa", ["0.81"]),
    ("On which weekday can students meet the department chair?", "leadership", ["wednesday"]),
    ("How many lab credits in the 2026 budget are reserved for annotation of NimbusQA?", "budget", ["12000"]),
    ("Below what cosine similarity must the department assistant refuse to answer?", "assistant", ["0.35"]),
    ("What is the Wi-Fi password for Building C?", None, []),
    ("How many students enrolled in NS-410 during 2025?", None, []),
]


def build_chunks(documents: list[dict[str, str]], chunk_words: int, overlap_words: int) -> list[dict[str, str]]:
    """Split each document into overlapping word windows."""
    if overlap_words >= chunk_words:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[dict[str, str]] = []
    for document in documents:
        words = document["text"].split()
        start = 0
        part = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}-{part}",
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "text": " ".join(words[start:end]),
                }
            )
            part += 1
            if end == len(words):
                break
            start = end - overlap_words
    return chunks


def normalize(text: str) -> str:
    return text.lower().replace(",", "")


def has_term(text: str, term: str) -> bool:
    """Match a keyword, treating numeric terms as whole numbers and percent words as '%' too."""
    candidates = [term]
    if term.endswith(" percent"):
        candidates.append(term[: -len(" percent")] + "%")
    for candidate in candidates:
        if re.fullmatch(r"[\d.]+%?", candidate):
            pattern = rf"(?<!\d){re.escape(candidate)}(?!\d)"
        else:
            pattern = rf"\b{re.escape(candidate)}\b"
        if re.search(pattern, text):
            return True
    return False


def keywords_present(text: str, keywords: list[str]) -> bool:
    normalized = normalize(text)
    return all(has_term(normalized, keyword) for keyword in keywords)


def abstained(answer: str) -> bool:
    lowered = normalize(answer)
    phrases = (
        "unknown",
        "unanswerable",
        "i cannot",
        "i can't",
        "don't know",
        "do not know",
        "not in the context",
        "cannot answer",
        "can't answer",
        "no information",
        "not provided",
        "not mentioned",
    )
    return any(phrase in lowered for phrase in phrases)


def score_answer(answer: str, keywords: list[str], answerable: bool) -> bool:
    if not answerable:
        return abstained(answer)
    return keywords_present(answer, keywords)


def make_prompt(question: str, passages: list[str] | None) -> str:
    """Ask for a short span. Numbered citations make the small generator echo '[1]'."""
    if not passages:
        return (
            "Answer the question about the Horizon Institute Department of Neural Systems. "
            "If you do not know the answer, say unanswerable.\n"
            f"Question: {question}"
        )
    context = "\n\n".join(passages)
    return (
        f"{context}\n\n"
        "Answer the question using only the passage. "
        "If the passage does not contain the answer, say unanswerable.\n"
        f"Question: {question}"
    )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token states, ignoring padding, then L2-normalize each row."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


class DenseRetriever:
    """Bi-encoder retriever: MiniLM mean-pooled embeddings and cosine similarity."""

    def __init__(self, chunks: list[dict[str, str]], model_name: str, device: torch.device) -> None:
        self.chunks = chunks
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.matrix = self.encode([chunk["text"] for chunk in chunks])

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            pooled = mean_pool(hidden, encoded["attention_mask"])
        return pooled.cpu().numpy()

    def search(self, query: str, k: int) -> list[tuple[dict[str, str], float]]:
        scores = (self.matrix @ self.encode([query])[0]).ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[int(index)], float(scores[index])) for index in order]


class SparseRetriever:
    """TF-IDF retriever over the same chunks, scored by cosine similarity."""

    def __init__(self, chunks: list[dict[str, str]]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    def search(self, query: str, k: int) -> list[tuple[dict[str, str], float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[int(index)], float(scores[index])) for index in order]


class Generator:
    """Open-source sequence-to-sequence generator. Beam search keeps runs repeatable."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
        elapsed = time.perf_counter() - start
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return text, elapsed


def retrieval_stats(
    hits: list[tuple[dict[str, str], float]], gold_doc: str | None, keywords: list[str]
) -> tuple[str, str, str, str, str]:
    """Return retrieved labels, top score, hit flag, reciprocal rank, and context coverage."""
    labels = "|".join(f"{chunk['chunk_id']}:{score:.3f}" for chunk, score in hits)
    top_score = f"{hits[0][1]:.4f}" if hits else ""
    if gold_doc is None:
        return labels, top_score, "", "", ""
    rank = next((index for index, (chunk, _) in enumerate(hits, start=1) if chunk["doc_id"] == gold_doc), None)
    hit = "1" if rank is not None else "0"
    reciprocal = f"{1 / rank:.4f}" if rank is not None else "0"
    context = " ".join(chunk["text"] for chunk, _ in hits)
    covered = "1" if keywords_present(context, keywords) else "0"
    return labels, top_score, hit, reciprocal, covered


def run_evaluation(args: argparse.Namespace, chunks: list[dict[str, str]]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading embedding model {args.embed_model} on {device}...")
    dense = DenseRetriever(chunks, args.embed_model, device)
    sparse = SparseRetriever(chunks)
    print(f"Loading generator {args.generator_model}...")
    generator = Generator(args.generator_model, device)
    retrievers = {"tfidf": sparse, "dense": dense}

    records: list[dict[str, object]] = []
    for mode in MODES:
        for index, (question, gold_doc, keywords) in enumerate(QUESTIONS, start=1):
            answerable = gold_doc is not None
            hits: list[tuple[dict[str, str], float]] = []
            if mode == "closed_book":
                retrieved, top_score, hit, reciprocal, covered = "", "", "", "", ""
                passages = None
            else:
                hits = retrievers[mode].search(question, args.top_k)
                retrieved, top_score, hit, reciprocal, covered = retrieval_stats(hits, gold_doc, keywords)
                passages = [chunk["text"] for chunk, _ in hits]
            prompt = make_prompt(question, passages)
            answer, elapsed = generator.generate(prompt, args.max_new_tokens)
            correct = score_answer(answer, keywords, answerable)
            records.append(
                {
                    "mode": mode,
                    "question_id": index,
                    "question": question,
                    "answerable": int(answerable),
                    "gold_doc": gold_doc or "",
                    "retrieved": retrieved,
                    "top_score": top_score,
                    "retrieval_hit": hit,
                    "reciprocal_rank": reciprocal,
                    "context_has_answer": covered,
                    "correct": int(correct),
                    "latency_seconds": round(elapsed, 4),
                    "answer": answer.replace("\n", " "),
                }
            )
            status = "correct" if correct else "wrong"
            print(f"{mode:12s} Q{index:02d}: {status}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "rag_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = []
    for mode in MODES:
        subset = [row for row in records if row["mode"] == mode]
        answerable_rows = [row for row in subset if row["answerable"]]
        unanswerable_rows = [row for row in subset if not row["answerable"]]
        hit_rows = [row for row in answerable_rows if row["retrieval_hit"] != ""]
        summary.append(
            {
                "mode": mode,
                "questions": len(subset),
                "answer_accuracy": round(sum(int(row["correct"]) for row in subset) / len(subset), 4),
                "answerable_accuracy": round(
                    sum(int(row["correct"]) for row in answerable_rows) / len(answerable_rows), 4
                ),
                "abstain_rate_on_unanswerable": round(
                    sum(int(row["correct"]) for row in unanswerable_rows) / len(unanswerable_rows), 4
                ),
                "retrieval_hit_rate": (
                    round(sum(int(row["retrieval_hit"]) for row in hit_rows) / len(hit_rows), 4) if hit_rows else None
                ),
                "mean_reciprocal_rank": (
                    round(sum(float(row["reciprocal_rank"]) for row in hit_rows) / len(hit_rows), 4)
                    if hit_rows
                    else None
                ),
                "context_has_answer_rate": (
                    round(sum(int(row["context_has_answer"]) for row in hit_rows) / len(hit_rows), 4)
                    if hit_rows
                    else None
                ),
                "mean_latency_seconds": round(
                    sum(float(row["latency_seconds"]) for row in subset) / len(subset), 4
                ),
            }
        )
    summary_path = OUTPUT_DIR / "rag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plot(summary, OUTPUT_DIR / "rag_comparison.png")
    print(f"\nSummary ({args.generator_model}, embed={args.embed_model}, device={device}):")
    for row in summary:
        hit = "n/a" if row["retrieval_hit_rate"] is None else f"{row['retrieval_hit_rate']:.3f}"
        print(
            f"{row['mode']:12s} accuracy={row['answer_accuracy']:.3f} "
            f"answerable={row['answerable_accuracy']:.3f} "
            f"abstain={row['abstain_rate_on_unanswerable']:.3f} "
            f"hit={hit} mean_latency={row['mean_latency_seconds']:.3f}s"
        )
    print(f"Saved {results_path.name}, {summary_path.name}, and rag_comparison.png in {OUTPUT_DIR}")


def save_plot(summary: list[dict[str, object]], path: Path) -> None:
    """Answer accuracy for every condition, and retrieval quality for the RAG conditions."""
    labels = [MODE_LABELS[str(row["mode"])] for row in summary]
    accuracy = [float(row["answer_accuracy"]) for row in summary]
    rag_rows = [row for row in summary if row["retrieval_hit_rate"] is not None]
    rag_labels = [MODE_LABELS[str(row["mode"])] for row in rag_rows]
    positions = np.arange(len(rag_labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    axes[0].bar(labels, accuracy, color=["#7f7f7f", "#4c78a8", "#f58518"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Answer accuracy")
    axes[1].bar(
        positions - width / 2,
        [float(row["retrieval_hit_rate"]) for row in rag_rows],
        width,
        label="Gold document in top-k",
        color="#4c78a8",
    )
    axes[1].bar(
        positions + width / 2,
        [float(row["context_has_answer_rate"]) for row in rag_rows],
        width,
        label="Answer text in top-k",
        color="#54a24b",
    )
    axes[1].set_xticks(positions, rag_labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Retrieval quality")
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_chunks(chunks: list[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "chunks.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["chunk_id", "doc_id", "title", "words", "text"])
        writer.writeheader()
        for chunk in chunks:
            writer.writerow(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "words": len(chunk["text"].split()),
                    "text": chunk["text"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="Hugging Face bi-encoder id.")
    parser.add_argument("--generator-model", default=GENERATOR_MODEL, help="Hugging Face seq2seq generator id.")
    parser.add_argument("--chunk-words", type=int, default=40, help="Words per chunk.")
    parser.add_argument("--overlap-words", type=int, default=10, help="Overlapping words between chunks.")
    parser.add_argument("--top-k", type=int, default=2, help="Chunks retrieved per question.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Generation length cap.")
    args = parser.parse_args()
    if args.chunk_words < 8:
        parser.error("--chunk-words must be at least 8")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.overlap_words < 0 or args.overlap_words >= args.chunk_words:
        parser.error("--overlap-words must be >= 0 and smaller than --chunk-words")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    chunks = build_chunks(DOCUMENTS, args.chunk_words, args.overlap_words)
    save_chunks(chunks)
    print(f"Indexed {len(DOCUMENTS)} documents as {len(chunks)} chunks.")
    run_evaluation(args, chunks)


if __name__ == "__main__":
    main()
