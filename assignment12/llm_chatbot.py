"""Assignment 12: question-answering chatbot built on a Large Language Model.

Run from the repository root after setting OPENAI_API_KEY and OPENAI_MODEL:
    uv run python assignment12/llm_chatbot.py            # evaluate the chatbot
    uv run python assignment12/llm_chatbot.py --chat     # interactive chat

The chatbot retrieves the most relevant passages from a small knowledge base
(TF-IDF similarity) and asks an OpenAI-compatible Chat Completions endpoint to
answer using only those passages. The same questions are also sent without
retrieval so grounded and ungrounded answers can be compared.
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
API_BASE_URL = "https://code.in2peta.com"
NO_ANSWER = "I don't know based on the provided context."

# Small knowledge base: one short passage per topic from LLM architecture.
KNOWLEDGE_BASE = [
    "Transformer architecture: a transformer is built from stacked blocks that each contain "
    "a self-attention layer and a feed-forward network, connected with residual connections "
    "and layer normalization. It replaced recurrence, so all tokens in a sequence are processed in parallel.",
    "Self-attention: each token is projected into a query, a key, and a value vector. Attention "
    "weights are the softmax of scaled dot products between queries and keys, and each output is "
    "the weighted sum of the value vectors. Multi-head attention runs several attention heads in "
    "parallel so the model can attend to different relationships at once.",
    "Tokenization: text is split into tokens, usually subwords, using algorithms such as byte-pair "
    "encoding (BPE) or WordPiece. Each token maps to an integer id in a fixed vocabulary, which "
    "keeps the vocabulary small while handling rare and unseen words.",
    "Positional encoding: attention has no built-in notion of order, so position information is "
    "added to token embeddings. The original transformer used sinusoidal encodings; many modern "
    "models use learned or rotary position embeddings.",
    "Pretraining: an LLM is pretrained on a very large text corpus with a self-supervised objective. "
    "Decoder-only models such as GPT predict the next token; encoder models such as BERT predict "
    "masked tokens. Pretraining gives the model general language knowledge.",
    "Fine-tuning: a pretrained model is adapted to a task or behavior using a smaller labeled dataset. "
    "Supervised fine-tuning teaches instruction following, and reinforcement learning from human "
    "feedback (RLHF) aligns outputs with human preferences. Parameter-efficient methods such as LoRA "
    "update only a small number of added weights.",
    "Parameter updates: parameters are updated by gradient descent. The loss, usually cross-entropy "
    "on the next token, is differentiated with backpropagation, and an optimizer such as AdamW "
    "adjusts the weights using the gradients.",
    "Hallucination: an LLM can generate fluent text that is false or unsupported by any source. "
    "Retrieval-augmented generation (RAG), which supplies trusted passages in the prompt, reduces "
    "hallucination but does not remove it.",
    "Retrieval-augmented generation: a RAG system retrieves documents relevant to the user question "
    "and inserts them into the prompt, so the model answers from current, verifiable context instead "
    "of relying only on what it memorized during training.",
    "Embeddings: a word embedding such as Word2Vec gives each word one fixed vector. A contextual "
    "embedding, produced by models like BERT or GPT, depends on the surrounding sentence, so the "
    "word 'bank' receives different vectors in 'river bank' and 'bank loan'.",
    "Context window: an LLM can only read a limited number of tokens per request, called the context "
    "window. Long documents must be split into chunks and only the relevant chunks are sent.",
    "Decoding: at generation time the model outputs a probability distribution over the vocabulary. "
    "Greedy decoding picks the most likely token, while sampling with temperature or top-p adds "
    "controlled randomness.",
]

# (question, keywords that a correct answer should mention, answerable from the KB?)
QUESTIONS = [
    ("What are the query, key and value vectors used for in self-attention?", ["query", "key", "value"], True),
    ("Why does a transformer need positional encoding?", ["order", "position"], True),
    ("What is byte-pair encoding used for?", ["token", "subword"], True),
    ("How does pretraining differ between GPT and BERT?", ["next token", "mask"], True),
    ("What does LoRA do during fine-tuning?", ["small", "weights"], True),
    ("Which optimizer is mentioned for updating parameters?", ["adamw"], True),
    ("How does RAG reduce hallucination?", ["retriev", "context"], True),
    ("How do contextual embeddings treat the word 'bank'?", ["different", "context"], True),
    ("What is the context window?", ["limit", "token"], True),
    ("What is the difference between greedy decoding and sampling?", ["most likely", "random"], True),
    ("How many parameters does GPT-4 have?", [], False),
    ("Who founded the company that made the first transformer?", [], False),
]


class Retriever:
    """TF-IDF retriever returning the top-k knowledge-base passages for a query."""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, k: int = 2) -> list[tuple[str, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [(self.documents[i], float(scores[i])) for i in order if scores[i] > 0]


def make_messages(question: str, passages: list[str] | None) -> list[dict[str, str]]:
    """Build a grounded prompt (with passages) or a plain prompt (passages=None)."""
    if passages is None:
        system = "You are a helpful assistant that answers questions about large language models concisely."
        return [{"role": "system", "content": system}, {"role": "user", "content": question}]
    system = (
        "You are a question-answering assistant about large language models. Answer in at most "
        "three sentences using ONLY the context. If the context does not contain the answer, "
        f'reply exactly: "{NO_ANSWER}"'
    )
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1)) or "(no relevant context found)"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def call_model(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    timeout: float,
) -> tuple[str, float]:
    """Send a chat-completions request and return text plus elapsed seconds."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": messages, "temperature": temperature}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach API endpoint: {exc.reason}") from exc
    elapsed = time.perf_counter() - start
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("API response did not contain choices[0].message.content") from exc
    if not isinstance(text, str):
        raise RuntimeError("API returned an empty or non-text message")
    return text.strip(), elapsed


def score_answer(answer: str, keywords: list[str], answerable: bool) -> bool:
    """Answerable: every keyword must appear. Unanswerable: the model must abstain."""
    lowered = answer.lower()
    if not answerable:
        return "don't know" in lowered or "do not know" in lowered
    return all(keyword in lowered for keyword in keywords)


def run_evaluation(args: argparse.Namespace, ask) -> None:
    retriever = Retriever(KNOWLEDGE_BASE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for mode in ("no_retrieval", "rag"):
        for index, (question, keywords, answerable) in enumerate(QUESTIONS, start=1):
            hits = retriever.search(question, k=args.top_k) if mode == "rag" else None
            messages = make_messages(question, [text for text, _ in hits] if hits is not None else None)
            if args.dry_run:
                print(json.dumps({"mode": mode, "question": index, "messages": messages}, ensure_ascii=False))
                continue
            answer, elapsed = ask(messages)
            correct = score_answer(answer, keywords, answerable)
            records.append(
                {
                    "mode": mode,
                    "question_id": index,
                    "question": question,
                    "answerable": int(answerable),
                    "top_score": round(hits[0][1], 4) if hits else "",
                    "correct": int(correct),
                    "latency_seconds": round(elapsed, 4),
                    "answer": answer.replace("\n", " "),
                }
            )
            print(f"{mode:12s} Q{index:02d}: {'correct' if correct else 'wrong'}")
    if args.dry_run:
        return

    results_path = OUTPUT_DIR / "qa_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = []
    for mode in ("no_retrieval", "rag"):
        subset = [r for r in records if r["mode"] == mode]
        unanswerable = [r for r in subset if not r["answerable"]]
        summary.append(
            {
                "mode": mode,
                "questions": len(subset),
                "accuracy": round(sum(int(r["correct"]) for r in subset) / len(subset), 4),
                "abstain_rate_on_unanswerable": round(
                    sum(int(r["correct"]) for r in unanswerable) / len(unanswerable), 4
                ),
                "mean_latency_seconds": round(sum(float(r["latency_seconds"]) for r in subset) / len(subset), 4),
            }
        )
    summary_path = OUTPUT_DIR / "qa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary ({args.model_name}, temperature={args.temperature}):")
    for row in summary:
        print(
            f"{row['mode']:12s} accuracy={row['accuracy']:.3f} "
            f"abstain={row['abstain_rate_on_unanswerable']:.3f} "
            f"mean_latency={row['mean_latency_seconds']:.3f}s"
        )
    print(f"Saved {results_path} and {summary_path}")


def run_chat(args: argparse.Namespace, ask) -> None:
    """Interactive loop: retrieve, show sources, and print the grounded answer."""
    retriever = Retriever(KNOWLEDGE_BASE)
    print("LLM question-answering chatbot. Type 'quit' to exit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", ""}:
            break
        hits = retriever.search(question, k=args.top_k)
        answer, _ = ask(make_messages(question, [text for text, _ in hits]))
        print(f"Bot: {answer}")
        for text, score in hits:
            print(f"  source (similarity {score:.2f}): {text[:70]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model name supported by the selected endpoint.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", API_BASE_URL))
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0).")
    parser.add_argument("--top-k", type=int, default=2, help="Passages retrieved per question.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds.")
    parser.add_argument("--chat", action="store_true", help="Start an interactive chat session.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without sending API requests.")
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")

    api_key = os.getenv("OPENAI_API_KEY")
    args.model_name = args.model or os.getenv("OPENAI_MODEL")
    if not args.dry_run and not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment before running the experiment.")
    if not args.dry_run and not args.model_name:
        raise SystemExit("Set OPENAI_MODEL or pass --model with a model supported by your provider.")

    def ask(messages: list[dict[str, str]]) -> tuple[str, float]:
        return call_model(
            messages,
            api_key=api_key or "",
            model=args.model_name or "",
            base_url=args.base_url,
            temperature=args.temperature,
            timeout=args.timeout,
        )

    if args.chat:
        run_chat(args, ask)
    else:
        run_evaluation(args, ask)


if __name__ == "__main__":
    main()
