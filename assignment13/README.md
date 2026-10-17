# Assignment 13 — Retrieval-Augmented Generation

## Title

Implementation of a Retrieval-Augmented Generation (RAG) pipeline using open-source frameworks.

## Aim

- To study RAG and the role of each stage in the pipeline.
- To implement chunking, sparse retrieval, dense retrieval, and grounded generation with open-source libraries.
- To compare closed-book generation with TF-IDF retrieval and dense bi-encoder retrieval on the same questions.
- To evaluate retrieval and answers, including questions the knowledge base cannot answer.

## Theory

### RAG

A large language model answers from patterns stored in its weights. That knowledge is fixed at training time, can be wrong, and does not include private documents written later. **Retrieval-augmented generation (RAG)** keeps the model weights frozen and, at question time, fetches passages from an external collection and places them in the prompt. The model then reads those passages while it generates.

The generator still produces the wording. Retrieval decides which evidence it sees. A correct passage makes a grounded answer possible; a wrong or incomplete passage can still produce a fluent error. RAG reduces dependence on memorized facts. It does not guarantee correctness.

### RAG architecture

A RAG system has two main parts, joined by a prompt:

1. **Retriever.** Turns the question into a query, scores stored chunks, and returns the top $k$.
2. **Generator.** A language model that reads the question together with the retrieved chunks and writes the answer.

Before either part runs, an **indexing** pipeline prepares the collection:

```text
documents --> clean --> chunk --> embed or vectorize --> index
                                                              |
question --> embed or vectorize --> search top-k ------------+--> prompt --> generator --> answer
```

In this assignment the generator is the open-source encoder-decoder **Flan-T5-small** (`google/flan-t5-small`), loaded with Hugging Face Transformers. Two retrievers search the same chunks:

- **Sparse:** scikit-learn TF-IDF (unigrams and bigrams) with cosine similarity. A chunk scores highly when it shares rare terms with the question.
- **Dense:** the open-source bi-encoder **all-MiniLM-L6-v2**. Each chunk and the question become a 384-dimensional vector by mean-pooling token states and L2-normalizing. Cosine similarity is the dot product of those vectors. Chunks can match when the wording differs.

Larger systems store the vectors in an approximate index such as FAISS. This handbook has 29 chunks, so exact search is used and returns the true nearest chunks.

### Process of RAG

1. **Collect sources.** Gather the documents the system is allowed to use (manuals, notes, web pages, rows from a database).
2. **Clean.** Remove boilerplate, normalize whitespace, and drop pages that should not be retrieved.
3. **Chunk.** Split long documents into passages that fit the retriever and the generator's context window. This experiment uses windows of 40 words with a 10-word overlap so a fact on a boundary can appear in two chunks.
4. **Embed and index.** Compute a TF-IDF row and a dense vector for every chunk, and keep the chunk text beside them.
5. **Embed the question** with the same vectorizer or the same bi-encoder. Sparse and dense queries are not interchangeable: each must use the space it was indexed in.
6. **Retrieve.** Rank chunks by cosine similarity and keep the top $k$ (here, 2).
7. **Augment the prompt.** Insert the retrieved text and instruct the model to answer from that text, or to say `unanswerable` when the text does not contain the answer.
8. **Generate.** Decode an answer. This script uses beam search with 4 beams and no sampling, so a repeated run with the same models produces the same text.
9. **Check.** Score the retrieved chunks and the answer. A production system may also show the source chunk to the user.

### Applications

- **Company and course assistants** over handbooks, policies, and lecture notes that were never in the model's training data.
- **Customer support** grounded in the current help center, so answers follow today's product rather than an old manual.
- **Search with a written answer** over news, papers, or documentation, often with a link to the passage that was used.
- **Software assistants** that retrieve the relevant file or ticket before suggesting a change.
- **Specialized domains** such as clinical or legal text, where the system must read a supplied source and where unsupported answers are costly.

The same pipeline applies whenever the facts live in a collection that can be updated by re-indexing, without training the generator again.

## Experiment Design

### Knowledge base and questions

The script builds a 10-note handbook for a fictional "Horizon Institute Department of Neural Systems." Names, room numbers, and figures are invented, so they are not facts Flan-T5 could have memorized. Each note leads with one fact and continues with related sentences. Chunking produces **29** overlapping windows.

There are **12** questions: 10 whose answers are in the notes, and 2 that are not (the building Wi-Fi password, and an enrollment count that is never stated). One answerable question is paraphrased: the note says the department **head** can be visited on **Wednesday**, while the question asks when students can meet the department **chair**. No dataset is downloaded. The first run does download the two pretrained models from the Hugging Face Hub.

### Conditions

| Condition | Description |
|---|---|
| `closed_book` | The question is sent to Flan-T5 with no passage. |
| `tfidf` | Top-2 TF-IDF chunks are inserted into the prompt. |
| `dense` | Top-2 MiniLM chunks are inserted into the prompt. |

All three conditions use the same generator, the same 12 questions, and the same instruction to answer from the given text or say `unanswerable`.

### Settings and measures

| Setting | Configuration |
|---|---:|
| Documents | 10 handbook notes |
| Chunks | 40 words, 10-word overlap (29 chunks) |
| Sparse retriever | TF-IDF, unigrams and bigrams, cosine similarity |
| Dense retriever | `sentence-transformers/all-MiniLM-L6-v2`, mean pooling, L2 normalization |
| Generator | `google/flan-t5-small`, 4-beam search, sampling off |
| Chunks per question | 2 (`--top-k`) |
| Questions | 12 (10 answerable, 2 unanswerable) |
| Generator calls | 36 (12 per condition) |
| Answer accuracy | Answerable: every expected keyword is present. Unanswerable: the model abstains. |
| Document hit rate | Share of answerable questions whose top-k includes any chunk of the gold note |
| Answer-in-context rate | Share of answerable questions whose retrieved text contains the keywords |
| Mean reciprocal rank | Mean of $1/\mathrm{rank}$ of the first gold-document chunk |
| Latency | Generator time only, measured on CPU in this run |

Keyword matching is an automatic proxy. Short answers are saved in `outputs/rag_results.csv` and should be read as well. The question set is a lab example, not a public benchmark. Calls stay on the local machine after the model download; there is no external LLM API.

### Run the experiment

From the repository root, with the dependencies already declared in `pyproject.toml`:

```powershell
uv run python assignment13/rag_pipeline.py
```

Optional arguments:

```powershell
uv run python assignment13/rag_pipeline.py --top-k 2 --chunk-words 40 --overlap-words 10
```

The first run needs a network connection so Transformers can download `all-MiniLM-L6-v2` and `flan-t5-small`. Later runs use the Hugging Face cache. CUDA is used when it is available; otherwise the script runs on CPU.

### Generated outputs

- `outputs/chunks.csv`: every indexed window.
- `outputs/rag_results.csv`: per-question retrieval, answer, correctness, and latency.
- `outputs/rag_summary.json`: accuracy, abstention, hit rate, and latency by condition.
- `outputs/rag_comparison.png`: answer accuracy and retrieval quality.

### Observed results

The assignment was run on 2026-10-16 on CPU with `sentence-transformers/all-MiniLM-L6-v2` and `google/flan-t5-small`, beam size 4, chunk size 40, overlap 10, and top-k 2.

| Condition | Correct / total | Accuracy | Answerable accuracy | Abstain on unanswerable | Document hit | Answer in context | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Closed book | 2 / 12 | 0.167 | 0.000 | 1.000 | — | — | 0.464 s |
| TF-IDF RAG | 11 / 12 | 0.917 | 0.900 | 1.000 | 1.000 | 0.900 | 0.487 s |
| Dense RAG | 12 / 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.546 s |

Closed-book Flan-T5 replied `unanswerable` to every question. That is the right behavior on the two questions the handbook does not cover, and it means the model did not invent the handbook facts. It also means the model could not answer any of the 10 questions the notes do cover.

With retrieval, both methods abstained on those same two questions and answered 9 of the 10 handbook questions with the expected span (for example, "Dr. Leena Voss", "Search and Memory by Idris Hale", "0.81"). The remaining question is the paraphrase: "On which weekday can students meet the department chair?" The note never uses the word "chair"; it says Professor Hana Ibarra leads the department as its head and can be visited on Wednesday.

- Dense retrieval placed that sentence first (cosine 0.521). The generator answered "Wednesday".
- TF-IDF ranked a later window of the same note first because that window contains the exact word "weekday" ("processed on the next weekday"). The generator copied "next weekday". Document hit rate is still 1 for that question, because the chunk belongs to the right note, but the answer span was not in the retrieved text.

On this small set, exact-term questions were easy for both retrievers. The wording mismatch is where the dense vectors helped. Mean generator latency was about half a second per question on CPU. Per-question rows are in `outputs/rag_results.csv`.

## Conclusion

RAG indexes an external collection, retrieves a few chunks for the question, and asks a generator to answer from those chunks. This lab builds that pipeline with Hugging Face Transformers, PyTorch, and scikit-learn: overlapping chunks, a TF-IDF index, a MiniLM bi-encoder, and Flan-T5-small. On the fictional handbook, closed-book generation abstained on every question, while retrieved context raised answer accuracy from 0.167 to 0.917 (TF-IDF) and 1.000 (dense). The single miss came from retrieving a neighboring chunk that shared a word with the question but stated a different fact. Retrieval quality and answer quality have to be measured separately. The handbook, the keyword checks, and the 12-question set are instructional; they do not rank retrievers in general.

## FAQs

### 1. Explain RAG architecture?

A RAG architecture is a retriever plus a generator. The retriever searches an index of external chunks and returns the top matches for the question. The generator, usually a language model, receives the question and those chunks in its prompt and writes the answer. An indexing stage runs earlier: documents are cleaned, split into chunks, and stored as sparse vectors, dense embeddings, or both. The model weights stay fixed; updating the index changes what the system can answer.

### 2. What are the main benefits of using RAG instead of just relying on an LLM’s internal knowledge?

The collection can be updated by adding or replacing documents, without retraining the model. Private or recent material can be used even though it was never in the training corpus. Answers can be tied to a visible passage, which makes them easier to check. A smaller generator can answer a specialized question when the relevant note is in the prompt, as in this experiment, where Flan-T5-small abstained until the handbook chunk was provided. The benefit depends on retrieval: if the wrong chunk is fetched, the model can still state that chunk confidently.

### 3. What types of external knowledge sources can RAG use?

Any source that can be turned into text chunks and searched. Typical sources are PDF and HTML manuals, wikis, lecture notes, support articles, code files, tickets, spreadsheets exported to text, and rows from a database. A web search API can supply pages at query time. Some systems also index images or audio with multimodal embeddings. The source has to be one the application is allowed to read and show. This assignment uses a short handbook stored in the script.

### 4. How does the retriever work in a RAG system? What are common retrieval methods?

The retriever scores every stored chunk against the question and returns the highest scores. **Sparse** methods such as TF-IDF and BM25 match shared terms and give extra weight to rare terms; they are strong when the question repeats names, codes, or numbers from the document. **Dense** methods encode the question and the chunks with a neural bi-encoder and compare the vectors, usually by cosine similarity or dot product; they can match a paraphrase such as "chair" and "head." **Hybrid** search mixes the two score lists. A **cross-encoder reranker** can then rescore the top candidates by reading the question and the chunk together, which is slower and usually more precise. This experiment compares TF-IDF and a MiniLM bi-encoder with no reranking step.

### 5. What are some common ways to evaluate RAG systems?

Evaluate retrieval and generation separately, then look at the full answer.

- **Retrieval:** Recall@k or hit rate (was the supporting chunk in the top k?), mean reciprocal rank, and nDCG. This lab reports document hit rate, whether the answer keywords appear in the retrieved text, and mean reciprocal rank.
- **Answer text:** Exact match, token F1, or a keyword check against a reference answer. This lab uses keyword presence for answerable questions.
- **Abstention:** On questions the collection cannot answer, the desirable output is a refusal. This lab scores replies that say they are unanswerable.
- **Groundedness:** Whether the answer is supported by the retrieved chunks, not only whether it matches a reference. Frameworks such as RAGAS score faithfulness, answer relevance, and context precision with a judge model or with human review.
- **Usefulness:** Latency, cost, and citation quality.

A high document hit rate can still hide a bad chunk, which is what happened here for the TF-IDF paraphrase. Automatic scores on a small set should be read together with the saved answers.
