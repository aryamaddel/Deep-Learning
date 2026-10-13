# Assignment 9 — Prompt Engineering for Generative AI

## Title

Prompt engineering, API-based text generation, and evaluation of prompt quality.

## Aim

- To identify the components of an effective prompt.
- To compare direct, structured, and few-shot prompting techniques.
- To understand how Python applications send requests to Generative AI APIs.
- To evaluate generated responses with task accuracy and output-format checks.
- To recognize limitations, safety concerns, and ethical responsibilities in prompt engineering.

## Theory

### Components of an Effective Prompt

A prompt is the input and instructions provided to a generative model. A useful
prompt communicates the task clearly and gives the model the context needed to
complete it. Depending on the task, it may include:

1. **Role or perspective:** The kind of assistant or expertise expected, such as
   “You classify product feedback.” A role can guide style, but does not prove
   that the model has real expertise.
2. **Task:** A specific action, such as summarize, classify, extract, compare,
   or draft.
3. **Context and input:** Relevant background, source material, definitions,
   and the actual content to process. Marking the input clearly helps separate
   it from instructions.
4. **Constraints:** Scope, audience, tone, length, exclusions, and rules for
   handling missing or uncertain information.
5. **Output format:** A required structure such as a table, bullet list, or
   schema-constrained JSON. A format instruction improves consistency, but
   application code should still validate the result.
6. **Examples:** One or more input-output pairs can demonstrate the desired
   behavior. Examples should be representative and should not disclose
   sensitive information.
7. **Grounding material:** Trusted documents or data the model should use.
   Explicitly ask the model to identify when the supplied evidence is
   insufficient rather than inventing facts.

The prompt should contain enough detail to remove ambiguity without adding
irrelevant instructions. For repeated tasks, save a versioned prompt template
and evaluate changes against the same test cases.

### Prompting Techniques

- **Zero-shot prompting:** Give the task and required output without examples.
  It is quick to write and works well for clear, familiar tasks.
- **Few-shot prompting:** Include a small number of examples in the prompt.
  Examples communicate labels, edge cases, or a target style; poor or biased
  examples can also teach the wrong behavior.
- **Structured prompting:** Organize the request into task, context,
  constraints, and output sections. This makes requirements easier to inspect.
- **Decomposition:** Break a complex task into smaller stages, such as
  extracting facts before summarizing. Each stage should be checked, since an
  early error can carry into later output.
- **Grounded prompting:** Supply relevant source text and ask for answers tied
  to that material, optionally requesting evidence or citations. Grounding
  reduces reliance on unsupported recall but does not guarantee correctness.
- **Iterative refinement:** Inspect errors, change one prompt element at a
  time, and rerun a fixed evaluation set. Keep successful examples and record
  prompt versions so results are comparable.
- **Output-constrained prompting:** Specify a schema or allowed labels for
  downstream processing. Validate outputs in code and define a recovery path
  for malformed or incomplete responses.

### Generative AI Tools and APIs

Generative AI tools include chat interfaces, coding assistants, image and
audio generators, and developer APIs. A chat interface is designed for direct
human interaction. An API exposes model capabilities to a program, which can
send inputs and receive structured responses. Available models, supported
parameters, pricing, data handling, and request formats vary by provider and
can change; consult the provider's current documentation before deployment.

An API request usually contains an endpoint, authentication, a model or task,
input messages, and optional generation parameters. The response may include
generated text, usage information, status metadata, or errors. The prompt is
the instruction/content sent to the model; the API request is the complete
network operation carrying that prompt and its settings.

This experiment uses Python's built-in `urllib` to call an OpenAI-compatible
Chat Completions endpoint. It is configured for the In2Peta Coding Platform at
`https://code.in2peta.com` with model `kimi-k3`. Many providers and local
inference servers expose compatible endpoints, but compatibility is not
universal. For example, OpenAI also documents its newer Responses API; the
endpoint and payload used here are specifically for compatible Chat
Completions services. See the [official API reference](https://platform.openai.com/docs/api-reference/chat)
and [authentication guidance](https://platform.openai.com/docs/api-reference/authentication).

Set credentials in the environment rather than committing them to source. For
the configured endpoint and model:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "kimi-k3"
$env:OPENAI_BASE_URL = "https://code.in2peta.com"
```

The API key authenticates the caller and associates requests with an account
or project. It is a secret credential: do not publish it, place it in a
notebook, or include it in logs. Use a restricted project key where available,
monitor usage, and rotate exposed credentials. API access may incur charges and
is subject to provider limits and data policies.

### Prompt Evaluation Criteria

Evaluation should be defined before comparing prompts. Select criteria that
match the task and score the same held-out examples for each prompt version.

- **Accuracy:** Does the answer match a known correct label or reference?
- **Relevance:** Does it address the requested task and avoid unrelated content?
- **Completeness:** Does it include all required fields or key information?
- **Groundedness:** Are factual claims supported by supplied or trusted sources?
- **Clarity:** Is the response understandable to the intended audience?
- **Format compliance:** Does it follow the requested schema or structure?
- **Consistency and robustness:** Does behavior hold across paraphrases,
  ambiguous inputs, and repeated runs?
- **Safety and fairness:** Does it avoid harmful output and perform equitably
  across relevant user groups and cases?
- **Cost and latency:** Is quality acceptable for the tokens, time, and money
  required?

Automated metrics can be useful, but no single score captures quality. Exact
match is suitable for a fixed label task; human review is often needed for
open-ended writing, factual nuance, fairness, and harmful edge cases. A model
should not grade its own output without independent validation.

### Temperature

Temperature controls the randomness of token selection in models that expose
the setting. Lower values generally favor more predictable completions, while
higher values generally allow more varied completions. It does not directly
measure factual confidence or quality, and providers may implement or restrict
it differently. The experiment holds temperature at `0` by default to reduce
sampling variation; this does not make outputs perfectly deterministic.

## Experiment Design

### Task and data

The script classifies 12 short product comments as positive or negative. The
balanced examples are built into the script, so there is no dataset download.
Each comment is sent once under each of three prompt conditions: a minimal
instruction, a structured instruction with clearer decision rules, and a
few-shot instruction with two examples. All conditions request the same JSON
label, use the same model and temperature, and are scored against the same
expected labels. The small illustrative set is not a benchmark of provider
quality or broad language understanding.

### Settings and measures

| Setting | Configuration |
|---|---:|
| Prompt conditions | Minimal, structured, few-shot |
| Test examples | 12 balanced product comments |
| API calls | 36 (12 per prompt condition) |
| Temperature | 0 by default; configurable |
| Primary metric | Exact label accuracy |
| Additional metrics | Valid JSON rate, mean latency |
| API format | OpenAI-compatible Chat Completions |

Accuracy is the fraction of responses whose parsed label matches the expected
label. Valid JSON rate measures strict JSON output. Mean latency is measured
from request start to response receipt and includes network variation. Inspect
individual errors as well as aggregate metrics. Calls may incur provider
charges.

### Run the experiment

Set `OPENAI_API_KEY` to your In2Peta credential, then run from the repository
root. The endpoint defaults to `https://code.in2peta.com` and the model can be
set to `kimi-k3`:

```powershell
uv run python assignment9/prompt_engineering.py
```

To use a compatible endpoint with a different base URL:

```powershell
$env:OPENAI_BASE_URL = "https://provider.example/v1"
uv run python assignment9/prompt_engineering.py --model "provider-model-id"
```

The script uses only Python's standard library. Use `--temperature 0.4` to
repeat the comparison at a different sampling setting. Inspect the request
prompts without making API calls using:

```powershell
uv run python assignment9/prompt_engineering.py --dry-run
```

### Generated outputs

- `outputs/prompt_results.csv`: one row per prompt and test comment, including
  expected and predicted labels, correctness, JSON validity, latency, and raw
  response.
- `outputs/prompt_summary.json`: accuracy, valid JSON rate, and mean latency
  summarized by prompt style.

Use a real API run to report observed results. Model behavior, provider,
endpoint, temperature, date, and network conditions should accompany the
reported measurements. Do not invent results from the dry run.

### Observed results

The assignment was run on 2026-10-10 against the In2Peta Coding Platform
(`https://code.in2peta.com`) with model `kimi-k3` at temperature `0.0`:

| Prompt style | Correct / total | Accuracy | Valid JSON | Mean latency |
|---|---:|---:|---:|---:|
| Minimal | 12 / 12 | 1.000 | 1.000 | 1.2942 s |
| Structured | 12 / 12 | 1.000 | 1.000 | 1.0199 s |
| Few-shot | 12 / 12 | 1.000 | 1.000 | 0.9341 s |

All prompt styles classified this small, straightforward set correctly and
returned valid JSON. The lower observed latency for the few-shot prompts is a
single sequential run and may reflect network or service variation; it is not
evidence that few-shot prompting is faster in general. The perfect scores also
create a ceiling effect, so a larger and more ambiguous test set would be
needed to distinguish prompt quality reliably. Per-example outputs are saved
in `outputs/prompt_results.csv`; aggregated values are in
`outputs/prompt_summary.json`.

## Conclusion

Effective prompts state the task, provide relevant context, set useful
constraints, and define the expected output. Structured instructions and
examples can improve consistency for some tasks, but the best prompt depends
on the model and data. The experiment compares three prompt styles on a small
sentiment dataset using label accuracy, JSON validity, and latency. The In2Peta
`kimi-k3` run scored 100% accuracy and JSON validity for all three styles on
the 12-example set. Since every style reached the same ceiling, the experiment
does not establish a quality winner. The result is an instructional
comparison, not a general ranking of prompting techniques.

## FAQs

### 1. What is a prompt in Generative AI?

A prompt is the instruction, question, context, and/or data provided to a
generative model to guide its output. It can include examples and formatting
requirements as well as the core task.

### 2. What is the difference between a prompt and an API request?

A prompt is the model-facing content. An API request is the full communication
sent by an application: it can include the prompt, endpoint, authentication,
model identifier, generation settings, and other metadata. The response and
errors are returned by the API.

### 3. What is a Generative AI API?

It is a software interface that lets an application submit inputs to a
generative model and receive outputs programmatically. APIs can support text,
images, audio, embeddings, or other capabilities depending on the provider.

### 4. How can a Generative AI API be accessed using Python?

Use a provider's SDK or send an HTTP request with a Python HTTP library. The
program supplies the endpoint, credentials, model, input, and supported
parameters, then parses the response and handles errors. This assignment
demonstrates the standard-library `urllib` approach in
`prompt_engineering.py`.

### 5. What is the purpose of an API key?

An API key authenticates a client and associates its requests with an account
or project. Treat it as a secret: store it outside source control, limit its
permissions where possible, and rotate it if exposed.

### 6. What is temperature in Generative AI models?

Temperature is a generation setting that influences randomness when selecting
tokens. Its supported range and exact behavior depend on the provider and
model.

### 7. How does changing temperature affect generated responses?

Lower temperatures generally make outputs more predictable; higher
temperatures generally make them more varied. Temperature does not guarantee
accuracy, and changing it may not be supported by every model.

### 8. What is prompt refinement?

Prompt refinement is the process of improving instructions based on observed
outputs. A careful process changes one element at a time and checks each
version against the same evaluation examples.

### 9. How can the quality of an AI-generated response be evaluated?

Use criteria tied to the task, such as accuracy, relevance, completeness,
groundedness, clarity, format compliance, safety, and latency. Combine
automated checks with human review where nuance matters, and evaluate on data
that was not used to write the prompt.

### 10. What are hallucinations in Generative AI?

Hallucinations are plausible-sounding outputs that contain unsupported or
incorrect information. Supplying trusted source material, asking the model to
state uncertainty, checking citations, and independently verifying important
claims can reduce risk, but cannot eliminate it.

### 11. What are the limitations and ethical considerations of Prompt Engineering?

Prompting cannot remove a model's knowledge limits, biases, or tendency to
make errors. Outputs can vary across models and versions, and a prompt may
fail on ambiguous or adversarial inputs. Consider privacy, consent,
copyright, bias and fairness, accessibility, harmful use, and the impact of
automated decisions. Minimize sensitive data, test representative edge cases,
keep human oversight for consequential decisions, and disclose AI use where
appropriate.
