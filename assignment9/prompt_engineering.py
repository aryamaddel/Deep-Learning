"""Assignment 9: compare prompt designs with a chat-completions API.

Run from the repository root after setting OPENAI_API_KEY and OPENAI_MODEL:
    uv run python assignment9/prompt_engineering.py

The script uses Python's standard library for HTTP requests. It expects an
OpenAI-compatible Chat Completions endpoint; the base URL can be configured for
compatible providers or local servers.
"""

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
API_BASE_URL = "https://code.in2peta.com"

# A small, balanced set of short product comments makes the study inexpensive
# and easy to inspect. Keep the same examples for every prompt condition.
EXAMPLES = [
    ("The headphones sound clear and feel comfortable all day.", "positive"),
    ("Setup was simple, and the battery lasts much longer than expected.", "positive"),
    ("The screen is bright and the controls are easy to use.", "positive"),
    ("It works exactly as described and arrived in good condition.", "positive"),
    ("The case feels sturdy and fits my device perfectly.", "positive"),
    ("The keyboard is responsive and makes typing pleasant.", "positive"),
    ("The charger stopped working after two days.", "negative"),
    ("The app crashes often and loses my settings.", "negative"),
    ("The material feels cheap, and one button is already loose.", "negative"),
    ("It arrived late with a cracked screen.", "negative"),
    ("The instructions are confusing and the setup failed twice.", "negative"),
    ("The battery drains quickly even when the device is idle.", "negative"),
]

PROMPT_STYLES = ("minimal", "structured", "few_shot")
FEW_SHOT = (
    'Example: comment="The package arrived early and works perfectly." '
    '-> {"label":"positive"}\n'
    'Example: comment="It broke on the first day." -> {"label":"negative"}\n\n'
)


def make_messages(style: str, comment: str) -> list[dict[str, str]]:
    """Build one experimental prompt while keeping the output contract fixed."""
    comment = comment.replace("\n", " ")
    if style == "minimal":
        system = 'Classify product-comment sentiment. Return JSON only: {"label":"positive"} or {"label":"negative"}.'
        user = f'Comment: "{comment}"'
    elif style == "structured":
        system = (
            "You classify the sentiment expressed in a product comment. "
            "Choose positive when the overall opinion is favorable and negative "
            "when it is unfavorable. Return exactly one JSON object with the key "
            '"label" and one value, "positive" or "negative". Do not add explanation.'
        )
        user = f"Product comment to classify:\n{comment}"
    elif style == "few_shot":
        system = (
            "Classify the overall sentiment of product comments. Return exactly "
            'one JSON object: {"label":"positive"} or {"label":"negative"}. '
            "Do not add explanation.\n\n"
            + FEW_SHOT
            + "Apply the same rule to the new comment."
        )
        user = f"New comment: {comment}"
    else:
        raise ValueError(f"Unknown prompt style: {style}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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
    return text, elapsed


def parse_label(response: str) -> tuple[str, bool]:
    """Read a sentiment label and report whether the response is valid JSON."""
    try:
        value = json.loads(response)
        valid_json = isinstance(value, dict)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", response, flags=re.DOTALL)
        if not match:
            return "invalid", False
        try:
            value = json.loads(match.group(0))
            valid_json = False
        except json.JSONDecodeError:
            return "invalid", False
    label = str(value.get("label", "")).strip().lower()
    return (label if label in {"positive", "negative"} else "invalid"), valid_json


def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    model = args.model or os.getenv("OPENAI_MODEL")
    if not args.dry_run and not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment before running the experiment.")
    if not args.dry_run and not model:
        raise SystemExit("Set OPENAI_MODEL or pass --model with a model supported by your provider.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for style in PROMPT_STYLES:
        for index, (comment, expected) in enumerate(EXAMPLES, start=1):
            messages = make_messages(style, comment)
            if args.dry_run:
                print(json.dumps({"style": style, "sample": index, "messages": messages}, ensure_ascii=False))
                continue
            response, elapsed = call_model(
                messages,
                api_key=api_key or "",
                model=model or "",
                base_url=args.base_url,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            predicted, valid_json = parse_label(response)
            records.append(
                {
                    "style": style,
                    "sample_id": index,
                    "comment": comment,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": int(predicted == expected),
                    "valid_json": int(valid_json),
                    "latency_seconds": round(elapsed, 4),
                    "response": response,
                }
            )
            print(f"{style:10s} sample {index:02d}/{len(EXAMPLES)}: {predicted}")

    if args.dry_run:
        return
    results_path = OUTPUT_DIR / "prompt_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = []
    for style in PROMPT_STYLES:
        subset = [record for record in records if record["style"] == style]
        summary.append(
            {
                "style": style,
                "examples": len(subset),
                "accuracy": round(sum(int(row["correct"]) for row in subset) / len(subset), 4),
                "valid_json_rate": round(sum(int(row["valid_json"]) for row in subset) / len(subset), 4),
                "mean_latency_seconds": round(
                    sum(float(row["latency_seconds"]) for row in subset) / len(subset), 4
                ),
            }
        )
    summary_path = OUTPUT_DIR / "prompt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary ({model}, temperature={args.temperature}):")
    for row in summary:
        print(
            f"{row['style']:10s} accuracy={row['accuracy']:.3f} "
            f"valid_json={row['valid_json_rate']:.3f} "
            f"mean_latency={row['mean_latency_seconds']:.3f}s"
        )
    print(f"Saved {results_path} and {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model name supported by the selected endpoint.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", API_BASE_URL),
        help="API base URL (default: OPENAI_BASE_URL or In2Peta endpoint).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0).")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without sending API requests.")
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    run(args)


if __name__ == "__main__":
    main()
