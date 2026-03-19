#!/usr/bin/env python3
"""Generate a contrastive dataset for steering vector computation.

Calls the OpenAI API to produce prompt pairs in the JSONL format expected by
each model's `compute_vectors.py` script:

    {"pos_style": "...", "neg_style": "...", "pos": "...", "neg": "..."}

Usage
-----
python -m dataset.generate --concept "smiling vs neutral" --num_examples 100 --out dataset/smiling_vs_neutral.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

REQUIRED_KEYS = {"pos_style", "neg_style", "pos", "neg"}


def build_prompt(concept: str, num_examples: int) -> str:
    return f"""\
You are an advanced data generation assistant.

Your task is to create a contrastive dataset of {num_examples} examples for computing a steering vector.

The steering concept to focus on is: {concept}

Output exactly {num_examples} JSON objects, one per line (JSON Lines), with no list brackets, no extra commentary, and no markdown. Each line must be:
{{"pos_style": "<positive identifier>", "neg_style": "<negative identifier>", "pos": "<positive full sentence>", "neg": "<negative full sentence>"}}

---
### 1. Internal Analysis (YOUR FIRST STEP)

Before generating, analyze the concept:
* Is it an Abstract Style? (e.g., "photorealistic vs cartoon", "bright vs dark", "metal vs wood"). These can apply to any subject.
* Is it a Subject-Specific Attribute? (e.g., "smiling vs neutral" [faces], "ripe vs unripe" [fruit], "young vs old" [living beings/objects]). These are tied to a class of subjects.

Based on your analysis, you MUST follow the correct rules from Section 2.

---
### 2. Generation Guidelines (STRICT)

A. Universal Rules (Apply to ALL concepts):
* PARALLELISM: The two sentences in a pair MUST share the same syntactic skeleton and content words (subject, setting, composition, perspective).
* MINIMAL DELTA: The ONLY differences between "pos" and "neg" are the minimal tokens that express the concept contrast (e.g., "smiling" ↔ "neutral").
* STYLE NEUTRALITY: Do NOT change rendering domain, lighting, camera, or layout.
* IDENTIFIERS: Use the SAME "pos_style" and "neg_style" identifiers for ALL {num_examples} lines. These identifiers MUST also appear in the corresponding sentences.

B. Content & Subject Rules (CHOOSE A or B based on your Analysis):

    [RULE SET A] For ABSTRACT STYLES (e.g., cartoon, bright):
    * SUBJECT: You MUST vary subjects and settings widely (e.g., objects, landscapes, animals, architecture, indoor/outdoor).
    * GOAL: Decouple the style from any one context.
    * SAFETY: Avoid specific identities (age, gender, race) if people/animals are used. Use neutral terms ("person", "animal").

    [RULE SET B] For SUBJECT-SPECIFIC ATTRIBUTES (e.g., smiling, ripe):
    * SUBJECT: You MUST focus *only* on the relevant subject class (e.g., "person" or "face" for smiling; "fruit" or "plant" for ripe). Do NOT use inanimate objects like statues for concepts like "smiling".
    * GOAL: Isolate the attribute's effect on its specific subject.
    * SAFETY: To avoid bias *within* the subject class, use neutral, general terms (e.g., "person," "face," "figure," "human"). Do NOT specify age, gender, race, or ethnicity unless it is the *target concept itself*.

---
### 3. Quality Checks (must pass):
* Exactly {num_examples} lines; each is valid JSON.
* "pos" and "neg" are grammatical, depictable, and differ ONLY by the minimal concept tokens.
* The rules from Section 2 (A and B) have been correctly followed.

---
### 4. Examples

* Example for "bright vs dark" (Abstract Style - Rule A):
    {{"pos_style": "bright", "neg_style": "dark", "pos": "A bright living room with large windows.", "neg": "A dark living room with large windows."}}
* Example for "smiling vs neutral" (Subject-Specific - Rule B):
    {{"pos_style": "smiling", "neg_style": "neutral", "pos": "A photorealistic portrait of a person with a smiling expression.", "neg": "A photorealistic portrait of a person with a neutral expression."}}

Now generate {num_examples} JSON objects that represent the {concept} contrast following these instructions, one per line.\
"""


def parse_jsonl_response(text: str) -> list[dict]:
    """Parse the model's response into validated JSONL records."""
    records: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"  [warn] line {lineno}: JSON parse error — {exc}", file=sys.stderr)
            continue
        missing = REQUIRED_KEYS - obj.keys()
        if missing:
            print(f"  [warn] line {lineno}: missing keys {missing} — skipping", file=sys.stderr)
            continue
        records.append(obj)
    return records


def generate_dataset(
    concept: str,
    num_examples: int,
    model: str,
    client: OpenAI,
) -> list[dict]:
    prompt = build_prompt(concept, num_examples)
    print(f"Calling {model} for {num_examples} pairs (concept: {concept!r})…")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    raw = response.choices[0].message.content or ""
    records = parse_jsonl_response(raw)
    print(f"  parsed {len(records)} valid records (requested {num_examples})")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a contrastive JSONL dataset via OpenAI.")
    parser.add_argument("--concept", type=str, required=True,
                        help="Concept to contrast, e.g. 'cartoon' or 'smiling vs neutral'.")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of contrastive pairs to request (default: 100).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSONL path. Defaults to dataset/<concept>.jsonl.")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use (default: gpt-4o).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key. Defaults to OPENAI_API_KEY env var.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY or pass --api_key.", file=sys.stderr)
        sys.exit(1)

    out_path: Path = args.out or Path("dataset") / f"{args.concept.replace(' ', '_')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)
    records = generate_dataset(
        concept=args.concept,
        num_examples=args.num_examples,
        model=args.model,
        client=client,
    )

    if not records:
        print("Error: no valid records were generated.", file=sys.stderr)
        sys.exit(1)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()
