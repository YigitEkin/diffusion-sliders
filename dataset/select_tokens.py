#!/usr/bin/env python3
"""Token selection: asks an LLM which tokens in a prompt to steer for a concept.

Rules follow token_finding.txt: identify the minimal set of tokens that attach the
steering concept to the prompt (subject nouns, style words, or all content words for
global edits).

Usage
-----
python -m dataset.select_tokens --prompt "make the scene cartoon" --concept "cartoon"
"""

from __future__ import annotations

import argparse

_INSTRUCTIONS = """\
You are an expert token selection assistant. Your job is to identify the exact tokens from a PROMPT that should be steered, based on a steering CONCEPT.

DEFINITIONS

CONCEPT TYPE:
\t•\tLocal Edit: A trait that applies to a specific subject (e.g., "smile", "age", "ripe", "sad").
\t•\tStylization Edit: A rendering style (e.g., "photorealistic", "cartoon", "anime", "dark").
\t•\tGlobal Edit: A change to the entire scene's context or environment (e.g., "winter", "summer", "crowded").

PROMPT TYPE:
\t•\tImplicit: The prompt is neutral and does not contain words related to the concept (e.g., PROMPT: "a man", CONCEPT: "smile").
\t•\tExplicit: The prompt contains a word related to the concept, usually a positive or negative pole (e.g., PROMPT: "a sad man", CONCEPT: "smile"; PROMPT: "a photorealistic man", CONCEPT: "cartoon").

⸻

RULES (Apply in order)
\t1.\tIf the CONCEPT is a "Global Edit":
\t\t•\t-> Output all meaningful content words from the prompt that describe the scene or objects (ignore filler words and punctuation).
\t\t•\tDo not include articles, prepositions, or conjunctions unless they are semantically part of a named object or phrase.
\t2.\tIf the CONCEPT is a "Stylization Edit":
\t\t•\tIf the PROMPT is Explicit:
\t\t\t-> Output only the explicit style or appearance words (e.g., "photorealistic", "cinematic", "cartoon").
\t\t•\tIf the PROMPT is Implicit:
\t\t\t-> Output only the main subject nouns that the style can logically apply to (e.g., "man", "lighthouse", "forest").
\t3.\tIf the CONCEPT is a "Local Edit":
\t\t•\tIf the PROMPT is Explicit:
\t\t\t-> Output only the explicit descriptive or emotional words expressing the local attribute (e.g., "sad", "old", "angry", "smiling").
\t\t•\tIf the PROMPT is Implicit:
\t\t\t-> Output only the subject nouns that the local attribute can naturally attach to (e.g., "man", "woman", "child", "face", "fruit").
\t\tPrefer the main human, animal, or animate entity; if none, choose the most central object noun in the description.
\t4.\tGeneral constraints:
\t\t•\tExclude punctuation and purely functional words (articles, prepositions, etc.).
\t\t•\tReturn only the minimal set of tokens required to attach the concept.

⸻

EXAMPLES

Global Edit:
\t•\tPROMPT: "a woman in a park", CONCEPT: "winter" (Global Edit)
\t•\tOUTPUT:
\t\tCONCEPT_TYPE: Global Edit
\t\tTOKENS: woman park

Stylization Edit:
\t•\tPROMPT: "a photorealistic lighthouse on a cliff", CONCEPT: "cartoon" (Stylization Edit, Explicit)
\t•\tOUTPUT:
\t\tCONCEPT_TYPE: Stylization Edit
\t\tTOKENS: photorealistic
\t•\tPROMPT: "a lighthouse on a cliff", CONCEPT: "cartoon" (Stylization Edit, Implicit)
\t•\tOUTPUT:
\t\tCONCEPT_TYPE: Stylization Edit
\t\tTOKENS: lighthouse

Local Edit:
\t•\tPROMPT: "a portrait of a sad man", CONCEPT: "smile" (Local Edit, Explicit)
\t•\tOUTPUT:
\t\tCONCEPT_TYPE: Local Edit
\t\tTOKENS: sad
\t•\tPROMPT: "a portrait of a man", CONCEPT: "smile" (Local Edit, Implicit)
\t•\tOUTPUT:
\t\tCONCEPT_TYPE: Local Edit
\t\tTOKENS: man
\t•\tPROMPT: "a ripe tomato on the vine", CONCEPT: "age" (Local Edit, Explicit)
\t•\tOUTPUT:
\t\tCONCEPT_TYPE: Local Edit
\t\tTOKENS: ripe
\t•\tPROMPT: "a tomato on the vine", CONCEPT: "age" (Local Edit, Implicit)
\t•\tOUTPUT:
\t\tCONCEPT_TYPE: Local Edit
\t\tTOKENS: tomato

⸻

YOUR TASK

Analyze the following PROMPT and CONCEPT using this logic. Output exactly two lines and nothing else:
Line 1: CONCEPT_TYPE: <one of: Local Edit, Stylization Edit, Global Edit>
Line 2: TOKENS: <the tokens separated by a single space>
Do not add any commentary, explanation, or punctuation beyond this format.\
"""


def build_prompt(prompt: str, concept: str) -> str:
    return f"""{_INSTRUCTIONS}

PROMPT: "{prompt}"
CONCEPT: "{concept}"
OUTPUT:"""


def select_tokens(
    prompt: str,
    concept: str,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
) -> tuple[list[str], str]:
    """Return (tokens, concept_type) for *prompt* steered by *concept*.

    tokens: the minimal token list to steer.
    concept_type: one of "Local Edit", "Stylization Edit", or "Global Edit".

    Runs Qwen3-8B locally via transformers — no API key required.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [{"role": "user", "content": build_prompt(prompt, concept)}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=4096)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Skip thinking block if present (</think> token id = 151668)
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    raw = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    del model
    torch.cuda.empty_cache()

    concept_type: str = ""
    tokens: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("CONCEPT_TYPE:"):
            concept_type = line[len("CONCEPT_TYPE:"):].strip()
        elif line.startswith("TOKENS:"):
            tokens = [t.strip() for t in line[len("TOKENS:"):].split() if t.strip()]

    if not tokens:
        raise ValueError(
            f"Token selection returned no tokens for prompt={prompt!r}, concept={concept!r}. "
            f"Raw response: {raw!r}"
        )
    if not concept_type:
        raise ValueError(
            f"Token selection returned no concept type for prompt={prompt!r}, concept={concept!r}. "
            f"Raw response: {raw!r}"
        )
    return tokens, concept_type


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select steering tokens for a prompt+concept pair.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokens, concept_type = select_tokens(args.prompt, args.concept, args.model, args.device)
    print(f"CONCEPT_TYPE: {concept_type}")
    print(f"TOKENS: {' '.join(tokens)}")


if __name__ == "__main__":
    main()
