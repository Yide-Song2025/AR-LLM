"""Fix model naming and remove duplicates in OpenRouter evaluation result files.

Uses the OPENROUTER_TO_MODELS_INFO mapping (same as evaluate_openrouter_models.py)
to convert OpenRouter API model IDs to canonical models_info.json names.
Duplicates (same model + query_id) are resolved by keeping the last occurrence.
"""

import json
from pathlib import Path

OPENROUTER_TO_MODELS_INFO = {
    "qwen/qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",
    "qwen/qwen3-14b": "Qwen/Qwen3-14B",
    "qwen/qwen3-32b": "Qwen/Qwen3-32B",
    "qwen/qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen/qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    "qwen/qwen3.5-122b-a10b": "Qwen/Qwen3.5-122B-A10B",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/llama-4-scout": "meta-llama/Llama-4-Scout",
    "google/gemma-4-31b-it": "google/gemma-4-31B",
    "google/gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B"
}

FILES = [
    "data/raw_data/tele_openrouter_results.jsonl",
    "data/raw_data/math_openrouter_results.jsonl",
    "data/raw_data/bbh_openrouter_results.jsonl",
    "data/raw_data/mmlu_pro_openrouter_results.jsonl",
    "data/raw_data/openrouter_eval_cache.jsonl"
]


def fix_model_name(model: str) -> str:
    return OPENROUTER_TO_MODELS_INFO.get(model, model)


def parse_jsonl_robust(filepath: str):
    """Yield parsed JSON objects from a JSONL file, handling concatenated
    objects and literal newlines embedded inside string values."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    depth = 0
    in_string = False
    escape = False
    start = None

    for i, ch in enumerate(content):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                raw = content[start:i + 1]
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    fixed = raw.replace('\n', '\\n').replace('\r', '\\r')
                    try:
                        yield json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
                start = None


def clean_file(path: str):
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {path} — file not found")
        return

    seen: dict = {}
    total = 0
    fixed = 0

    for r in parse_jsonl_robust(str(p)):
        total += 1
        old_model = r["model"]
        r["model"] = fix_model_name(old_model)
        if r["model"] != old_model:
            fixed += 1
        key = (r["model"], r["query_id"])
        seen[key] = r

    unique = len(seen)
    dupes = total - unique

    with open(p, "w", encoding="utf-8") as f:
        for r in sorted(seen.values(), key=lambda r: (r["model"], r["query_id"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    status_parts = [f"{total} -> {unique} entries"]
    if dupes:
        status_parts.append(f"{dupes} duplicates removed")
    if fixed:
        status_parts.append(f"{fixed} names fixed")
    print(f"  {path}: {', '.join(status_parts)}")


def main():
    print("Cleaning files:")
    for path in FILES:
        clean_file(path)
    print("\nDone.")


if __name__ == "__main__":
    main()
