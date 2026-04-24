"""Fix model naming and remove duplicates in OpenRouter evaluation result files.

Uses the OPENROUTER_TO_MODELS_INFO mapping (same as evaluate_openrouter_models.py)
to convert OpenRouter API model IDs to canonical models_info.json names.
Duplicates (same model + query_id) are resolved by keeping the last occurrence.
"""

import json
from pathlib import Path

OPENROUTER_TO_MODELS_INFO = {
    "maziyarpanahi/calme-3.2-instruct-78b": "MaziyarPanahi/calme-3.2-instruct-78b",
    "qwen/qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "qwen/qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "qwen/qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen/qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen/qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen/qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen/qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "deepseek/deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek/deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek/deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

FILES = [
    "data/model_data/openrouter_eval_cache.jsonl",
    "data/model_data/teleqna_openrouter_results.jsonl",
    "data/model_data/teleqna_test_openrouter_results.jsonl",
    "data/model_data/telequad_openrouter_results.jsonl",
]


def fix_model_name(model: str) -> str:
    return OPENROUTER_TO_MODELS_INFO.get(model, model)


def clean_file(path: str):
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {path} — file not found")
        return

    seen: dict = {}
    total = 0
    fixed = 0

    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            r = json.loads(line)
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
