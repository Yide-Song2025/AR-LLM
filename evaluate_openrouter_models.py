import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI, RateLimitError, APIError, BadRequestError
from datasets import load_dataset

# Map OpenRouter model names to their models_info.json keys
OPENROUTER_TO_MODELS_INFO = {
    "maziyarpanahi/calme-3.2-instruct-78b": "MaziyarPanahi/calme-3.2-instruct-78b",
    # "qwen/qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "qwen/qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "qwen/qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen/qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen/qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen/qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen/qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    # "meta-llama/llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    # "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    # "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "deepseek/deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek/deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek/deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

MODELS = list(OPENROUTER_TO_MODELS_INFO.keys())


def canonical_model_name(openrouter_model: str) -> str:
    """Return the models_info key, falling back to the input name."""
    return OPENROUTER_TO_MODELS_INFO.get(openrouter_model, openrouter_model)

# ----------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------
OUT_TELEQNA_ALL  = "data/model_data/teleqna_openrouter_results.jsonl"
OUT_TELEQNA_TEST = "data/model_data/teleqna_test_openrouter_results.jsonl"
OUT_TELEQUAD     = "data/model_data/telequad_openrouter_results.jsonl"

# Temporary cache file for interrupt recovery
CACHE_FILE = "data/model_data/openrouter_eval_cache.jsonl"

MAX_WORKERS = 32

# ----------------------------------------------------------------------
# API clients (defaults — can be overridden via --base-url / --api-key)
# ----------------------------------------------------------------------
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY = "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"

openrouter_client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY)
deepseek_client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-a98fe343cce6419b97eb35829b0d72e5")


def init_clients(base_url: str, api_key: str):
    global openrouter_client
    openrouter_client = OpenAI(base_url=base_url, api_key=api_key)

# ----------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------
TELEQNA_TEMPLATE = """{question}

{choices_formatted}

Reply with just the answer inside /box{{}}."""

TELEQUAD_TEMPLATE = """Context: {context}

Question: {question}

Reply with just the answer inside /box{{}}."""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content inside /box{...} from model output."""
    match = re.search(r'/box\{([^}]*)\}', text)
    return match.group(1).strip() if match else None


def call_openrouter(model: str, prompt: str, max_retries: int = 1) -> Optional[str]:
    """Call OpenRouter API with exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except BadRequestError as e:
            # Model unavailable or bad request
            if attempt == 0:
                print(f"    [WARN] BadRequestError for {model}: {e}")
            return None
        except RateLimitError as e:
            if attempt == 0:
                print(f"    [WARN] Rate limited for {model}: {e}")
            time.sleep(2)
        except APIError as e:
            if attempt == 0:
                print(f"    [WARN] APIError for {model}: {e}")
            time.sleep(2)
        except Exception as e:
            if attempt == 0:
                print(f"    [WARN] Unexpected error for {model}: {e}")
            time.sleep(2)
    return None


def check_model_available(model: str) -> bool:
    """Probe model with a trivial prompt to check availability."""
    try:
        response = openrouter_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
            max_tokens=5,
        )
        return True
    except BadRequestError:
        return False
    except Exception:
        return True  # Assume available on other errors


def judge_with_deepseek(question: str, reference: str, response_text: str, max_retries: int = 3, verbose: bool = False) -> int:
    """Use DeepSeek to judge if model's answer is semantically equivalent to reference."""
    prompt = (
        "You are a precise answer evaluator. Given a question, the reference (correct) answer, "
        "and a model's response, determine if the model's answer is semantically equivalent to the reference.\n\n"
        f"Question: {question}\n\n"
        f"Reference Answer: {reference}\n\n"
        f"Model Response: {response_text}\n\n"
        "Respond with ONLY '1' if semantically equivalent (same meaning), or '0' if not equivalent or cannot be determined."
    )
    for attempt in range(max_retries):
        try:
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a precise answer evaluator. Respond only with 1 or 0."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10,
            )
            result = resp.choices[0].message.content.strip()
            if verbose:
                print(f"    [DEEPSEEK PROMPT]\n{prompt}")
                print(f"    [DEEPSEEK JUDGE] {result}")
            if result == "1":
                return 1
            elif result == "0":
                return 0
        except Exception as e:
            if attempt == 0:
                print(f"    [WARN] DeepSeek judge error: {e}")
            time.sleep(1)
    return 0


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------
def load_teleqna() -> Tuple[List[Dict], List[Dict]]:
    """Load TeleQnA train and test splits from HuggingFace."""
    ds = load_dataset("ymoslem/TeleQnA-processed", split="train")
    train_items = []
    for idx, row in enumerate(ds):
        train_items.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer_idx": int(row["answer"]),
            "subject": row["subject"],
            "explanation": row.get("explanation", ""),
            "split": "train",
            "global_idx": idx,
        })

    ds_test = load_dataset("ymoslem/TeleQnA-processed", split="test")
    test_items = []
    for idx, row in enumerate(ds_test):
        test_items.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer_idx": int(row["answer"]),
            "subject": row["subject"],
            "explanation": row.get("explanation", ""),
            "split": "test",
            "global_idx": idx,
        })

    print(f"  Loaded TeleQnA: {len(train_items)} train, {len(test_items)} test")
    return train_items, test_items


def load_telequad() -> List[Dict]:
    """Parse TeleQuAD from local JSON, flatten to list of items."""
    path = Path("data/TeleQuAD-v4-full.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for doc in raw["data"]:
        source = doc.get("source", "telequad")
        for para in doc.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                items.append({
                    "context": context,
                    "question": qa["question"],
                    "ref_answer": answers[0]["text"],
                    "qa_id": qa["id"],
                    "source": source,
                })

    print(f"  Loaded TeleQuAD: {len(items)} questions")
    return items


# ----------------------------------------------------------------------
# Evaluation logic
# ----------------------------------------------------------------------
def evaluate_teleqna_item(item: Dict, model: str, verbose: bool = False) -> Optional[Dict]:
    """Evaluate a single TeleQnA item via OpenRouter. Returns None on error."""
    choices = item["choices"]
    choices_formatted = "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
    prompt = TELEQNA_TEMPLATE.format(
        question=item["question"],
        choices_formatted=choices_formatted,
    )

    response_text = call_openrouter(model, prompt)
    if response_text is None:
        return None

    query_id = f"teleqna_{item['split']}_{item['global_idx']}"
    ground_truth = choices[item["answer_idx"]]

    if verbose:
        print(f"\n    [PROMPT]\n{prompt}")
        print(f"    [RESPONSE] {response_text}")
        print(f"    [GT] {ground_truth}")

    boxed = extract_boxed_answer(response_text)

    # Try matching: first as integer index, then as text
    correct = 0.0
    if boxed is not None:
        # Try index match
        try:
            idx = int(boxed)
            if 0 <= idx < len(choices) and idx == item["answer_idx"]:
                correct = 1.0
        except ValueError:
            pass

        # Try text match (case-insensitive, strip)
        if correct == 0.0:
            if boxed.strip().lower() == ground_truth.strip().lower():
                correct = 1.0

    return {
        "query": item["question"],
        "answer": ground_truth,
        "model": model,
        "dataset": "teleqna",
        "subset": item["subject"],
        "correct": correct,
        "query_id": query_id,
    }


def evaluate_telequad_item(item: Dict, model: str, idx: int, verbose: bool = False) -> Optional[Dict]:
    """Evaluate a single TeleQuAD item via OpenRouter, judge with DeepSeek. Returns None on error."""
    prompt = TELEQUAD_TEMPLATE.format(
        context=item["context"],
        question=item["question"],
    )

    response_text = call_openrouter(model, prompt)
    if response_text is None:
        return None

    query_id = f"telequad_{idx}"

    if verbose:
        print(f"\n    [PROMPT]\n{prompt}")
        print(f"    [RESPONSE] {response_text}")
        print(f"    [GT] {item['ref_answer']}")

    boxed = extract_boxed_answer(response_text)
    response_for_judge = boxed if boxed is not None else response_text

    # Judge with DeepSeek
    correct = judge_with_deepseek(item["question"], item["ref_answer"], response_for_judge, verbose=verbose)

    return {
        "query": item["question"],
        "answer": item["ref_answer"],
        "model": model,
        "dataset": "telequad",
        "subset": item["source"],
        "correct": float(correct),
        "query_id": query_id,
    }


# ----------------------------------------------------------------------
# Parallel evaluation helpers
# ----------------------------------------------------------------------
def _teleqna_worker(args: Tuple[Dict, str, bool]) -> Optional[Dict]:
    item, model, verbose = args
    return evaluate_teleqna_item(item, model, verbose)


def _telequad_worker(args: Tuple[Dict, str, int, bool]) -> Optional[Dict]:
    item, model, idx, verbose = args
    return evaluate_telequad_item(item, model, idx, verbose)


def process_teleqna(items: List[Dict], model: str, desc: str, verbose: bool = False, save_every: int = 10) -> List[Dict]:
    """Evaluate TeleQnA items in parallel, saving incrementally every `save_every` results."""
    results: List[Dict] = []
    pending: List[Dict] = []
    args_list = [(item, model, verbose) for item in items]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_teleqna_worker, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    pending.append(result)
                    if len(pending) >= save_every:
                        save_to_cache(pending)
                        pending = []
            except Exception as e:
                print(f"    [ERROR] Worker exception: {e}")
    if pending:
        save_to_cache(pending)
    return results


def process_telequad(indexed_items: List[Tuple[int, Dict]], model: str, verbose: bool = False, save_every: int = 10) -> List[Dict]:
    """Evaluate TeleQuAD items in parallel, saving incrementally every `save_every` results."""
    results: List[Dict] = []
    pending: List[Dict] = []
    args_list = [(item, model, idx, verbose) for idx, item in indexed_items]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_telequad_worker, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  TeleQuAD", leave=False):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    pending.append(result)
                    if len(pending) >= save_every:
                        save_to_cache(pending)
                        pending = []
            except Exception as e:
                print(f"    [ERROR] Worker exception: {e}")
    if pending:
        save_to_cache(pending)
    return results


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------
def load_cache() -> Dict[str, Dict]:
    """Load cache from disk. Returns {model: {query_id: result}}."""
    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        return {}
    cache: Dict[str, Dict] = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            model = r["model"]
            query_id = r["query_id"]
            if model not in cache:
                cache[model] = {}
            cache[model][query_id] = r
    return cache


def save_to_cache(results: List[Dict]):
    """Append results to the cache file."""
    path = Path(CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rebuild_outputs_from_cache(cache: Dict[str, Dict], available_models: List[str]):
    """Rebuild all three output files from cache, only for available models."""
    # Rebuild TeleQnA combined (train + test)
    teleqna_all: List[Dict] = []
    teleqna_test: List[Dict] = []
    telequad_all: List[Dict] = []

    for model in available_models:
        if model not in cache:
            continue
        for r in cache[model].values():
            if r["dataset"] == "teleqna":
                teleqna_all.append(r)
                if "_test_" in r["query_id"]:
                    teleqna_test.append(r)
            else:
                telequad_all.append(r)

    teleqna_all.sort(key=lambda r: r["query_id"])
    teleqna_test.sort(key=lambda r: r["query_id"])
    telequad_all.sort(key=lambda r: r["query_id"])

    for out, data in [(OUT_TELEQNA_ALL, teleqna_all), (OUT_TELEQNA_TEST, teleqna_test), (OUT_TELEQUAD, telequad_all)]:
        if data:
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------
# Write helpers
# ----------------------------------------------------------------------
def append_results(filepath: str, results: List[Dict]):
    """Append results to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs via OpenRouter on telecom datasets")
    parser.add_argument("--test", action="store_true", help="Quick test: 3 models, 3 items per dataset")
    parser.add_argument("--models", type=int, default=None, help="Limit number of models to evaluate")
    parser.add_argument("--items", type=int, default=None, help="Limit items per dataset split")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="API base URL (default: OpenRouter)")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY,
                        help="API key (default: OpenRouter key)")
    args = parser.parse_args()

    init_clients(args.base_url, args.api_key)

    print("=" * 70)
    print("OpenRouter Model Evaluation — TeleQnA + TeleQuAD")
    if args.test:
        print("[TEST MODE] 3 models, 3 items per split")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    teleqna_train, teleqna_test = load_teleqna()
    telequad_items = load_telequad()

    # Apply limits
    if args.test:
        teleqna_train = teleqna_train[:3]
        teleqna_test = teleqna_test[:3]
        telequad_items = telequad_items[:3]
        models_to_run = MODELS[:3]
    else:
        if args.items:
            teleqna_train = teleqna_train[:args.items]
            teleqna_test = teleqna_test[:args.items]
            telequad_items = telequad_items[:args.items]
        models_to_run = MODELS[:args.models] if args.models else MODELS

    # Load cache if present (reloaded per-model to catch incremental saves)
    initial_cache = load_cache()
    if initial_cache:
        cached_models = list(initial_cache.keys())
        print(f"\n  Loaded initial cache with {len(cached_models)} model(s): {cached_models}")

    unavailable_models: List[str] = []
    processed_models: List[str] = []

    # Process each model
    for model in models_to_run:
        print(f"\n{'─' * 70}")
        print(f"Model: {model}")
        print(f"{'─' * 70}")

        canonical = canonical_model_name(model)

        # Reload cache from disk — captures results saved incrementally in this or prior runs
        cache = load_cache()

        # Check if already fully cached
        cached_count = len(cache.get(canonical, {}))
        total_items = len(teleqna_train) + len(teleqna_test) + len(telequad_items)
        if cached_count >= total_items:
            print(f"  [CACHE HIT] All {total_items} items already cached — skipping.")
            processed_models.append(canonical)
            continue

        # Check availability
        print("  Checking model availability...")
        if not check_model_available(model):
            print(f"  [SKIP] Model unavailable on OpenRouter.")
            unavailable_models.append(canonical)
            continue

        # TeleQnA train + test
        for split, items in [("train", teleqna_train), ("test", teleqna_test)]:
            # Filter out already-cached query_ids for this model
            cached_ids = cache.get(canonical, {})
            uncached_items = [
                (idx, item) for idx, item in enumerate(items)
                if f"teleqna_{split}_{item['global_idx']}" not in cached_ids
            ]
            cached_here = len(items) - len(uncached_items)
            if cached_here > 0:
                print(f"  [CACHE HIT] {cached_here}/{len(items)} TeleQnA {split} items already cached — skipping.")
            if not uncached_items:
                continue

            desc = f"  TeleQnA {split}"
            items_to_eval = [item for _, item in uncached_items]
            results = process_teleqna(items_to_eval, model, desc, verbose=args.test)

            # Rewrite model name to match models_info.json key
            for r in results:
                r["model"] = canonical

            # Write to output files
            append_results(OUT_TELEQNA_ALL, results)
            if split == "test":
                append_results(OUT_TELEQNA_TEST, results)

            correct_count = sum(1 for r in results if r["correct"] == 1.0)
            print(f"  TeleQnA {split}: {correct_count}/{len(results)} correct")

        # TeleQuAD
        cached_ids = cache.get(canonical, {})
        uncached_items: List[Tuple[int, Dict]] = [
            (idx, item) for idx, item in enumerate(telequad_items)
            if f"telequad_{idx}" not in cached_ids
        ]
        cached_here = len(telequad_items) - len(uncached_items)
        if cached_here > 0:
            print(f"  [CACHE HIT] {cached_here}/{len(telequad_items)} TeleQuAD items already cached — skipping.")
        if uncached_items:
            results = process_telequad(uncached_items, model, verbose=args.test)
            for r in results:
                r["model"] = canonical
            append_results(OUT_TELEQUAD, results)

            correct_count = sum(1 for r in results if r["correct"] == 1.0)
            print(f"  TeleQuAD: {correct_count}/{len(results)} correct")

        processed_models.append(canonical)

    # Rebuild output files from cache to ensure consistency
    if processed_models:
        cache = load_cache()
        rebuild_outputs_from_cache(cache, processed_models)

        # Compact cache file to remove duplicates
        cache_path = Path(CACHE_FILE)
        all_entries: List[Dict] = []
        for model_data in cache.values():
            all_entries.extend(model_data.values())
        all_entries.sort(key=lambda r: (r["model"], r["query_id"]))
        with open(cache_path, "w", encoding="utf-8") as f:
            for r in all_entries:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  Compacted cache: {len(all_entries)} unique entries → {CACHE_FILE}")

    print(f"\n{'=' * 70}")
    print("Done! Results written to:")
    print(f"  {OUT_TELEQNA_ALL}")
    print(f"  {OUT_TELEQNA_TEST}")
    print(f"  {OUT_TELEQUAD}")

    if unavailable_models:
        print(f"\n  Unavailable models ({len(unavailable_models)}):")
        for m in unavailable_models:
            print(f"    - {m}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
