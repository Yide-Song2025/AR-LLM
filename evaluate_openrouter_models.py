import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

from openai import OpenAI, RateLimitError, APIError, BadRequestError
from datasets import load_dataset, get_dataset_config_names

# Map OpenRouter model names to their models_info.json keys
OPENROUTER_TO_MODELS_INFO = {
    "qwen/qwen3-8b": {"name":"Qwen/Qwen3-8B", "provider":"alibaba"},
    "qwen/qwen3-14b": {"name":"Qwen/Qwen3-14B", "provider":"alibaba"},
    # "qwen/qwen3-32b": {"name": "Qwen/Qwen3-32B", "provider":"alibaba"},
    # "qwen/qwen3.5-9b": "Qwen/Qwen3.5-9B",
    # "qwen/qwen3.5-27b": {"name": "Qwen/Qwen3.5-27B", "provider":"alibaba"},
    # "qwen/qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    # "qwen/qwen3.5-122b-a10b": {"name": "Qwen/Qwen3.5-122B-A10B", "provider":"alibaba"},
    # "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    # "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/llama-4-scout": "meta-llama/Llama-4-Scout",
    # "google/gemma-4-31b-it": "google/gemma-4-31B",
    # "google/gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B"
}

MODELS = list(OPENROUTER_TO_MODELS_INFO.keys())

BBH_SUBSETS = [
    'boolean_expressions', 'causal_judgement', 'date_understanding',
    'disambiguation_qa', 'formal_fallacies',
    'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
    'logical_deduction_seven_objects', 'logical_deduction_three_objects',
    'movie_recommendation', 'navigate',
    'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
    'ruin_names', 'salient_translation_error_detection', 'snarks',
    'sports_understanding', 'temporal_sequences',
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects', 'web_of_lies'
]


def _get_model_config(openrouter_model: str) -> Tuple[str, Optional[str]]:
    """Return (canonical_name, provider_or_None) for a model."""
    val = OPENROUTER_TO_MODELS_INFO.get(openrouter_model, openrouter_model)
    if isinstance(val, dict):
        return val["name"], val.get("provider")
    return val, None


def canonical_model_name(openrouter_model: str) -> str:
    """Return the models_info key, falling back to the input name."""
    return _get_model_config(openrouter_model)[0]


root = 'data'
# ----------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------
OUT_TELE     = root + "/raw_data/tele_openrouter_results.jsonl"
OUT_MATH     = root + "/raw_data/math_openrouter_results.jsonl"
OUT_BBH      = root + "/raw_data/bbh_openrouter_results.jsonl"
OUT_MMLU_PRO = root + "/raw_data/mmlu_pro_openrouter_results.jsonl"

# Temporary cache file for interrupt recovery
CACHE_FILE = root + "/raw_data/openrouter_eval_cache.jsonl"

MAX_WORKERS = 512

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

Reply with just the answer inside \\boxed{{}}. Only the number inside the box, nothing else."""

TELEQUAD_TEMPLATE = """Context: {context}

Question: {question}

Reply with just the answer inside \\boxed{{}}. Keep your answer brief and concise."""

MATH_TEMPLATE = """Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.

Problem:
{problem}"""

BBH_TEMPLATE = """{input}

Think step by step, then put your final answer inside \\boxed{{}}."""

MMLU_PRO_TEMPLATE = """{question}

{options_formatted}

Reply with just the letter of the correct answer inside \\boxed{{}}. Only the letter inside the box, nothing else."""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content inside \\boxed{...} from model output."""
    match = re.search(r'[/\\]box(?:ed)?\{([^}]*)\}', text)
    return match.group(1).strip() if match else None


def extract_math_answer_from_solution(solution: str) -> Optional[str]:
    """Extract the final boxed answer from a MATH solution (handles nested braces)."""
    idx = solution.rfind('\\boxed{')
    if idx == -1:
        return None
    start = idx + len('\\boxed{')
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == '{':
            depth += 1
        elif solution[i] == '}':
            depth -= 1
        i += 1
    return solution[start:i - 1].strip()


def call_openrouter(model: str, prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call OpenRouter API with exponential backoff."""
    _, provider = _get_model_config(model)
    provider_prefs = {"provider": {"only": [provider]}} if provider else None
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
            )
            if provider_prefs:
                kwargs["extra_body"] = provider_prefs
            response = openrouter_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except BadRequestError as e:
            if attempt == 0:
                print(f"    [WARN] BadRequestError for {model}: {e}")
            return None
        except (RateLimitError, APIError) as e:
            wait = min(2 ** (attempt + 1), 30)
            print(f"    [WARN] {type(e).__name__} for {model} (attempt {attempt+1}/{max_retries}), retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = min(2 ** (attempt + 1), 30)
            print(f"    [WARN] Unexpected error for {model}: {e} (attempt {attempt+1}/{max_retries}), retrying in {wait}s")
            time.sleep(wait)
    return None


def check_model_available(model: str) -> bool:
    """Probe model with a trivial prompt to check availability."""
    _, provider = _get_model_config(model)
    provider_prefs = {"provider": {"only": [provider]}} if provider else None
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
            max_tokens=5,
        )
        if provider_prefs:
            kwargs["extra_body"] = provider_prefs
        response = openrouter_client.chat.completions.create(**kwargs)
        return True
    except BadRequestError:
        return False
    except Exception:
        return True  # Assume available on other errors


def judge_with_deepseek(question: str, reference: str, response_text: str,
                        max_retries: int = 3, verbose: bool = False) -> int:
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
def load_teleqna() -> List[Dict]:
    """Load TeleQnA train + test from HuggingFace."""
    items = []
    for split in ["train", "test"]:
        ds = load_dataset("ymoslem/TeleQnA-processed", split=split)
        for row in ds:
            items.append({
                "question": row["question"],
                "choices": row["choices"],
                "answer_idx": int(row["answer"]),
                "subject": row["subject"],
                "explanation": row.get("explanation", ""),
                "source": "teleqna",
                "orig_split": split,
            })
    print(f"  Loaded TeleQnA: {len(items)} items")
    return items


def load_telequad() -> List[Dict]:
    """Parse TeleQuAD from local JSON."""
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
                    "source": "telequad",
                })
    print(f"  Loaded TeleQuAD: {len(items)} questions")
    return items


def load_tele() -> List[Dict]:
    """Load unified tele dataset: TeleQnA + TeleQuAD with sequential tele_ IDs."""
    teleqna = load_teleqna()
    telequad = load_telequad()
    all_items = teleqna + telequad
    for idx, item in enumerate(all_items):
        item["global_idx"] = idx
    print(f"  Unified tele: {len(all_items)} items ({len(teleqna)} TeleQnA + {len(telequad)} TeleQuAD)")
    return all_items


def load_math() -> List[Dict]:
    """Load MATH Level 5 from HuggingFace."""
    ds = load_dataset("qwedsacf/competition_math", split="train")
    items = []
    for idx, row in enumerate(ds):
        if row.get("level", "") != "Level 5":
            continue
        answer = extract_math_answer_from_solution(row.get("solution", ""))
        items.append({
            "problem": row["problem"],
            "solution": row.get("solution", ""),
            "ref_answer": answer or "",
            "level": row.get("level", ""),
            "type": row.get("type", ""),
            "global_idx": idx,
        })
    print(f"  Loaded MATH Level 5: {len(items)} items")
    return items


def load_bbh() -> List[Dict]:
    """Load BBH from HuggingFace (predefined subsets)."""
    items = []
    idx = 0
    for subset in BBH_SUBSETS:
        try:
            ds = load_dataset("lukaemon/bbh", subset, split="test")
            for row in ds:
                items.append({
                    "input": row["input"],
                    "target": row["target"],
                    "subset": subset,
                    "global_idx": idx,
                })
                idx += 1
        except Exception as e:
            print(f"    [WARN] Failed to load BBH subset '{subset}': {e}")
    print(f"  Loaded BBH: {len(items)} items ({len(BBH_SUBSETS)} subsets)")
    return items


def load_mmlu_pro() -> List[Dict]:
    """Load MMLU-Pro from HuggingFace."""
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items = []
    for idx, row in enumerate(ds):
        answer_raw = row.get("answer", 0)
        if isinstance(answer_raw, str):
            answer_idx = "ABCDEFGHIJ".find(answer_raw.upper())
            if answer_idx == -1:
                try:
                    answer_idx = int(answer_raw)
                except ValueError:
                    answer_idx = 0
        else:
            answer_idx = int(answer_raw)
        items.append({
            "question": row["question"],
            "options": row.get("options", []),
            "answer_idx": answer_idx,
            "category": row.get("category", ""),
            "global_idx": idx,
        })
    print(f"  Loaded MMLU-Pro: {len(items)} items")
    return items


# ----------------------------------------------------------------------
# Evaluation logic
# ----------------------------------------------------------------------
def evaluate_tele_item(item: Dict, model: str, verbose: bool = False) -> Optional[Dict]:
    """Evaluate a unified tele item — dispatches to TeleQnA or TeleQuAD logic."""
    if item["source"] == "teleqna":
        return _evaluate_teleqna(item, model, verbose)
    else:
        return _evaluate_telequad(item, model, verbose)


def _evaluate_teleqna(item: Dict, model: str, verbose: bool = False) -> Optional[Dict]:
    choices = item["choices"]
    choices_formatted = "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
    prompt = TELEQNA_TEMPLATE.format(
        question=item["question"],
        choices_formatted=choices_formatted,
    )

    response_text = call_openrouter(model, prompt)
    if response_text is None:
        return None

    query_id = f"tele_{item['global_idx']}"
    ground_truth = choices[item["answer_idx"]]
    boxed = extract_boxed_answer(response_text)

    if verbose:
        print(f"\n    [PROMPT]\n{prompt}")
        print(f"    [RESPONSE] {response_text}")
        print(f"    [GT] {ground_truth}")
        print(f"    [BOXED] {boxed}") if boxed is not None else print("    [BOXED] None")

    correct = 0.0
    if boxed is not None:
        try:
            idx = int(boxed)
            if 0 <= idx < len(choices) and idx == item["answer_idx"]:
                correct = 1.0
        except ValueError:
            pass
        if correct == 0.0:
            if boxed.strip().lower() == ground_truth.strip().lower():
                correct = 1.0

    return {
        "query": item["question"],
        "answer": ground_truth,
        "model": model,
        "dataset": "tele",
        "subset": item.get("subject", ""),
        "correct": correct,
        "query_id": query_id,
        "source": "teleqna",
    }


def _evaluate_telequad(item: Dict, model: str, verbose: bool = False) -> Optional[Dict]:
    prompt = TELEQUAD_TEMPLATE.format(
        context=item["context"],
        question=item["question"],
    )

    response_text = call_openrouter(model, prompt)
    if response_text is None:
        return None

    query_id = f"tele_{item['global_idx']}"
    boxed = extract_boxed_answer(response_text)

    if verbose:
        print(f"\n    [PROMPT]\n{prompt}")
        print(f"    [RESPONSE] {response_text}")
        print(f"    [GT] {item['ref_answer']}")
        print(f"    [BOXED] {boxed}") if boxed is not None else print("    [BOXED] None")

    if boxed is not None:
        response_for_judge = boxed
    else:
        parts = response_text.rsplit("\n", 1)
        response_for_judge = parts[-1] if len(parts) > 1 else response_text

    correct = judge_with_deepseek(
        item["question"], item["ref_answer"], response_for_judge, verbose=verbose
    ) if response_for_judge else 0

    return {
        "query": item["question"],
        "answer": item["ref_answer"],
        "model": model,
        "dataset": "tele",
        "subset": item.get("source", "telequad"),
        "correct": float(correct),
        "query_id": query_id,
        "source": "telequad",
    }


def evaluate_math_item(item: Dict, model: str, verbose: bool = False) -> Optional[Dict]:
    """Evaluate a MATH item via direct text matching."""
    prompt = MATH_TEMPLATE.format(problem=item["problem"])

    response_text = call_openrouter(model, prompt)
    if response_text is None:
        return None

    query_id = f"math_{item['global_idx']}"
    boxed = extract_boxed_answer(response_text)
    ref_answer = item["ref_answer"]

    if verbose:
        print(f"\n    [PROMPT]\n{prompt}")
        print(f"    [RESPONSE] {response_text}")
        print(f"    [GT] {ref_answer}")
        print(f"    [BOXED] {boxed}") if boxed is not None else print("    [BOXED] None")

    correct = 0.0
    if boxed is not None:
        boxed_stripped = boxed.strip()
        if boxed_stripped.lower() == ref_answer.lower():
            correct = 1.0
        elif boxed_stripped.strip("()").lower() == ref_answer.strip().strip("()").lower():
            correct = 1.0
        else:
            try:
                if float(boxed_stripped) == float(ref_answer):
                    correct = 1.0
            except (ValueError, TypeError):
                pass

    return {
        "query": item["problem"],
        "answer": ref_answer,
        "model": model,
        "dataset": "math",
        "subset": item.get("type", ""),
        "correct": float(correct),
        "query_id": query_id,
    }


def evaluate_bbh_item(item: Dict, model: str, verbose: bool = False) -> Optional[Dict]:
    """Evaluate a BBH item via direct text matching."""
    prompt = BBH_TEMPLATE.format(input=item["input"])

    response_text = call_openrouter(model, prompt)
    if response_text is None:
        return None

    query_id = f"bbh_{item['global_idx']}"
    boxed = extract_boxed_answer(response_text)
    target = item["target"]

    if verbose:
        print(f"\n    [PROMPT]\n{prompt}")
        print(f"    [RESPONSE] {response_text}")
        print(f"    [GT] {target}")
        print(f"    [BOXED] {boxed}") if boxed is not None else print("    [BOXED] None")

    correct = 0.0
    if boxed is not None:
        target_clean = target.strip().lower()
        boxed_clean = boxed.strip().lower()
        if boxed_clean == target_clean or boxed_clean.strip("()") == target_clean.strip("()"):
            correct = 1.0
        elif target_clean in boxed_clean:
            correct = 1.0

    return {
        "query": item["input"],
        "answer": target,
        "model": model,
        "dataset": "bbh",
        "subset": item.get("subset", ""),
        "correct": correct,
        "query_id": query_id,
    }


LETTERS = "ABCDEFGHIJ"


def evaluate_mmlu_pro_item(item: Dict, model: str, verbose: bool = False) -> Optional[Dict]:
    """Evaluate a MMLU-Pro multiple-choice item."""
    options = item["options"]
    options_formatted = "\n".join(
        f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    prompt = MMLU_PRO_TEMPLATE.format(
        question=item["question"],
        options_formatted=options_formatted,
    )

    response_text = call_openrouter(model, prompt)
    if response_text is None:
        return None

    query_id = f"mmlu_pro_{item['global_idx']}"
    boxed = extract_boxed_answer(response_text)
    answer_idx = item["answer_idx"]
    ground_truth_letter = LETTERS[answer_idx]
    ground_truth_text = options[answer_idx] if answer_idx < len(options) else ""

    if verbose:
        print(f"\n    [PROMPT]\n{prompt}")
        print(f"    [RESPONSE] {response_text}")
        print(f"    [GT] {ground_truth_letter}. {ground_truth_text}")
        print(f"    [BOXED] {boxed}") if boxed is not None else print("    [BOXED] None")

    correct = 0.0
    if boxed is not None:
        boxed_stripped = boxed.strip()
        if boxed_stripped.upper() == ground_truth_letter:
            correct = 1.0
        elif boxed_stripped.strip("()").lower() == ground_truth_text.strip().lower():
            correct = 1.0
        else:
            try:
                idx = int(boxed_stripped)
                if idx == answer_idx:
                    correct = 1.0
            except ValueError:
                pass

    return {
        "query": item["question"],
        "answer": f"{ground_truth_letter}. {ground_truth_text}",
        "model": model,
        "dataset": "mmlu_pro",
        "subset": item.get("category", ""),
        "correct": correct,
        "query_id": query_id,
    }


# ----------------------------------------------------------------------
# Parallel evaluation helpers
# ----------------------------------------------------------------------
def _tele_worker(args: Tuple[Dict, str, bool]) -> Optional[Dict]:
    item, model, verbose = args
    return evaluate_tele_item(item, model, verbose)


def _math_worker(args: Tuple[Dict, str, bool]) -> Optional[Dict]:
    item, model, verbose = args
    return evaluate_math_item(item, model, verbose)


def _bbh_worker(args: Tuple[Dict, str, bool]) -> Optional[Dict]:
    item, model, verbose = args
    return evaluate_bbh_item(item, model, verbose)


def _mmlu_pro_worker(args: Tuple[Dict, str, bool]) -> Optional[Dict]:
    item, model, verbose = args
    return evaluate_mmlu_pro_item(item, model, verbose)


def process_items(items: List[Dict], model: str, dataset_name: str,
                  worker_fn, verbose: bool = False, save_every: int = 10,
                  max_workers: int = None, cache_lock: threading.Lock = None) -> List[Dict]:
    """Generic parallel evaluation with incremental cache saving."""
    results: List[Dict] = []
    pending: List[Dict] = []
    args_list = [(item, model, verbose) for item in items]
    workers = max_workers or MAX_WORKERS
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_fn, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"  {dataset_name}", leave=False):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    pending.append(result)
                    if len(pending) >= save_every:
                        if cache_lock:
                            with cache_lock:
                                save_to_cache(pending)
                        else:
                            save_to_cache(pending)
                        pending = []
            except Exception as e:
                print(f"    [ERROR] Worker exception: {e}")
    if pending:
        if cache_lock:
            with cache_lock:
                save_to_cache(pending)
        else:
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
    """Rebuild all output files from cache, only for available models."""
    datasets: Dict[str, List[Dict]] = {"tele": [], "math": [], "bbh": [], "mmlu_pro": []}
    output_files = {
        "tele": OUT_TELE,
        "math": OUT_MATH,
        "bbh": OUT_BBH,
        "mmlu_pro": OUT_MMLU_PRO,
    }

    for model in available_models:
        if model not in cache:
            continue
        for r in cache[model].values():
            ds = r.get("dataset", "")
            if ds in datasets:
                datasets[ds].append(r)

    for ds, data in datasets.items():
        if data:
            data.sort(key=lambda r: r["query_id"])
            path = Path(output_files[ds])
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
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs via OpenRouter on Tele + MATH + BBH + MMLU-Pro"
    )
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 3 models, 3 items per dataset")
    parser.add_argument("--models", type=int, default=None,
                        help="Limit number of models to evaluate")
    parser.add_argument("--parallel-models", type=int, default=2,
                        help="Number of models to evaluate concurrently (default: 1)")
    parser.add_argument("--items", type=int, default=None,
                        help="Limit items per dataset")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "tele", "math", "bbh", "mmlu_pro"],
                        help="Which dataset(s) to evaluate (default: all)")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="API base URL (default: OpenRouter)")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY,
                        help="API key (default: OpenRouter key)")
    args = parser.parse_args()

    init_clients(args.base_url, args.api_key)

    print("=" * 70)
    print("OpenRouter Model Evaluation — Tele + MATH + BBH + MMLU-Pro")
    if args.test:
        print("[TEST MODE] 3 models, 3 items per dataset")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    all_datasets: Dict[str, List[Dict]] = {}
    if args.dataset in ("all", "tele"):
        all_datasets["tele"] = load_tele()
    if args.dataset in ("all", "math"):
        all_datasets["math"] = load_math()
    if args.dataset in ("all", "bbh"):
        all_datasets["bbh"] = load_bbh()
    if args.dataset in ("all", "mmlu_pro"):
        all_datasets["mmlu_pro"] = load_mmlu_pro()

    # Apply limits
    if args.test:
        for name in all_datasets:
            all_datasets[name] = all_datasets[name][:3]
        models_to_run = MODELS[:3]
    else:
        if args.items:
            for name in all_datasets:
                all_datasets[name] = all_datasets[name][:args.items]
        models_to_run = MODELS[:args.models] if args.models else MODELS

    # Dataset config: output file + worker function
    dataset_config = {
        "tele":     {"output": OUT_TELE,     "worker": _tele_worker},
        "math":     {"output": OUT_MATH,     "worker": _math_worker},
        "bbh":      {"output": OUT_BBH,      "worker": _bbh_worker},
        "mmlu_pro": {"output": OUT_MMLU_PRO, "worker": _mmlu_pro_worker},
    }

    initial_cache = load_cache()
    if initial_cache:
        cached_models = list(initial_cache.keys())
        print(f"\n  Loaded initial cache with {len(cached_models)} model(s): {cached_models}")

    unavailable_models: List[str] = []
    processed_models: List[str] = []
    cache_lock = threading.Lock()
    parallel = args.parallel_models
    inner_workers = max(1, MAX_WORKERS // parallel)

    def _evaluate_model_task(model: str) -> Tuple[str, bool]:
        """Evaluate a single model on all datasets. Thread-safe."""
        canonical = canonical_model_name(model)

        cache = load_cache()
        cached_count = len(cache.get(canonical, {}))
        total_items = sum(len(items) for items in all_datasets.values())
        if cached_count >= total_items:
            print(f"  [{model}] All {total_items} items cached — skipping.")
            return canonical, True

        print(f"  Checking {model} availability...")
        if not check_model_available(model):
            print(f"  [{model}] Unavailable — skipping.")
            return canonical, False

        for ds_name, items in all_datasets.items():
            cfg = dataset_config[ds_name]
            cache = load_cache()
            cached_ids = cache.get(canonical, {})

            uncached_items = [
                item for item in items
                if f"{ds_name}_{item['global_idx']}" not in cached_ids
            ]
            cached_here = len(items) - len(uncached_items)
            if cached_here > 0:
                print(f"  [{model}] {cached_here}/{len(items)} {ds_name} cached — skipping.")
            if not uncached_items:
                continue

            results = process_items(
                uncached_items, model, f"{model}/{ds_name.upper()}", cfg["worker"],
                verbose=args.test, max_workers=inner_workers, cache_lock=cache_lock,
            )
            for r in results:
                r["model"] = canonical

            correct_count = sum(1 for r in results if r["correct"] == 1.0)
            print(f"  [{model}] {ds_name.upper()}: {correct_count}/{len(results)} correct")

        return canonical, True

    if parallel <= 1:
        for model in models_to_run:
            print(f"\n{'─' * 70}")
            print(f"Model: {model}")
            print(f"{'─' * 70}")
            canonical, available = _evaluate_model_task(model)
            (processed_models if available else unavailable_models).append(canonical)
    else:
        print(f"\n  Parallel mode: {parallel} models concurrently "
              f"({inner_workers} workers/model)")
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(_evaluate_model_task, model): model
                for model in models_to_run
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Models", leave=False):
                canonical, available = future.result()
                (processed_models if available else unavailable_models).append(canonical)

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
    for ds_name in all_datasets:
        print(f"  {dataset_config[ds_name]['output']}")

    if unavailable_models:
        print(f"\n  Unavailable models ({len(unavailable_models)}):")
        for m in unavailable_models:
            print(f"    - {m}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
