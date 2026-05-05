#!/usr/bin/env python3
"""
measure_inference_time.py

从 models.txt 读取模型列表，依次使用 vLLM 推理，
从5个数据集中各抽100条样本，统计推理时间，
每完成一个模型立即更新 models_info.json 中的 co2_cost 字段。

Usage:
    python measure_inference_time.py
    python measure_inference_time.py --samples-per-dataset 50
    python measure_inference_time.py --tp 4 --gpu-util 0.95
    python measure_inference_time.py --datasets bbh,math
    python measure_inference_time.py --model Qwen/Qwen3-8B   # 只跑指定模型
"""

import os
import sys
import json
import time
import gc
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field, asdict

try:
    from datasets import load_dataset
except ImportError:
    print("Error: pip install datasets")
    sys.exit(1)

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
MODELS_TXT = SCRIPT_DIR / "models.txt"
MODELS_INFO_PATH = SCRIPT_DIR / "data" / "model_data" / "models_info.json"

# ============================================================================
# Data classes
# ============================================================================

@dataclass
class InferenceResult:
    query_id: str
    dataset: str
    prompt_length: int
    total_time: float
    output_length: int
    success: bool
    error: str = ""


@dataclass
class DatasetStats:
    name: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_total_time: float
    median_total_time: float
    std_total_time: float
    min_time: float
    max_time: float
    avg_prompt_length: float
    avg_output_length: float


@dataclass
class ModelStats:
    model_name: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_time: float
    avg_time_per_query: float
    queries_per_second: float
    dataset_stats: List[DatasetStats] = field(default_factory=list)


# ============================================================================
# Prompt templates
# ============================================================================

BBH_TEMPLATE = "{input}\n\nThink step by step, then put your final answer inside \\boxed{{}}."
MATH_TEMPLATE = "Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.\n\nProblem:\n{problem}"
MMLU_PRO_TEMPLATE = "{question}\n\n{options_formatted}\n\nReply with just the letter of the correct answer inside \\boxed{{}}. Only the letter inside the box, nothing else."
TELEQNA_TEMPLATE = "{question}\n\n{choices_formatted}\n\nReply with just the answer inside \\boxed{{}}. Only the number inside the box, nothing else."
TELEQUAD_TEMPLATE = "Context: {context}\n\nQuestion: {question}\n\nReply with just the answer inside \\boxed{{}}. Keep your answer brief and concise."

LETTERS = "ABCDEFGHIJ"

# ============================================================================
# Dataset loading
# ============================================================================

def load_bbh(limit=None):
    subsets = [
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
        'tracking_shuffled_objects_three_objects', 'web_of_lies',
    ]
    items = []
    for subset in subsets:
        try:
            ds = load_dataset("lukaemon/bbh", subset, split="test")
            for row in ds:
                items.append({
                    "prompt": BBH_TEMPLATE.format(input=row["input"]),
                    "query_id": f"bbh_{len(items)}",
                })
        except Exception as e:
            print(f"  [WARN] BBH subset '{subset}': {e}")
    print(f"  BBH: {len(items)} items")
    return items[:limit] if limit else items


def load_math(limit=None):
    ds = load_dataset("qwedsacf/competition_math", split="train")
    items = []
    for idx, row in enumerate(ds):
        if row.get("level", "") != "Level 5":
            continue
        items.append({
            "prompt": MATH_TEMPLATE.format(problem=row["problem"]),
            "query_id": f"math_{idx}",
        })
    print(f"  MATH: {len(items)} items (Level 5)")
    return items[:limit] if limit else items


def load_mmlu_pro(limit=None):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items = []
    for idx, row in enumerate(ds):
        options_formatted = "\n".join(
            f"{LETTERS[i]}. {opt}" for i, opt in enumerate(row.get("options", []))
        )
        items.append({
            "prompt": MMLU_PRO_TEMPLATE.format(
                question=row["question"], options_formatted=options_formatted
            ),
            "query_id": f"mmlu_pro_{idx}",
        })
    print(f"  MMLU-Pro: {len(items)} items")
    return items[:limit] if limit else items


def load_teleqna(limit=None):
    items = []
    for split in ["train", "test"]:
        ds = load_dataset("ymoslem/TeleQnA-processed", split=split)
        for row in ds:
            choices_formatted = "\n".join(
                f"{i}. {c}" for i, c in enumerate(row["choices"])
            )
            items.append({
                "prompt": TELEQNA_TEMPLATE.format(
                    question=row["question"], choices_formatted=choices_formatted
                ),
                "query_id": f"teleqna_{len(items)}",
            })
    print(f"  TeleQnA: {len(items)} items")
    return items[:limit] if limit else items


def load_telequad(limit=None):
    path = SCRIPT_DIR / "data" / "TeleQuAD-v4-full.json"
    if not path.exists():
        print(f"  [WARN] TeleQuAD not found at {path}")
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for doc in raw.get("data", []):
        for para in doc.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                items.append({
                    "prompt": TELEQUAD_TEMPLATE.format(
                        context=context, question=qa["question"]
                    ),
                    "query_id": f"telequad_{len(items)}",
                })
    print(f"  TeleQuAD: {len(items)} items")
    return items[:limit] if limit else items


DATASET_LOADERS = {
    "bbh": load_bbh,
    "math": load_math,
    "mmlu_pro": load_mmlu_pro,
    "teleqna": load_teleqna,
    "telequad": load_telequad,
}

# ============================================================================
# models_info.json helpers
# ============================================================================

def load_models_info() -> List[Dict]:
    if not MODELS_INFO_PATH.exists():
        print(f"[WARN] {MODELS_INFO_PATH} not found")
        return []
    with open(MODELS_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_models_info(data: List[Dict]):
    MODELS_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_co2_cost(model_name: str, avg_time_per_query: float):
    """Update co2_cost for a model in models_info.json.

    We store avg_time_per_query (seconds) as the cost proxy.
    A real CO2 conversion can be added later.
    """
    data = load_models_info()
    updated = False
    for entry in data:
        if entry["name"] == model_name:
            entry["co2_cost"] = round(avg_time_per_query, 6)
            updated = True
            print(f"  [Updated] {model_name}: co2_cost = {avg_time_per_query:.4f}s/query")
            break
    if updated:
        save_models_info(data)
    else:
        print(f"  [WARN] {model_name} not found in models_info.json, skipping update")


def read_models_from_txt() -> List[str]:
    if not MODELS_TXT.exists():
        print(f"[ERROR] {MODELS_TXT} not found")
        sys.exit(1)
    models = []
    with open(MODELS_TXT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                models.append(line)
    return models


def get_pending_models(models: List[str]) -> List[str]:
    """Return models that haven't been measured yet (co2_cost is null)."""
    data = load_models_info()
    info_map = {entry["name"]: entry for entry in data}
    pending = []
    for m in models:
        if m in info_map and info_map[m].get("co2_cost") is not None:
            print(f"  [Skip] {m}: already measured (co2_cost={info_map[m]['co2_cost']})")
        else:
            pending.append(m)
    return pending

# ============================================================================
# Statistics
# ============================================================================

def compute_dataset_stats(results, dataset_name):
    ok = [r for r in results if r.success]
    times = [r.total_time for r in ok if r.total_time > 0]
    prompt_lens = [r.prompt_length for r in ok]
    output_lens = [r.output_length for r in ok]
    return DatasetStats(
        name=dataset_name,
        total_queries=len(results),
        successful_queries=len(ok),
        failed_queries=len(results) - len(ok),
        avg_total_time=statistics.mean(times) if times else 0.0,
        median_total_time=statistics.median(times) if times else 0.0,
        std_total_time=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_time=min(times) if times else 0.0,
        max_time=max(times) if times else 0.0,
        avg_prompt_length=statistics.mean(prompt_lens) if prompt_lens else 0.0,
        avg_output_length=statistics.mean(output_lens) if output_lens else 0.0,
    )


def format_stats(stats, load_time, total_infer):
    lines = []
    lines.append("=" * 80)
    lines.append(f"  {stats.model_name}")
    lines.append("=" * 80)
    lines.append(f"  Load time:     {load_time:.1f}s")
    lines.append(f"  Infer time:    {total_infer:.1f}s")
    lines.append(f"  Queries:       {stats.successful_queries}/{stats.total_queries}")
    lines.append(f"  Avg/query:     {stats.avg_time_per_query:.3f}s")
    lines.append(f"  Throughput:    {stats.queries_per_second:.2f} q/s")
    lines.append("-" * 80)
    lines.append(f"  {'Dataset':<12} {'N':>5} {'Avg(s)':>8} {'Med(s)':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    lines.append("-" * 80)
    for ds in stats.dataset_stats:
        lines.append(
            f"  {ds.name:<12} {ds.successful_queries:>5} "
            f"{ds.avg_total_time:>8.3f} {ds.median_total_time:>8.3f} "
            f"{ds.std_total_time:>8.3f} {ds.min_time:>8.3f} {ds.max_time:>8.3f}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def save_results(stats, results_by_dataset, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_name = stats.model_name.replace("/", "_")

    summary = {
        "model_name": stats.model_name,
        "total_queries": stats.total_queries,
        "successful_queries": stats.successful_queries,
        "total_infer_time": stats.total_time,
        "avg_time_per_query": stats.avg_time_per_query,
        "queries_per_second": stats.queries_per_second,
        "datasets": {ds.name: asdict(ds) for ds in stats.dataset_stats},
    }
    stats_path = out / f"{safe_name}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    detail_path = out / f"{safe_name}_details.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for ds_name, results in results_by_dataset.items():
            for r in results:
                entry = asdict(r)
                entry["dataset"] = ds_name
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return str(stats_path), str(detail_path)

# ============================================================================
# Run one model
# ============================================================================

def run_model(model_name, datasets, tp, gpu_util, max_model_len, per_sample, output_dir):
    """Load model, run inference on all datasets, return (stats, load_time, total_infer)."""
    from vllm import LLM, SamplingParams

    print(f"\n{'#' * 80}")
    print(f"  Loading: {model_name}")
    print(f"{'#' * 80}")

    load_start = time.time()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    load_time = time.time() - load_start
    print(f"  Loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    results_by_dataset = {}
    all_infer_start = time.time()

    for ds_name, items in datasets.items():
        if not items:
            continue
        prompts_text = [it["prompt"] for it in items]
        query_ids = [it["query_id"] for it in items]

        if per_sample:
            print(f"\n  [{ds_name}] {len(prompts_text)} queries (per-sample)...")
            ds_results = []
            for j, (qid, prompt) in enumerate(zip(query_ids, prompts_text)):
                t0 = time.time()
                out = llm.generate([prompt], sampling_params)
                elapsed = time.time() - t0
                text = out[0].outputs[0].text if out[0].outputs else ""
                ds_results.append(InferenceResult(
                    query_id=qid, dataset=ds_name,
                    prompt_length=len(prompt), total_time=elapsed,
                    output_length=len(text), success=True,
                ))
                if (j + 1) % 20 == 0:
                    print(f"    {j+1}/{len(prompts_text)} done")
            results_by_dataset[ds_name] = ds_results
        else:
            print(f"\n  [{ds_name}] {len(prompts_text)} queries (batch)...")
            t0 = time.time()
            outputs = llm.generate(prompts_text, sampling_params, use_tqdm=True)
            elapsed = time.time() - t0
            per_query_time = elapsed / len(prompts_text) if prompts_text else 0
            ds_results = []
            for j, output in enumerate(outputs):
                text = output.outputs[0].text if output.outputs else ""
                ds_results.append(InferenceResult(
                    query_id=query_ids[j], dataset=ds_name,
                    prompt_length=len(prompts_text[j]),
                    total_time=per_query_time,
                    output_length=len(text), success=True,
                ))
            results_by_dataset[ds_name] = ds_results
            print(f"    {ds_name}: {elapsed:.2f}s total, {per_query_time:.3f}s/query")

    total_infer = time.time() - all_infer_start

    # Cleanup GPU
    del llm
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute stats
    all_results = []
    for r_list in results_by_dataset.values():
        all_results.extend(r_list)
    ok = [r for r in all_results if r.success]
    times = [r.total_time for r in ok if r.total_time > 0]
    total = sum(times) if times else 0.0
    dataset_stats = [compute_dataset_stats(r, n) for n, r in results_by_dataset.items()]

    stats = ModelStats(
        model_name=model_name,
        total_queries=len(all_results),
        successful_queries=len(ok),
        failed_queries=len(all_results) - len(ok),
        total_time=total,
        avg_time_per_query=total / len(ok) if ok else 0.0,
        queries_per_second=len(ok) / total if total > 0 else 0.0,
        dataset_stats=dataset_stats,
    )

    print(format_stats(stats, load_time, total_infer))
    save_results(stats, results_by_dataset, output_dir)

    return stats, load_time, total_infer

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure LLM inference time via vLLM, update models_info.json"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Run a single model (default: read from models.txt)")
    parser.add_argument("--samples-per-dataset", type=int, default=20)
    parser.add_argument("--datasets", type=str, default="all",
                        help="bbh,math,mmlu_pro,teleqna,telequad (default: all)")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--per-sample", action="store_true",
                        help="Per-sample timing (slower, more detailed)")
    parser.add_argument("--output-dir", type=str, default="data/inference_time")
    parser.add_argument("--force", action="store_true",
                        help="Re-measure even if co2_cost already set")
    args = parser.parse_args()

    # Determine model list
    if args.model:
        models = [args.model]
    else:
        print(f"Reading models from {MODELS_TXT} ...")
        models = read_models_from_txt()
    print(f"Models: {len(models)}")
    for m in models:
        print(f"  - {m}")

    # Filter already-measured models
    if not args.force:
        models = get_pending_models(models)
    if not models:
        print("No models to measure.")
        return

    print(f"\nPending models: {len(models)}")
    for m in models:
        print(f"  - {m}")

    # Determine datasets
    if args.datasets == "all":
        ds_names = ["bbh", "math", "mmlu_pro", "teleqna", "telequad"]
    else:
        ds_names = [d.strip() for d in args.datasets.split(",")]

    # Load datasets once (shared across all models)
    print("\nLoading datasets...")
    datasets = {}
    for name in ds_names:
        loader = DATASET_LOADERS.get(name)
        if loader:
            datasets[name] = loader(limit=args.samples_per_dataset)

    total_queries_per_model = sum(len(v) for v in datasets.values())
    print(f"\nTotal: {len(datasets)} datasets, {total_queries_per_model} queries/model")

    # Run each model
    print(f"\n{'=' * 80}")
    print(f"  Starting inference for {len(models)} model(s)")
    print(f"{'=' * 80}")

    for i, model_name in enumerate(models):
        print(f"\n>>> Model [{i+1}/{len(models)}]: {model_name}")

        try:
            stats, load_time, total_infer = run_model(
                model_name, datasets, args.tp, args.gpu_util,
                args.max_model_len, args.per_sample, args.output_dir,
            )

            # Update models_info.json immediately
            update_co2_cost(model_name, stats.avg_time_per_query)

        except Exception as e:
            print(f"\n  [ERROR] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            # Try to clean up GPU even on failure
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    # Final summary
    print(f"\n{'=' * 80}")
    print("  Done! Updated models_info.json:")
    data = load_models_info()
    for entry in data:
        cost = entry.get("co2_cost")
        cost_str = f"{cost:.4f}s/q" if cost is not None else "null"
        print(f"    {entry['name']:<50} {cost_str}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
