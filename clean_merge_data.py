"""
Clean and merge extracted_dataset_samples.jsonl + openrouter_eval_cache.jsonl.

- Keeps only the 16 target models from models.txt
- Normalizes model names (strips -Instruct, case-insensitive match)
- Normalizes dataset names (file2 "tele" → "teleqna" or "telequad" by subset)
- Matches queries across files by (query_text, dataset) ignoring subset differences
- Deduplicates (query, dataset, model) entries, keeping first occurrence
- Retains only queries with all 16 model results
- Assigns unified sequential query_ids per dataset
"""

import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ── 1. Load target models ────────────────────────────────────────────
target_models = []
with open(BASE / "models.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        target_models.append(parts[-1].strip())

target_set = set(target_models)
print(f"Target models ({len(target_models)}): {target_models}")


# ── 2. Model name normalization ──────────────────────────────────────
_lower_map = {m.lower(): m for m in target_set}


def normalize_model(name: str) -> str | None:
    if name in target_set:
        return name
    for suffix in ("-Instruct",):
        if name.endswith(suffix):
            stripped = name[: -len(suffix)]
            if stripped in target_set:
                return stripped
    if name.lower() in _lower_map:
        return _lower_map[name.lower()]
    return None


# ── 3. Dataset normalization ─────────────────────────────────────────


def normalize_dataset(dataset: str, subset: str) -> str:
    if dataset in ("tele", "teleqna", "telequad"):
        return "tele"
    return dataset


# ── 4. Read and normalize records ────────────────────────────────────
def read_file(path: Path) -> list[dict]:
    records = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            model = normalize_model(d.get("model", ""))
            if model is None:
                skipped += 1
                continue

            dataset = normalize_dataset(d.get("dataset", ""), d.get("subset", ""))
            records.append({
                "query": d["query"],
                "answer": d["answer"],
                "model": model,
                "dataset": dataset,
                "subset": d.get("subset") or "",
                "correct": d["correct"],
            })

    print(f"  {path.name}: {len(records)} kept, {skipped} skipped")
    return records


print("\n── Reading files ──")
all_records = (
    read_file(BASE / "data" / "model_data" / "extracted_dataset_samples.jsonl")
    + read_file(BASE / "data" / "raw_data" / "openrouter_eval_cache.jsonl")
)
print(f"  Total: {len(all_records)}")


# ── 5. Group by (query, dataset); dedupe per model ───────────────────
print("\n── Grouping queries by (query_text, dataset) ──")

# groups: key → { model_name → (record_dict, subset) }
groups: dict[tuple, dict[str, tuple[dict, str]]] = defaultdict(dict)

for r in all_records:
    key = (r["query"], r["dataset"])
    model = r["model"]
    if model not in groups[key]:
        groups[key][model] = (r, r["subset"])

print(f"  Unique groups: {len(groups)}")

# Pick the most informative subset for each group (prefer non-empty)
def pick_subset(entries: dict[str, tuple[dict, str]]) -> str:
    """Return the best subset from the group's entries."""
    subsets = set(s for _, s in entries.values())
    subsets.discard("")
    if not subsets:
        return ""
    # Prefer file1-style detailed subsets over file2-style generic ones
    # Just return the first non-empty one
    return next(iter(subsets))


complete = {}
incomplete_counts = defaultdict(int)
for key, entries in groups.items():
    if len(entries) == 16:
        subset = pick_subset(entries)
        complete[key] = (entries, subset)
    else:
        incomplete_counts[len(entries)] += 1

print(f"  Complete (16 models): {len(complete)}")
print(f"  Incomplete: {dict(sorted(incomplete_counts.items()))}")

# Breakdown by dataset
ds_counts = defaultdict(int)
for (query, ds) in complete:
    ds_counts[ds] += 1
print(f"  By dataset: {dict(sorted(ds_counts.items()))}")


# ── 6. Assign unified query_ids ──────────────────────────────────────
sorted_keys = sorted(complete.keys(), key=lambda k: (k[1], k[0]))

id_counters: dict[str, int] = defaultdict(int)
query_id_map: dict[tuple, str] = {}

for key in sorted_keys:
    dataset = key[1]
    idx = id_counters[dataset]
    query_id_map[key] = f"{dataset}_q{idx}"
    id_counters[dataset] += 1

print(f"\n── Assigned unified query_ids ──")
for ds, cnt in sorted(id_counters.items()):
    print(f"  {ds}: {cnt} queries")


# ── 7. Write output ──────────────────────────────────────────────────
output_path = BASE / "data" / "model_data" / "merged_16model_results.jsonl"
written = 0
with open(output_path, "w", encoding="utf-8") as f:
    for key in sorted_keys:
        qid = query_id_map[key]
        entries, subset = complete[key]
        for model_name in target_models:
            rec = entries[model_name][0]
            out = {
                "query": rec["query"],
                "answer": rec["answer"],
                "model": rec["model"],
                "dataset": rec["dataset"],
                "subset": subset,
                "correct": rec["correct"],
                "query_id": qid,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

print(f"\n── Done ──")
print(f"  Output: {output_path}")
print(f"  Queries: {len(complete)}")
print(f"  Records: {written} ({len(complete)} queries x 16 models)")
