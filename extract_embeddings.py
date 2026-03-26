"""
Extract query embeddings and model embeddings using google/embeddinggemma-300m.

Query embeddings: encode raw query text -> mean-pooled hidden states (768-dim).
Model embeddings: compose a textual description of each model (name + per-dataset
                  performance) -> encode -> mean-pooled hidden states (768-dim).

Outputs
-------
  data/embeddings/query_embeddings.jsonl   – {query_id, dataset, query, embedding}
  data/embeddings/model_embeddings.json    – {model_name: embedding}
"""

import os
import json
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def mean_pool(last_hidden_state, attention_mask):
    """Mean-pool token embeddings, respecting the attention mask."""
    mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return summed / counts


class TextDataset(Dataset):
    """Simple dataset that tokenises a list of strings."""

    def __init__(self, texts, tokenizer, max_length=2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Model description text
# ---------------------------------------------------------------------------

def build_model_description(model_name: str, info: dict) -> str:
    """
    Create a natural-language description of a model that captures its
    identity and benchmark performance so the embedding is informative.
    """
    parts = [f"Model: {model_name}."]
    # if "base_model" in info:
    #     parts.append(f"Base model: {info['base_model']}.")
    if "bbh_acc" in info:
        parts.append(f"BBH accuracy: {info['bbh_acc']:.4f}.")
    if "math_acc" in info:
        parts.append(f"Math accuracy: {info['math_acc']:.4f}.")
    if "mmlu_pro_acc" in info:
        parts.append(f"MMLU-Pro accuracy: {info['mmlu_pro_acc']:.4f}.")
    # if "co2_cost" in info:
    #     parts.append(f"CO2 cost: {info['co2_cost']} kg.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Extraction routines
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(texts, tokenizer, model, batch_size, max_length, device):
    """Return a list of numpy arrays (one per text)."""
    ds = TextDataset(texts, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in tqdm(loader, desc="Encoding"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)  # (B, 768)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).numpy()


def extract_query_embeddings(
    input_file,
    output_file,
    tokenizer,
    model,
    seen_models,
    batch_size,
    max_length,
    device,
):
    print(f"Loading queries from {input_file} ...")
    raw = load_jsonl(input_file)

    # De-duplicate by query_id (keep first occurrence)
    seen_ids = set()
    unique = []
    for r in raw:
        qid = r["query_id"]
        if qid not in seen_ids:
            seen_ids.add(qid)
            unique.append(r)
    print(f"  Unique queries: {len(unique)}")

    texts = [r["query"] for r in unique]
    embs = extract_embeddings(texts, tokenizer, model, batch_size, max_length, device)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(unique):
            obj = {
                "query_id": r["query_id"],
                "dataset": r.get("dataset", "unknown"),
                "query": r["query"],
                "embedding": embs[i].tolist(),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  Saved {len(unique)} query embeddings -> {output_file}")


def extract_model_embeddings(
    models_info_file,
    output_file,
    tokenizer,
    model,
    batch_size,
    max_length,
    device,
):
    print(f"Loading model info from {models_info_file} ...")
    with open(models_info_file, "r", encoding="utf-8") as f:
        models_info = json.load(f)
    print(f"  Models: {len(models_info)}")

    names = list(models_info.keys())
    texts = [build_model_description(n, models_info[n]) for n in names]

    # Show one example description
    if texts:
        print(f"  Example description: {texts[0]}")

    embs = extract_embeddings(texts, tokenizer, model, batch_size, max_length, device)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result = {}
    for i, name in enumerate(names):
        result[name] = embs[i].tolist()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"  Saved {len(result)} model embeddings -> {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract query & model embeddings with embeddinggemma-300m"
    )
    parser.add_argument(
        "--query_input",
        type=str,
        default="data/model_data/extracted_dataset_samples.jsonl",
        help="Path to query JSONL file",
    )
    parser.add_argument(
        "--models_info",
        type=str,
        default="data/model_data/seen_models.json",
        help="Path to model info JSON (seen_models.json)",
    )
    parser.add_argument(
        "--query_output",
        type=str,
        default="data/embeddings/query_embeddings.jsonl",
        help="Output path for query embeddings",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="data/embeddings/model_embeddings.json",
        help="Output path for model embeddings",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/embeddinggemma-300m",
        help="HuggingFace model name for the encoder",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Loading tokenizer & model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    encoder = encoder.to(args.device).eval()

    # --- Query embeddings ---
    extract_query_embeddings(
        input_file=args.query_input,
        output_file=args.query_output,
        tokenizer=tokenizer,
        model=encoder,
        seen_models=None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    # --- Model embeddings ---
    extract_model_embeddings(
        models_info_file=args.models_info,
        output_file=args.model_output,
        tokenizer=tokenizer,
        model=encoder,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
