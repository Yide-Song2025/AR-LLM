import os
import json
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.utils import load_jsonl, InferenceDataset
from utils.model import BERTBinaryClassifier, MultiClassClassifier


def predict_and_save(
    multi_class_model_path,
    binary_model_path,
    input_file,
    output_dir,
    model_name="google/embeddinggemma-300m",
    batch_size=16,
    max_length=2048,
    device="cuda"
):
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    
    # Remove duplicates based on query_id to get unique queries
    # Keep dataset information for each query
    unique_data = {}
    for sample in data:
        query_id = sample.get("query_id", "")
        if query_id not in unique_data:
            unique_data[query_id] = sample
    
    unique_samples = list(unique_data.values())
    print(f"Found {len(unique_samples)} unique queries")
    
    # Count datasets
    dataset_counts = {}
    for sample in unique_samples:
        ds = sample.get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    print(f"Datasets: {dataset_counts}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = InferenceDataset(unique_samples, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nLoading multi-class model from {multi_class_model_path}...")
    multi_class_model = MultiClassClassifier(
        model_name=model_name,
        hidden_dim=768,
        num_classes=3,
        dropout=0.1
    ).to(device)
    
    multi_class_checkpoint = torch.load(multi_class_model_path, map_location=device)
    multi_class_model.load_state_dict(multi_class_checkpoint)
    multi_class_model.eval()
    print("Multi-class model loaded successfully")
    
    print(f"\nLoading binary model from {binary_model_path}...")
    binary_model = BERTBinaryClassifier(
        model_name=model_name,
        hidden_dim=768,
        dropout=0.1
    ).to(device)
        
    binary_checkpoint = torch.load(binary_model_path, map_location=device)
    binary_model.load_state_dict(binary_checkpoint)
    binary_model.eval()
    print("Binary model loaded successfully")
    
    print("\nPerforming inference...")
    results = []
    
    # Create a mapping from query_id to dataset
    query_to_dataset = {sample.get("query_id", ""): sample.get("dataset", "unknown") for sample in unique_samples}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            queries = batch['query']
            query_ids = batch['query_id']
            
            multi_class_logits = multi_class_model(input_ids, attention_mask)
            multi_class_probs = torch.softmax(multi_class_logits, dim=1)  # [batch_size, 3]
            
            binary_logits = binary_model(input_ids, attention_mask)
            binary_probs = torch.sigmoid(binary_logits)  # [batch_size]
            binary_probs = binary_probs.unsqueeze(1)  # [batch_size, 1]
                
            feature_vectors = torch.cat([multi_class_probs, binary_probs], dim=1)
            
            feature_vectors_np = feature_vectors.cpu().numpy()
            
            for i in range(len(queries)):
                query_id = query_ids[i]
                result = {
                    "query_id": query_id,
                    "dataset": query_to_dataset.get(query_id, "unknown"),
                    "query": queries[i],
                    "feature_vector": feature_vectors_np[i].tolist()
                }
                results.append(result)
    
    # Determine output filename based on input file or dataset diversity
    if len(dataset_counts) == 1:
        data_set = list(dataset_counts.keys())[0]
        output_file = os.path.join(output_dir, f"{data_set}_samples_features.jsonl")
    else:
        # Multiple datasets, use input filename as base
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{input_basename}_features.jsonl")
    
    print(f"\nSaving results to {output_file}...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {len(results)} predictions")
    print(f"Feature vector dimension: {len(results[0]['feature_vector'])}")


def main():
    parser = argparse.ArgumentParser(description="Predict on math samples using trained models")
    parser.add_argument(
        "--multi_class_model",
        type=str,
        default="model_checkpoints/embeddinggemma_multi_class_classifier/best_model_acc_0.9849_epoch_1.pth",
        help="Path to multi-class model checkpoint"
    )
    parser.add_argument(
        "--binary_model",
        type=str,
        default="model_checkpoints/embeddinggemma_difficulty_predictor/best_model_acc_0.5203_epoch_0.pth",
        help="Path to binary model checkpoint (optional)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/feature_vectors",
        help="Path to output directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/embeddinggemma-300m",
        help="Pre-trained model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    
    args = parser.parse_args()
    
    predict_and_save(
        multi_class_model_path=args.multi_class_model,
        binary_model_path=args.binary_model,
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )


if __name__ == "__main__":
    main()
