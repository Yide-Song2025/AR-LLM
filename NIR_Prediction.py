import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from utils.utils import load_jsonl, InferenceDataset
from utils.model import BERTBinaryClassifier, MultiClassClassifier

def predict(
    multi_class_model_path,
    binary_model_path,
    input_file,
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

    return results


#simple MLP should work
class RouterNN(nn.Module):
    def __init__(self, query_vector,model_vector,projection_dim =32, hidden_size=(256,128), dropout_rate=0.1, output_size=1):
        super(RouterNN, self).__init__()
        self.q_proj = nn.Sequential(
            nn.Linear((query_vector), projection_dim),
            nn.ReLU(),
        )
        self.m_proj = nn.Sequential(
            nn.Linear((model_vector), projection_dim),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(projection_dim*4 + 1, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size[1], output_size)
        )


    def forward(self, query,model):
        q_emb = self.q_proj(query)
        m_emb = self.m_proj(model)

        prod = q_emb * m_emb
        diff  = torch.abs(q_emb - m_emb)
        cos = nn.functional.cosine_similarity(q_emb, m_emb, dim=1, eps=1e-8).unsqueeze(1)

        x = torch.cat((q_emb, m_emb, prod, diff, cos), dim=1)
        out = self.fc_layers(x)



        return out
    

def route_model_selection(model_feature_and_cost_df, query_vector_df,accuracy_perdictor_model_pth,alpha,device):
  # model set up
  route_model = RouterNN(query_vector=4,model_vector=3, projection_dim=32, hidden_size=(256,128), dropout_rate=0.1, output_size=1)
  route_model.load_state_dict(torch.load(accuracy_perdictor_model_pth))
  route_model.to(device)
  route_model.eval()

  #query df setting
  query_id_and_vector = query_vector_df[['query_id','feature_vector']]

  #Need to nomalize co2 cost in order to avoid complete lead by co2 cost
  cost_min = model_feature_and_cost_df['model_co2_cost'].min()
  cost_max = model_feature_and_cost_df['model_co2_cost'].max()
  query=[]
  for query_id, input_query_vector in query_id_and_vector.itertuples(index=False):
      best_model_name = None
      best_score = -float('inf')
      print(f'query id{query_id}')
      for model_name, model_vector, co2_cost in model_feature_and_cost_df.itertuples(index=True):
        input_query_tensor = torch.tensor(input_query_vector, dtype=torch.float32).unsqueeze(0).to(device)
        model_vector_tensor = torch.tensor(model_vector, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = route_model(input_query_tensor, model_vector_tensor)
            prob_correct = torch.sigmoid(logit).item()

            #normalize co2 cost
            co2_cost_norm = (co2_cost - cost_min) / (cost_max - cost_min)
            score = (1 - alpha)  * prob_correct - alpha * co2_cost_norm

            #update best model for this query if needed
            if score > best_score:
                best_score = score
                best_model_name = model_name

      query.append((query_id, best_model_name, best_score, alpha))
  return query


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict feature vectors for queries using trained models")
    parser.add_argument("--multi_class_model_path", type=str, required=True, help="Path to the trained multi-class classification model")
    parser.add_argument("--binary_model_path", type=str, required=True, help="Path to the trained binary classification model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with queries")
    parser.add_argument("--model_name", type=str, default="google/embeddinggemma-300m", help="Pretrained model name or path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for tokenizer")
    parser.add_argument("--model_cost_map", type=str, required=True, help="Path to JSON file containing model feature vectors and CO2 costs")
    parser.add_argument("--accuracy_predictor_model_path", type=str, required=True, help="Path to the trained accuracy predictor model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Trade-off parameter between accuracy and CO2 cost")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the models on (e.g., 'cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    query_feature_results_DF = predict(
        multi_class_model_path=args.multi_class_model_path,
        binary_model_path=args.binary_model_path,
        input_file=args.input_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    ) # This result will given feature result 

    #model info load 
    model_info = pd.read_json(args.model_cost_map).T
    model_info_feature_cost = model_info[['feature_vector', 'co2_cost']]
    model_info_feature_cost = model_info_feature_cost.rename(columns={"feature_vector": "model_feature_vector", "co2_cost": "model_co2_cost"})







    final_model =route_model_selection(
        model_feature_and_cost_df=model_info_feature_cost,
        query_vector_df=pd.DataFrame(query_feature_results_DF),
        accuracy_perdictor_model_pth=args.accuracy_predictor_model_path,
        alpha=args.alpha,
        device=args.device


    )
    print("\nFinal routing decisions:")
    for query_id, model_name, score, alpha in final_model:
        print(f"Query ID: {query_id}, Selected Model: {model_name}, Score: {score:.4f}, Alpha: {alpha}")
    

