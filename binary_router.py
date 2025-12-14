import numpy as np
import json
from utils.utils import load_jsonl
import matplotlib.pyplot as plt

# Cost with 1% acc lose (mf): 
# all: 49.66, 
# bbh: 44.20, 
# For math, slm = "Qwen/Qwen2.5-32B-Instruct" llm = "MaziyarPanahi/calme-3.2-instruct-78b" is not suitable, SLM > LLM 
# mmlu_pro: 60.56

# Cost with 1% acc lose (bert):
# all: 60.56,
# bbh: 60.56,
# math:  not suitable, SLM > LLM
# mmlu_pro: 60.56

num_points = 10
dataset = None    # None, "math", "bbh", "mmlu_pro"
slm = "Qwen/Qwen2.5-0.5B-Instruct"
llm = "MaziyarPanahi/calme-3.2-instruct-78b"

query_data = load_jsonl(f"data/feature_vectors/ID/data_samples_features.jsonl")
difficulty_data = load_jsonl("data/model_data/query_difficulty.jsonl")
baseline_data = load_jsonl("data/baseline_scores/bert_router_scores.jsonl")
slm_data = load_jsonl(f"data/model_data/extracted_dataset_samples.jsonl", model=slm)
llm_data = load_jsonl(f"data/model_data/extracted_dataset_samples.jsonl", model=llm)

# Load model costs
with open("data/model_data/models_info.json", 'r', encoding='utf-8') as f:
    models_info = json.load(f)

slm_cost = models_info[slm]['co2_cost']
llm_cost = models_info[llm]['co2_cost']

print(f"Loaded {len(query_data)} queries, {len(slm_data)} SLM results, {len(llm_data)} LLM results")
print(f"SLM ({slm}): cost = {slm_cost}")
print(f"LLM ({llm}): cost = {llm_cost}")

if dataset:
    query_data = [item for item in query_data if item['dataset'] == dataset]
    difficulty_data = [item for item in difficulty_data if item['dataset'] == dataset]
    baseline_data = [item for item in baseline_data if item['dataset'] == dataset]
    slm_data = [item for item in slm_data if item['dataset'] == dataset]
    llm_data = [item for item in llm_data if item['dataset'] == dataset]

fourth_values = []
for item in query_data:
    fourth_values.append(item['feature_vector'][3])

fourth_values = np.array(fourth_values)

difficulty_dict = {}
for item in difficulty_data:
    difficulty_dict[item['query_id']] = item['difficulty']

difficulty_scores = []
for item in query_data:
    query_id = item['query_id']
    if query_id in difficulty_dict:
        difficulty_scores.append(difficulty_dict[query_id])

difficulty_scores = np.array(difficulty_scores)

baseline_dict = {}
for item in baseline_data:
    baseline_dict[item['query_id']] = item['score']

baseline_scores = []
for item in query_data:
    query_id = item['query_id']
    if query_id in baseline_dict:
        baseline_scores.append(baseline_dict[query_id])

baseline_scores = np.array(baseline_scores)

# Calculate accuracies and costs based on 4th feature percentiles
percentiles_4th = np.percentile(fourth_values, np.arange(0, 101, 100//num_points)).tolist()
print("Percentiles (4th feature):", percentiles_4th)

accuracies_4th = []
costs_4th = []
for p in percentiles_4th:
    correct = 0
    total_cost = 0
    for slm_item, llm_item, f4_value in zip(slm_data, llm_data, fourth_values):
        if f4_value > p:
            correct += int(slm_item["correct"])
            total_cost += slm_cost
        else:
            correct += int(llm_item["correct"])
            total_cost += llm_cost
    accuracy = correct / len(fourth_values)
    avg_cost = total_cost / len(fourth_values)
    accuracies_4th.append(accuracy)
    costs_4th.append(avg_cost)

x_points = np.arange(0, 101, 100//num_points)

# Calculate accuracies and costs based on difficulty percentiles
percentiles = np.percentile(difficulty_scores, np.arange(0, 101, 100//num_points)).tolist()
print("Percentiles (difficulty):", percentiles)

accuracies_difficulty = []
costs_difficulty = []
for p in percentiles:
    correct = 0
    total_cost = 0
    for slm_item, llm_item, diff_score in zip(slm_data, llm_data, difficulty_scores):
        if diff_score > p:
            correct += int(slm_item["correct"])
            total_cost += slm_cost
        else:
            correct += int(llm_item["correct"])
            total_cost += llm_cost
    accuracy = correct / len(difficulty_scores)
    avg_cost = total_cost / len(difficulty_scores)
    accuracies_difficulty.append(accuracy)
    costs_difficulty.append(avg_cost)

# Calculate accuracies and costs based on baseline percentiles
percentiles_baseline = np.percentile(baseline_scores, np.arange(0, 101, 100//num_points)).tolist()
print("Percentiles (baseline):", percentiles_baseline)

accuracies_baseline = []
costs_baseline = []
for p in percentiles_baseline:
    correct = 0
    total_cost = 0
    for slm_item, llm_item, base_score in zip(slm_data, llm_data, baseline_scores):
        if base_score > p:
            correct += int(slm_item["correct"])
            total_cost += slm_cost
        else:
            correct += int(llm_item["correct"])
            total_cost += llm_cost
    accuracy = correct / len(baseline_scores)
    avg_cost = total_cost / len(baseline_scores)
    accuracies_baseline.append(accuracy)
    costs_baseline.append(avg_cost)

# Calculate random routing baseline
np.random.seed(42)
random_routing_accuracies = []
random_routing_costs = []
for percent in x_points:
    correct = 0
    total_cost = 0
    for slm_item, llm_item in zip(slm_data, llm_data):
        if np.random.random() > (percent / 100):
            correct += int(slm_item["correct"])
            total_cost += slm_cost
        else:
            correct += int(llm_item["correct"])
            total_cost += llm_cost
    accuracy = correct / len(slm_data)
    avg_cost = total_cost / len(slm_data)
    random_routing_accuracies.append(accuracy)
    random_routing_costs.append(avg_cost)

# Print results
print("\n" + "=" * 80)
print(f"Results for dataset: {dataset if dataset else 'all datasets'}")
print("=" * 80)
print(f"\n{'Percentile':<12} {'4th Feature':<25} {'Difficulty':<25} {'Baseline':<25} {'Random':<25}")
print(f"{'(% LLM)':<12} {'Acc':>10} {'Cost':>12} {'Acc':>10} {'Cost':>12} {'Acc':>10} {'Cost':>12} {'Acc':>10} {'Cost':>12}")
print("-" * 120)

for i, pct in enumerate(x_points):
    print(f"{pct:<12.0f} {accuracies_4th[i]:>10.4f} {costs_4th[i]:>12.2f} "
          f"{accuracies_difficulty[i]:>10.4f} {costs_difficulty[i]:>12.2f} "
          f"{accuracies_baseline[i]:>10.4f} {costs_baseline[i]:>12.2f} "
          f"{random_routing_accuracies[i]:>10.4f} {random_routing_costs[i]:>12.2f}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"SLM only - Accuracy: {accuracies_4th[0]:.4f}, Cost: {slm_cost:.2f}")
print(f"LLM only - Accuracy: {accuracies_4th[-1]:.4f}, Cost: {llm_cost:.2f}")
print("=" * 80)

# Plot accuracies by percentile
plt.figure(figsize=(10, 6))
plt.plot(x_points, accuracies_4th, marker='o', label=f'Routing Based on 4th Feature')
plt.plot(x_points, accuracies_difficulty, marker='s', label='Routing Based on Difficulty Score')
plt.plot(x_points, accuracies_baseline, marker='^', label=f'Routing Based on Baseline Score')
plt.plot(x_points, random_routing_accuracies, color='r', linestyle='--', marker='x', label='Random Routing')
plt.xlabel('Percentile (% using LLM)')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs Percentile with Baselines (SLM%→LLM%) for {dataset if dataset else "all datasets"}')
plt.legend()
plt.grid(True)
plt.show()