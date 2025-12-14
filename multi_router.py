import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def load_query_features(feature_file: str, dataset_filter: Optional[str] = None) -> Dict:
    queries = {}
    with open(feature_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            if dataset_filter and data['dataset'] != dataset_filter:
                continue

            query_id = data['query_id']
            feature_vector = data['feature_vector']
            queries[query_id] = {
                'query': data['query'],
                'dataset_probs': feature_vector[:3],  # [bbh, math, mmlu_pro]
                'difficulty': feature_vector[3],
                'dataset': data['dataset']
            }
    return queries


def load_models_info(models_file: str) -> Dict:
    with open(models_file, 'r', encoding='utf-8') as f:
        models_info = json.load(f)

    models = {}
    for model_name, info in models_info.items():
        models[model_name] = {
            'accuracies': [info['bbh_acc'], info['math_acc'], info['mmlu_pro_acc']],
            'co2_cost': info['co2_cost']
        }
    return models


def load_query_results(results_file: str) -> Dict:
    results = defaultdict(dict)
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['query_id']
            model_name = data['model']
            results[query_id][model_name] = {
                'correct': data['correct'],
                'answer': data['answer']
            }
    return results


def calculate_routing_score(query_probs: List[float], 
                            model_accs: List[float], 
                            difficulty: float,
                            cost: float,
                            all_costs: List[float],
                            beta: float) -> float:
    # Calculate weighted accuracy
    weighted_acc = sum(p * a for p, a in zip(query_probs, model_accs))

    # Relative score S: Adjust weighted accuracy by difficulty
    # Higher difficulty requires higher accuracy models; lower difficulty can tolerate lower accuracy models
    # Normalize result to [0,1]
    adjusted_score = weighted_acc - difficulty
    relative_score = (adjusted_score + 1) / 2
    relative_score = max(0, min(1, relative_score))

    # Normalize cost to [0,1]
    min_cost = min(all_costs)
    max_cost = max(all_costs)
    if max_cost > min_cost:
        normalized_cost = (cost - min_cost) / (max_cost - min_cost)
    else:
        normalized_cost = 0

    # Final score: (1-β)S - βC
    # S is performance score (higher is better), C is cost (lower is better, hence the minus sign)
    routing_score = (1 - beta) * relative_score - beta * normalized_cost

    return routing_score


def route_query(query_info: Dict, 
                models: Dict,
                all_costs: List[float],
                beta: float) -> str:
    """Select the best model for a single query"""
    best_model = None
    best_score = -float('inf')

    for model_name, model_info in models.items():
        score = calculate_routing_score(
            query_info['dataset_probs'],
            model_info['accuracies'],
            query_info['difficulty'],
            model_info['co2_cost'],
            all_costs,
            beta
        )

        if score > best_score:
            best_score = score
            best_model = model_name

    return best_model


def evaluate_router(queries: Dict, 
                    models: Dict, 
                    results: Dict, 
                    beta: float,
                    verbose: bool = False) -> Tuple[float, float]:
    """
    Evaluate router performance

    Returns:
        (accuracy, average cost)
    """
    all_costs = [info['co2_cost'] for info in models.values()]

    total_correct = 0
    total_cost = 0
    total_queries = 0

    model_selection_count = defaultdict(int)

    for query_id, query_info in queries.items():
        # Route to the best model
        selected_model = route_query(query_info, models, all_costs, beta)

        model_selection_count[selected_model] += 1

        # Get the results for the selected model
        if query_id in results and selected_model in results[query_id]:
            result = results[query_id][selected_model]
            total_correct += result['correct']
            total_cost += models[selected_model]['co2_cost']
            total_queries += 1

    accuracy = total_correct / total_queries if total_queries > 0 else 0
    avg_cost = total_cost / total_queries if total_queries > 0 else 0

    if verbose:
        print(f"\n  Model selection distribution (β={beta}):")
        sorted_selections = sorted(model_selection_count.items(), key=lambda x: x[1], reverse=True)
        for model_name, count in sorted_selections[:5]:  # Show top 5 most selected models
            pct = count / total_queries * 100
            print(f"    {model_name}: {count} times ({pct:.2f}%)")

    return accuracy, avg_cost


def evaluate_single_model(model_name: str,
                         queries: Dict,
                         models: Dict,
                         results: Dict) -> Tuple[float, float]:
    """Evaluate the performance of a single model (baseline)"""
    total_correct = 0
    total_cost = 0
    total_queries = 0

    model_cost = models[model_name]['co2_cost']

    for query_id in queries.keys():
        if query_id in results and model_name in results[query_id]:
            result = results[query_id][model_name]
            total_correct += result['correct']
            total_cost += model_cost
            total_queries += 1

    accuracy = total_correct / total_queries if total_queries > 0 else 0
    avg_cost = total_cost / total_queries if total_queries > 0 else 0

    return accuracy, avg_cost


def main():
    parser = argparse.ArgumentParser(description='Multi-model Router')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['bbh', 'math', 'mmlu_pro', 'gpqa', 'musr'],
                        help='Select the dataset to evaluate (default: all datasets)')
    parser.add_argument('--beta', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.1018, 0.102, 0.11, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
                        help='Specify the list of beta values')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed model selection distribution')
    parser.add_argument('--feature_file', type=str, default='data/feature_vectors/ID/data_samples_features.jsonl',
                        help='Path to the feature vectors file')
    parser.add_argument('--models_file', type=str, default='data/model_data/models_info.json',
                        help='Path to the models info file')
    parser.add_argument('--results_file', type=str, default='data/model_data/extracted_dataset_samples.jsonl',
                        help='Path to the extracted dataset samples file')
    args = parser.parse_args()

    print("=" * 80)
    if args.dataset:
        print(f"Multi-model Router - Dataset: {args.dataset.upper()}")
    else:
        print("Multi-model Router - Dataset: All")
    print("=" * 80)
    print()

    print("Loading data...")
    queries = load_query_features(args.feature_file, args.dataset)
    models = load_models_info(args.models_file)
    results = load_query_results(args.results_file)

    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(models)} models")
    print()

    # Evaluate baseline: all single models
    print("=" * 80)
    print("Baseline Evaluation: Single Model Performance")
    print("=" * 80)

    baseline_results = []
    for model_name in models.keys():
        acc, cost = evaluate_single_model(model_name, queries, models, results)
        baseline_results.append((model_name, acc, cost))

    baseline_results.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Model Name':<50} {'Accuracy':>12} {'Avg Cost':>12}")
    print("-" * 80)
    for model_name, acc, cost in baseline_results:
        print(f"{model_name:<50} {acc:>10.4f} {cost:>12.2f}")

    print()
    print("=" * 80)
    print("Router Evaluation: Performance at Different β Values")
    print("=" * 80)

    # Evaluate router performance at different beta values
    beta_values = args.beta

    print(f"{'β Value':>12} {'Accuracy':>10} {'Avg Cost':>10}")
    print("-" * 80)

    router_results = []
    for beta in beta_values:
        acc, cost = evaluate_router(queries, models, results, beta, verbose=args.verbose)
        router_results.append((beta, acc, cost))
        print(f"{beta:>10.4f} {acc:>10.4f} {cost:>12.2f}")

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    # Find the best single model baseline
    best_baseline = max(baseline_results, key=lambda x: x[1])
    print(f"Best Single Model Baseline: {best_baseline[0]}")
    print(f"  Accuracy: {best_baseline[1]:.4f}")
    print(f"  Avg Cost: {best_baseline[2]:.2f}")
    print()

    # Find the cheapest model
    cheapest_baseline = min(baseline_results, key=lambda x: x[2])
    print(f"Cheapest Model: {cheapest_baseline[0]}")
    print(f"  Accuracy: {cheapest_baseline[1]:.4f}")
    print(f"  Avg Cost: {cheapest_baseline[2]:.2f}")
    print()

    # Analyze router results
    print("Router Performance Analysis:")
    print()
    for beta, acc, cost in router_results:
        print(f"  β={beta:.4f}:")
        print(f"    Accuracy: {acc:.4f} | Avg Cost: {cost:.2f}")
        print(f"    vs Best Baseline: Accuracy Difference={acc-best_baseline[1]:+.4f}, Cost Difference={cost-best_baseline[2]:+.2f}")
        acc_loss_pct = (acc - best_baseline[1]) / best_baseline[1] * 100
        cost_saving_pct = (best_baseline[2] - cost) / best_baseline[2] * 100
        print(f"    Accuracy Change: {acc_loss_pct:+.2f}%, Cost Savings: {cost_saving_pct:+.2f}%")
        print()


if __name__ == "__main__":
    main()
