"""
Generate query scores using RouteLLM routers as baseline comparison.
Reads queries from extracted_dataset_samples.jsonl and outputs routing scores.
"""

import json
import argparse
import os
import sys
from collections import defaultdict
from tqdm import tqdm

# Add RouteLLM to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RouteLLM'))

from routellm.routers.routers import ROUTER_CLS


def load_unique_queries(input_file):
    """Load unique queries from the dataset samples file."""
    queries = {}
    print(f"Loading queries from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['query_id']
            
            # Store only unique queries (one per query_id)
            if query_id not in queries:
                queries[query_id] = {
                    'query': data['query'],
                    'dataset': data['dataset'],
                    'subset': data.get('subset', '')
                }
    
    print(f"Loaded {len(queries)} unique queries")
    return queries


def initialize_router(router_type, router_config):
    """Initialize a router based on type and configuration."""
    print(f"\nInitializing {router_type} router...")
    
    router_cls = ROUTER_CLS[router_type]
    
    if router_config:
        router = router_cls(**router_config)
    else:
        router = router_cls()
    
    print(f"{router_type} router initialized successfully")
    return router


def get_default_router_configs():
    """Get default configurations for each router type."""
    configs = {
        "random": {},
        "mf": {
            "checkpoint_path": "routellm/mf",
            "strong_model": "gpt-4-1106-preview",
            "weak_model": "mixtral-8x7b-instruct-v0.1",
        },
        "bert": {
            "checkpoint_path": "routellm/bert",
        },
        "causal_llm": {
            "checkpoint_path": "routellm/causal_llm",
        },
        "sw_ranking": {
            "arena_battle_datasets": ["lmsys/lmsys-arena-human-preference-55k"],
            "arena_embedding_datasets": ["routellm/lmsys-arena-human-preference-55k-embed"],
            "strong_model": "gpt-4-1106-preview",
            "weak_model": "mixtral-8x7b-instruct-v0.1",
        }
    }
    return configs


def generate_router_scores(queries, router, router_name, output_file):
    """Generate routing scores for all queries and save to file."""
    print(f"\nGenerating scores with {router_name} router...")
    
    results = []
    
    for query_id, query_info in tqdm(queries.items(), desc=f"Processing with {router_name}"):
        try:
            # Calculate routing score (strong model win rate)
            score = router.calculate_strong_win_rate(query_info['query'])
            
            result = {
                'query_id': query_id,
                'dataset': query_info['dataset'],
                'subset': query_info['subset'],
                'query': query_info['query'],
                'router': router_name,
                'score': float(score)
            }
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing query {query_id}: {e}")
            continue
    
    # Save results
    print(f"Saving {len(results)} results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Results saved successfully")
    
    # Print statistics
    if results:
        scores = [r['score'] for r in results]
        print(f"\nScore Statistics for {router_name}:")
        print(f"  Min:  {min(scores):.4f}")
        print(f"  Max:  {max(scores):.4f}")
        print(f"  Mean: {sum(scores)/len(scores):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate query routing scores using RouteLLM routers'
    )
    parser.add_argument(
        '--router',
        type=str,
        choices=['random', 'mf', 'bert', 'causal_llm', 'sw_ranking', 'all'],
        default='random',
        help='Router type to use (default: random)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/model_data/extracted_dataset_samples.jsonl',
        help='Input file with dataset samples'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/baseline_scores',
        help='Output directory for router scores'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='JSON file with custom router configuration'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load queries
    queries = load_unique_queries(args.input)
    
    # Get router configurations
    if args.config:
        with open(args.config, 'r') as f:
            router_configs = json.load(f)
    else:
        router_configs = get_default_router_configs()
    
    # Determine which routers to run
    if args.router == 'all':
        routers_to_run = list(ROUTER_CLS.keys())
    else:
        routers_to_run = [args.router]
    
    print("=" * 80)
    print("RouteLLM Baseline Score Generation")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Routers to run: {', '.join(routers_to_run)}")
    
    # Process each router
    for router_name in routers_to_run:
        try:
            print("\n" + "=" * 80)
            
            # Initialize router
            router_config = router_configs.get(router_name, {})
            router = initialize_router(router_name, router_config)
            
            # Generate scores
            output_file = os.path.join(
                args.output_dir,
                f"{router_name}_router_scores.jsonl"
            )
            generate_router_scores(queries, router, router_name, output_file)
            
        except Exception as e:
            print(f"\nError with {router_name} router: {e}")
            print("Skipping this router...")
            continue
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
