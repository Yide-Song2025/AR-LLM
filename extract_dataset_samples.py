import json
from collections import defaultdict
from utils.utils import load_jsonl

def main():
    data = load_jsonl('data/model_data/openllm_leaderboard_extracted_data_ood.jsonl', model=None)
    print(f"Successfully loaded {len(data)} records")
    
    dataset_queries = defaultdict(lambda: defaultdict(list))
    
    print("\nOrganizing data...")
    for record in data:
        dataset = record.get('dataset', '')
        query = record.get('query', '')
        
        if query:
            dataset_queries[dataset][query].append(record)
    
    print("\nDataset statistics:")
    for dataset, queries in sorted(dataset_queries.items()):
        print(f"  {dataset}: {len(queries)} unique queries")
    
    print("\nExtracting queries for each dataset...")
    extracted_data = defaultdict(list)
    query_id_mapping = {}
    
    for dataset, queries_dict in dataset_queries.items():
        queries = list(queries_dict.keys())
        
        for query_idx, query in enumerate(queries):
            query_id = f"{dataset}_q{query_idx}"
            query_id_mapping[query] = query_id

        for query in queries:
            records = queries_dict[query]
            extracted_data[dataset].extend(records)
    
    print("\nDataset statistics:")
    for dataset, records in sorted(extracted_data.items()):
        unique_queries = len(set(r['query'] for r in records))
        unique_models = len(set(r['model'] for r in records))
        print(f"  {dataset}:")
        print(f"    - Queries: {unique_queries}")
        print(f"    - Models: {unique_models}")
        print(f"    - Total records: {len(records)}")
    
    output_file = 'extracted_dataset_samples.jsonl'
    print(f"\nSaving data to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for dataset, records in sorted(extracted_data.items()):
            for record in records:
                query = record.get('query', '')
                record['query_id'] = query_id_mapping.get(query, 'unknown')
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {sum(len(records) for records in extracted_data.values())} records")
    
    print("\nGenerating files grouped by dataset...")
    for dataset, records in sorted(extracted_data.items()):
        dataset_file = f'extracted_{dataset}_samples.jsonl'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            for record in records:
                query = record.get('query', '')
                record['query_id'] = query_id_mapping.get(query, 'unknown')
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  Saved {dataset} dataset to {dataset_file} ({len(records)} records)")

if __name__ == "__main__":
    main()
