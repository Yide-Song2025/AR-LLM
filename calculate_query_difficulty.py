"""Calculate query difficulty based on percentage of wrong answers across models"""
import json
from collections import defaultdict
from pathlib import Path

print("Calculating query difficulty from extracted_dataset_samples.jsonl")
print("=" * 80)

query_model_answers = defaultdict(lambda: defaultdict(list))
query_info = {}

input_file = Path('data/model_data/extracted_dataset_samples.jsonl')

print("\nReading data and handling duplicates...")
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        
        query = data.get('query', '').strip()
        query_id = data.get('query_id', '').strip()
        dataset = data.get('dataset', '').strip()
        subset = data.get('subset', '') or ''
        if subset:
            subset = subset.strip()
        correct = float(data.get('correct', 0))
        model = data.get('model', '').strip()
        
        if query_id not in query_info:
            query_info[query_id] = {
                'dataset': dataset,
                'subset': subset,
                'query': query
            }
        
        query_model_answers[query_id][model].append(correct)

print(f"Found {len(query_info)} unique queries")

print("\nCalculating difficulty based on actual answer counts...")
query_difficulty = {}

for query_id, models_answers in query_model_answers.items():
    total_answers = 0  
    wrong_answers = 0  
    unique_models = len(models_answers) 
    
    for model, answers in models_answers.items():
        total_answers += len(answers)
        wrong_answers += sum(1 for ans in answers if ans == 0)
    
    difficulty = (wrong_answers / total_answers * 100.0) if total_answers > 0 else 0.0
    
    query_difficulty[query_id] = {
        'dataset': query_info[query_id]['dataset'],
        'subset': query_info[query_id]['subset'],
        'query': query_info[query_id]['query'],
        'query_id': query_id,
        'unique_models': unique_models,
        'total_answers': total_answers,
        'wrong_answers': wrong_answers,
        'correct_answers': total_answers - wrong_answers,
        'difficulty': round(difficulty, 2)
    }

# Check for duplicates and model counts
print("\nChecking for duplicates and model counts...")
total_records = sum(len(answers) for models in query_model_answers.values() for answers in models.values())
expected_records = len(query_info) * 16
duplicates = total_records - expected_records

print(f"  Total records: {total_records}")
print(f"  Expected records (queries × 16): {expected_records}")
if duplicates > 0:
    print(f"  WARNING: Found {duplicates} duplicate records ({duplicates/total_records*100:.2f}%)")
else:
    print(f"  OK: No duplicates found")

queries_with_issues = []
for query_id, models_answers in query_model_answers.items():
    unique_models = len(models_answers)
    total_answers = sum(len(answers) for answers in models_answers.values())
    if unique_models != 16 or total_answers != unique_models:
        queries_with_issues.append((query_id, unique_models, total_answers))

if queries_with_issues:
    print(f"\n  WARNING: Found {len(queries_with_issues)} queries with issues:")
    for query_id, unique_models, total_answers in queries_with_issues[:10]:
        dataset = query_difficulty[query_id]['dataset']
        query = query_difficulty[query_id]['query']
        duplicates_count = total_answers - unique_models
        print(f"    {dataset} ({query_id}): {unique_models} models, {total_answers} answers (duplicates: {duplicates_count})")
    if len(queries_with_issues) > 10:
        print(f"    ... and {len(queries_with_issues) - 10} more")
else:
    print(f"  OK: All queries have exactly 16 unique models with no duplicates")

# Group by dataset category
print("\nGrouping by dataset category...")
bbh_queries = []
math_queries = []
mmlu_queries = []

for query_id, data in query_difficulty.items():
    dataset = data['dataset']
    query_entry = {
        'query_id': data['query_id'],
        'dataset': dataset,
        'subset': data['subset'],
        'query': data['query'],
        'difficulty': data['difficulty'],
        'wrong_answers': data['wrong_answers'],
        'unique_models': data['unique_models'],
        'total_answers': data['total_answers']
    }
    
    if dataset == 'bbh':
        bbh_queries.append(query_entry)
    elif dataset == 'math':
        math_queries.append(query_entry)
    elif dataset == 'mmlu_pro':
        mmlu_queries.append(query_entry)

print(f"  BBH queries: {len(bbh_queries)}")
print(f"  MATH queries: {len(math_queries)}")
print(f"  MMLU_PRO queries: {len(mmlu_queries)}")

# Calculate statistics
def calculate_stats(queries):
    if not queries:
        return {}
    difficulties = [q['difficulty'] for q in queries]
    return {
        'count': len(queries),
        'min': min(difficulties),
        'max': max(difficulties),
        'mean': sum(difficulties) / len(difficulties),
        'median': sorted(difficulties)[len(difficulties) // 2]
    }

bbh_stats = calculate_stats(bbh_queries)
math_stats = calculate_stats(math_queries)
mmlu_stats = calculate_stats(mmlu_queries)

print("\n" + "=" * 80)
print("Difficulty Statistics:")
print("=" * 80)
print(f"\nBBH:")
if bbh_stats:
    print(f"  Total queries: {bbh_stats['count']}")
    print(f"  Min difficulty: {bbh_stats['min']:.2f}%")
    print(f"  Max difficulty: {bbh_stats['max']:.2f}%")
    print(f"  Mean difficulty: {bbh_stats['mean']:.2f}%")
    print(f"  Median difficulty: {bbh_stats['median']:.2f}%")

print(f"\nMATH:")
if math_stats:
    print(f"  Total queries: {math_stats['count']}")
    print(f"  Min difficulty: {math_stats['min']:.2f}%")
    print(f"  Max difficulty: {math_stats['max']:.2f}%")
    print(f"  Mean difficulty: {math_stats['mean']:.2f}%")
    print(f"  Median difficulty: {math_stats['median']:.2f}%")

print(f"\nMMLU_PRO:")
if mmlu_stats:
    print(f"  Total queries: {mmlu_stats['count']}")
    print(f"  Min difficulty: {mmlu_stats['min']:.2f}%")
    print(f"  Max difficulty: {mmlu_stats['max']:.2f}%")
    print(f"  Mean difficulty: {mmlu_stats['mean']:.2f}%")
    print(f"  Median difficulty: {mmlu_stats['median']:.2f}%")

all_queries = bbh_queries + math_queries + mmlu_queries

output_jsonl = "query_difficulty.jsonl"
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for q in sorted(all_queries, key=lambda x: (x['dataset'], -x['difficulty'])):
        json.dump({
            'query_id': q['query_id'],
            'dataset': q['dataset'],
            'subset': q['subset'],
            'query': q['query'],
            'difficulty': q['difficulty'],
            'wrong_answers': q['wrong_answers'],
            'unique_models': q['unique_models'],
            'total_answers': q['total_answers'],
            'correct_answers': q['total_answers'] - q['wrong_answers']
        }, f, ensure_ascii=False)
        f.write('\n')

print(f"Saved to: {output_jsonl}")

# Show sample of most difficult queries
print("\n" + "=" * 80)
print("Sample: Top 10 Most Difficult Queries (by category):")
print("=" * 80)

for category_name, queries in [("BBH", bbh_queries), ("MATH", math_queries), ("MMLU_PRO", mmlu_queries)]:
    if queries:
        sorted_queries = sorted(queries, key=lambda x: -x['difficulty'])[:10]
        print(f"\n{category_name} (Top 10):")
        for i, q in enumerate(sorted_queries, 1):
            total_answers = q['total_answers']
            wrong = q['wrong_answers']
            unique_models = q['unique_models']
            print(f"  {i}. Difficulty: {q['difficulty']:.2f}% ({wrong}/{total_answers} wrong, {unique_models} models)")
            print(f"     Query ID: {q['query_id']}")
            print(f"     Subset: {q['subset']}")
            print(f"     Query: {q['query'][:80]}...")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)

