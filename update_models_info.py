import json
from collections import defaultdict

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_model_accuracies(data_file):
    print(f"Loading data from: {data_file}")
    data = load_jsonl(data_file)
    print(f"Loaded {len(data)} records")
    
    model_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for record in data:
        model = record.get('model', '')
        dataset = record.get('dataset', '')
        correct = record.get('correct', 0.0)
        
        if not model or not dataset:
            continue
        
        model_stats[model][dataset]['total'] += 1
        model_stats[model][dataset]['correct'] += correct
    
    model_accuracies = {}
    for model, datasets in model_stats.items():
        model_accuracies[model] = {}
        for dataset, stats in datasets.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                model_accuracies[model][dataset] = acc
            else:
                model_accuracies[model][dataset] = 0.0
    
    return model_accuracies

def update_models_info(models_info_path, data_file):
    model_accuracies = calculate_model_accuracies(data_file)
    
    print(f"\nFound {len(model_accuracies)} models")
    print("Sample model accuracies:")
    for i, (model, accs) in enumerate(list(model_accuracies.items())[:3]):
        print(f"  {model}:")
        for dataset, acc in accs.items():
            print(f"    {dataset}: {acc:.4f}")
    
    try:
        with open(models_info_path, 'r', encoding='utf-8') as f:
            models_info = json.load(f)
        print(f"\nLoaded {len(models_info)} models from {models_info_path}")
    except FileNotFoundError:
        print(f"\n{models_info_path} not found. Creating new list...")
        models_info = []
    
    existing_models = {model['name']: model for model in models_info}
    
    updated_count = 0
    new_count = 0
    
    for model_name, accuracies in model_accuracies.items():
        bbh_acc = accuracies.get('bbh', 0.0)
        math_acc = accuracies.get('math', 0.0)
        mmlu_pro_acc = accuracies.get('mmlu_pro', 0.0)
        
        feature_vector = [bbh_acc, math_acc, mmlu_pro_acc]
        
        if model_name in existing_models:
            existing_models[model_name]['bbh_acc'] = bbh_acc
            existing_models[model_name]['math_acc'] = math_acc
            existing_models[model_name]['mmlu_pro_acc'] = mmlu_pro_acc
            existing_models[model_name]['feature_vector'] = feature_vector
            updated_count += 1
        else:
            new_model = {
                'name': model_name,
                'bbh_acc': bbh_acc,
                'math_acc': math_acc,
                'mmlu_pro_acc': mmlu_pro_acc,
                'feature_vector': feature_vector,
                'co2_cost': None,
                'base_model': None
            }
            existing_models[model_name] = new_model
            new_count += 1
    
    models_info = list(existing_models.values())
    
    models_info.sort(key=lambda x: x['name'])
    
    with open(models_info_path, 'w', encoding='utf-8') as f:
        json.dump(models_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nUpdated {models_info_path}")
    print(f"  Updated {updated_count} existing models")
    print(f"  Added {new_count} new models")
    print(f"  Total models: {len(models_info)}")
    
    print("\nSample updated models:")
    for model in models_info[:5]:
        print(f"\n  {model['name']}:")
        print(f"    BBH acc: {model['bbh_acc']:.4f}")
        print(f"    Math acc: {model['math_acc']:.4f}")
        print(f"    MMLU Pro acc: {model['mmlu_pro_acc']:.4f}")
        print(f"    Feature vector: {model['feature_vector']}")

if __name__ == "__main__":
    models_info_path = "data/model_data/models_info.json"
    data_file = "data/model_data/extracted_dataset_samples.jsonl"
    
    print("=" * 80)
    print("Updating models_info.json with feature vectors")
    print("=" * 80)
    
    update_models_info(models_info_path, data_file)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
