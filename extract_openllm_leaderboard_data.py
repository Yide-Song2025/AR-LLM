import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# Models to extract data for
MODELS = [
    "MaziyarPanahi/calme-3.2-instruct-78b",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
]

# BBH subsets
BBH_SUBSETS = [
    'boolean_expressions', 'causal_judgement', 'date_understanding', 
    'disambiguation_qa', 'formal_fallacies', 
    'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 
    'logical_deduction_seven_objects', 'logical_deduction_three_objects', 
    'movie_recommendation', 'navigate', 
    'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 
    'ruin_names', 'salient_translation_error_detection', 'snarks', 
    'sports_understanding', 'temporal_sequences', 
    'tracking_shuffled_objects_five_objects', 
    'tracking_shuffled_objects_seven_objects', 
    'tracking_shuffled_objects_three_objects', 'web_of_lies'
]

# MATH subsets
MATH_SUBSETS = [
    'math_algebra_hard', 'math_counting_and_prob_hard', 
    'math_geometry_hard', 'math_intermediate_algebra_hard', 
    'math_num_theory_hard', 'math_prealgebra_hard', 'math_precalculus_hard'
]

OUTPUT_FILE = "data/model_data/openllm_leaderboard_extracted_data_ood.jsonl"

BATCH_SIZE = 16  # Number of concurrent API calls
MAX_WORKERS = 16 # Maximum number of parallel threads

# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )


def check_math_answer_with_deepseek(question: str, true_answer: str, model_response: str, max_retries: int = 3) -> int:
    prompt = f"""You are a math answer evaluator. Given a math question, the correct answer, and a model's response, determine if the model's answer is correct.

Question: {question}

Correct Answer: {true_answer}

Model's Response: {model_response}

Please analyze if the model's final answer matches the correct answer. Consider mathematical equivalence (e.g., 7, 7.0, and seven are equivalent).

Respond with ONLY "1" if the answer is correct, or "0" if the answer is incorrect or cannot be determined. No explanation needed."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a precise math answer evaluator. Respond only with 1 or 0."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip()
            
            if result == "1":
                return 1
            elif result == "0":
                return 0
            else:
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"Warning: DeepSeek API returned unclear result: {result}. Defaulting to 0.")
                    return 0
                    
        except Exception as e:
            print(f"Error calling DeepSeek API (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt >= max_retries - 1:
                return 0
    
    return 0


def check_math_answer_batch(index: int, question: str, true_answer: str, model_response: str) -> Tuple[int, int]:
    try:
        correct = check_math_answer_with_deepseek(question, true_answer, model_response)
        return (index, correct)
    except Exception as e:
        print(f"Error in batch processing for item {index}: {e}")
        return (index, 0)


def extract_model_results_from_leaderboard(model_name: str, dataset: str, subset: str = None, max_retries: int = 3) -> List[Dict]:
    try:
        results = []
        model_name_formatted = model_name.replace("/", "__")
        
        details_dataset = f"open-llm-leaderboard/{model_name_formatted}-details"
        
        if subset:
            config_name = f"{model_name_formatted}__leaderboard_{dataset}_{subset}"
        else:
            config_name = f"{model_name_formatted}__leaderboard_{dataset}"
        
        ds = load_dataset(
            details_dataset, 
            config_name, 
            split="latest"
        )
        
        # First pass: collect all data
        math_batch_items = []  # Store items that need API evaluation
        
        for idx, row_data in enumerate(ds):
            doc = row_data.get("doc")
            
            if dataset != "musr":
                query = doc.get("input", 
                        doc.get("problem", 
                        doc.get("question", 
                        doc.get("Pre-Revision Question",
                        doc.get("Question")))))
            
            else:
                query = doc.get("narrative") + "\n" + doc.get("question")
            
            answer = row_data.get("target")
            
            result_dict = {
                "query": query,
                "answer": answer,
                "model": model_name,
                "dataset": dataset,
                "subset": subset,
                "correct": None
            }
            
            if dataset == "math" and model_name != "MaziyarPanahi/calme-3.2-instruct-78b" and model_name != "meta-llama/Llama-3.1-8B-Instruct":
                filtered_resps = row_data.get("filtered_resps")
                if filtered_resps and len(filtered_resps) > 0:
                    response = filtered_resps[0]
                else:
                    response = ""
                
                math_batch_items.append((len(results), query, answer, response))
            else:
                result_dict["correct"] = row_data.get("acc_norm", row_data.get("exact_match", row_data.get("acc")))
            
            results.append(result_dict)
        
        if math_batch_items:
            print(f"  Processing {len(math_batch_items)} math items in parallel...")
            correctness_map = {}
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(check_math_answer_batch, idx, q, a, r): idx 
                    for idx, q, a, r in math_batch_items
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="  API calls", leave=False):
                    try:
                        idx, correct = future.result()
                        correctness_map[idx] = correct
                    except Exception as e:
                        print(f"  Error processing future: {e}")
                        idx = futures[future]
                        correctness_map[idx] = 0
            
            for idx, _, _, _ in math_batch_items:
                results[idx]["correct"] = correctness_map.get(idx, 0)
        
        return results
    except Exception as e:
        print(f"Error in {details_dataset}: {e}")
        return []


def main():

    all_results = []
    
    print(f"Target models: {len(MODELS)}")
    for model in MODELS:
        print(f"  - {model}")
    
    print(f"\nTarget datasets:")
    print(f"  - BBH ({len(BBH_SUBSETS)} subsets)")
    print(f"  - MATH ({len(MATH_SUBSETS)} subsets)")
    print(f"  - MMLU-Pro")
    
    for model in MODELS:
        print(f"\n{'=' * 80}")
        print(f"Processing model: {model}")
        print(f"{'=' * 80}")
        
        # print(f"\nExtracting BBH data...")
        # bbh_success = 0
        # bbh_failed = 0
        # for subset in tqdm(BBH_SUBSETS, desc="BBH subsets"):
        #     results = extract_model_results_from_leaderboard(model, "bbh", subset)
        #     if results:
        #         bbh_success += 1
        #         all_results.extend(results)
        #     else:
        #         bbh_failed += 1
        # print(f"  BBH : Success {bbh_success}, Failed {bbh_failed}")
        
        # print(f"\nExtracting MATH data...")
        # math_success = 0
        # math_failed = 0
        # for subset in tqdm(MATH_SUBSETS, desc="MATH subsets"):
        #     results = extract_model_results_from_leaderboard(model, "math", subset)
        #     if results:
        #         math_success += 1
        #         all_results.extend(results)
        #     else:
        #         math_failed += 1
        # print(f"  MATH : Success {math_success}, Failed {math_failed}")
        
        # print(f"\nExtracting MMLU-Pro data...")
        # results = extract_model_results_from_leaderboard(model, "mmlu_pro")
        # if results:
        #     all_results.extend(results)
        #     print(f"  MMLU-Pro : Success")
        # else:
        #     print(f"  MMLU-Pro : Failed")

        results = extract_model_results_from_leaderboard(model, "gpqa", subset="main")
        if results:
            all_results.extend(results)
            print(f"  GPQA-Main : Success")
        
        results = extract_model_results_from_leaderboard(model, "gpqa", subset="diamond")
        if results:
            all_results.extend(results)
            print(f"  GPQA-Diamond : Success")

        results = extract_model_results_from_leaderboard(model, "gpqa", subset="extended")
        if results:
            all_results.extend(results)
            print(f"  GPQA-Extended : Success")

        results = extract_model_results_from_leaderboard(model, "musr", subset="murder_mysteries")
        if results:
            all_results.extend(results)
            print(f"  MUSR-Murder-Mysteries : Success")
        
        results = extract_model_results_from_leaderboard(model, "musr", subset="object_placements")
        if results:
            all_results.extend(results)
            print(f"  MUSR-Object-Placement : Success")

        results = extract_model_results_from_leaderboard(model, "musr", subset="team_allocation")
        if results:
            all_results.extend(results)
            print(f"  MUSR-Team-Allocation : Success")

        if all_results:
            output_path = Path(OUTPUT_FILE)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            temp_file = output_path.parent / f"{output_path.stem}_temp{output_path.suffix}"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for result in all_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nExtraction completed!")
    print(f"Total records extracted: {len(all_results)}")
    print(f"Saved to: {output_path}")
    
    print(f"\nData statistics:")
    models_count = {}
    datasets_count = {}
    
    for result in all_results:
        model = result.get("model", "unknown")
        dataset = result.get("dataset", "unknown")
        
        models_count[model] = models_count.get(model, 0) + 1
        datasets_count[dataset] = datasets_count.get(dataset, 0) + 1
    
    print(f"\nBy model:")
    for model, count in models_count.items():
        print(f"  {model}: {count} records")
    
    print(f"\nBy dataset:")
    for dataset, count in datasets_count.items():
        print(f"  {dataset}: {count} records")


if __name__ == "__main__":
    main()
