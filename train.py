import random
import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.model import BERTBinaryClassifier, MultiClassClassifier, train_loops
from utils.utils import DifficultyDataset, MultiClassClassificationDataset, preprocess_data, load_file, preprocess_task_data, load_jsonl

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def argument_parser():
    parser = argparse.ArgumentParser(description="Train BERT Binary Classifier with EmbeddingGemma")
    parser.add_argument("--model_name", type=str, default="google/embeddinggemma-300m", help="Pre-trained model name")
    parser.add_argument("--query_difficulty_path", type=str, default="data/model_data/query_difficulty.jsonl", help="Path to query difficulty data")
    parser.add_argument("--bbh_datasets", type=str, default="lukaemon/bbh", help="Path to BBH data")
    parser.add_argument("--math_datasets", type=str, default="qwedsacf/competition_math", help="Path to Math data")
    parser.add_argument("--mmlu_pro_datasets", type=str, default="TIGER-Lab/MMLU-Pro", help="Path to MMLU Pro data")
    parser.add_argument("--num_samples_per_dataset", type=int, default=10000, help="Number of samples to load from each dataset")
    parser.add_argument("--save_path", type=str, default="./model_checkpoints", help="Path to save model checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--min_len", type=int, default=0, help="Minimum length for filtering data")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for scheduler")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps interval to save checkpoints")
    parser.add_argument("--max_checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep")
    parser.add_argument("--save_best_only", action='store_true', help="Save only the best model based on test accuracy")
    parser.add_argument("--multi_class", action='store_true', help="Use multi-class classification instead of binary")
    return parser

bbh_config = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa',
              'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton',
              'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects',
              'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting',
              'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names',
              'salient_translation_error_detection', 'snarks', 'sports_understanding',
              'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
              'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting'
              ]


if __name__ == "__main__":
    # Configuration
    parser = argument_parser()
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("Loading data...")
    if args.multi_class:
        # Load BBH dataset
        bbh_samples = []
        for config in bbh_config:
            try:
                bbh_data = load_dataset(args.bbh_datasets, config, split="test")
                bbh_sub_samples = bbh_data.to_pandas().to_dict('records')
                bbh_samples.extend(bbh_sub_samples)
            except Exception as e:
                print(f"  Warning: Could not load BBH/{config}: {e}")
                continue
        random.shuffle(bbh_samples)
        bbh_samples = bbh_samples[:min(args.num_samples_per_dataset, len(bbh_samples))]
        
        # Load Math dataset
        math_data = load_dataset(args.math_datasets, split="train")
        math_samples = math_data.to_pandas().to_dict('records')
        random.shuffle(math_samples)
        math_samples = math_samples[:min(args.num_samples_per_dataset, len(math_samples))]

        # Load MMLU Pro dataset
        mmlu_pro_data = load_dataset(args.mmlu_pro_datasets, split="test")
        mmlu_pro_samples = mmlu_pro_data.to_pandas().to_dict('records')
        random.shuffle(mmlu_pro_samples)
        mmlu_pro_samples = mmlu_pro_samples[:min(args.num_samples_per_dataset, len(mmlu_pro_samples))]

        data_dict = {
            'bbh': bbh_samples,
            'math': math_samples,
            'mmlu_pro': mmlu_pro_samples
        }
        
        print(f"Loaded {len(bbh_samples)} samples from BBH dataset")
        print(f"Loaded {len(math_samples)} samples from Math dataset")
        print(f"Loaded {len(mmlu_pro_samples)} samples from MMLU Pro dataset")

        
        # Print sample data from each dataset
        print("\n" + "="*60)
        print("Sample data from each dataset:")
        print("="*60)

        if bbh_samples:
            print("\nBBH sample:")
            print(bbh_samples[0])

        if math_samples:
            print("\nMath sample:")
            print(math_samples[0])

        if mmlu_pro_samples:
            print("\nMMLU Pro sample:")
            print(mmlu_pro_samples[0])

        print("="*60 + "\n")

        data = preprocess_task_data(data_dict)

        print(f"Total samples after preprocessing: {len(data)}")

    else:
        print(f"Loading query difficulty data from: {args.query_difficulty_path}")
        data = load_jsonl(args.query_difficulty_path)
        print(f"Loaded {len(data)} queries with difficulty scores")
        print(f"Sample data: {data[0]}")
    
    random.shuffle(data)
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    print("Creating datasets...")
    if args.multi_class:
        num_classes = 3  # Example for multi-class classification
        print(f"Using {num_classes} classes for multi-class classification.")
        train_dataset = MultiClassClassificationDataset(train_data, tokenizer, args.max_length)
        test_dataset = MultiClassClassificationDataset(test_data, tokenizer, args.max_length)
    else:
        print("Using difficulty prediction with soft labels (0-1 normalized from difficulty scores).")
        train_dataset = DifficultyDataset(train_data, tokenizer, args.max_length)
        test_dataset = DifficultyDataset(test_data, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    print("Creating model...")
    if args.multi_class:
        model = MultiClassClassifier(
            model_name=args.model_name,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_classes=num_classes
        ).to(device)
    else:
        model = BERTBinaryClassifier(
            model_name=args.model_name,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Check if resuming from checkpoint
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"\n{'='*60}")
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        print(f"{'='*60}\n")
    else:
        print("\nStarting training from scratch...")

    if args.multi_class:
        save_path = os.path.join(args.save_path, "embeddinggemma_multi_class_classifier")
    elif os.path.exists(args.query_difficulty_path) and data and 'difficulty' in data[0]:
        save_path = os.path.join(args.save_path, "embeddinggemma_difficulty_predictor")
    else:
        save_path = os.path.join(args.save_path, "embeddinggemma_binary_classifier")
    
    print(f"Batch size: {args.batch_size}, Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    best_model_path = train_loops(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        save_path=save_path,
        device=device,
        warmup_steps=args.warmup_steps,
        save_best_only=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        max_checkpoints=args.max_checkpoints,
        resume_from_checkpoint=args.resume_from_checkpoint,
        multi_class=args.multi_class
    )
    
    print(f"\nTraining complete! Final model saved at: {best_model_path}")
