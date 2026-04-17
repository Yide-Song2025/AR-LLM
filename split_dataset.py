import json
import random
import os

random.seed(42)

input_path = "data/feature_vectors/ID/data_samples_features.jsonl"
train_path = "data/feature_vectors/ID/data_samples_features_train.jsonl"
test_path = "data/feature_vectors/ID/data_samples_features_test.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

split_idx = int(len(lines) * 0.9)
train_lines = lines[:split_idx]
test_lines = lines[split_idx:]

with open(train_path, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(test_path, "w", encoding="utf-8") as f:
    f.writelines(test_lines)

print(f"Total: {len(lines)}")
print(f"Train: {len(train_lines)} ({len(train_lines)/len(lines)*100:.1f}%)")
print(f"Test:  {len(test_lines)} ({len(test_lines)/len(lines)*100:.1f}%)")
print(f"Saved to:\n  {train_path}\n  {test_path}")
