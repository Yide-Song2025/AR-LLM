import os
import json

import torch
from torch.utils.data import Dataset


class DifficultyDataset(Dataset):
    """
    Dataset for difficulty prediction using soft labels from query_difficulty.jsonl.
    The difficulty score (0-100) is normalized to (0-1) to serve as soft labels.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = []
        self.labels = []  # Soft labels: difficulty / 100
        
        for sample in data:
            query = sample.get("query", "").strip()
            difficulty = sample.get("difficulty", 0.0)
            
            # Skip empty queries
            if not query:
                continue
            
            # Normalize difficulty from [0, 100] to [0, 1] for soft label
            soft_label = difficulty / 100.0
            
            self.prompts.append(query)
            self.labels.append(soft_label)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }
    

class MultiClassClassificationDataset(Dataset):
    """
    Dataset for multi-class classification: predict winner among model_a, model_b, or tie.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = []
        self.labels = []
        
        for sample in data:
            prompt_text = sample["prompt"]
            
            if sample["task_name"] == "bbh":
                label = 0
            elif sample["task_name"] == "math":
                label = 1
            elif sample["task_name"] == "mmlu_pro":
                label = 2
            else:
                continue
            if prompt_text and len(prompt_text.strip()) > 0:
                self.prompts.append(prompt_text)
                self.labels.append(label)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
    

class InferenceDataset(Dataset):
    """
    Dataset for inference.
    """
    def __init__(self, data, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        query = sample.get("query", "")
        query_id = sample.get("query_id", "")
        
        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'query': query,
            'query_id': query_id
        }


def load_jsonl(file_path, model=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                if model is not None:
                    record = json.loads(line)
                    if record.get('model', '') == model:
                        data.append(record)
                else:
                    data.append(json.loads(line))
    return data


def save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, best_test_acc, train_losses, test_losses, test_acces, save_path, checkpoint_name="checkpoint.pth"):
    """
    Save checkpoint with all training state information.
    """
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_test_acc': best_test_acc,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_acces': test_acces,
    }
    checkpoint_path = os.path.join(save_path, checkpoint_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    Load checkpoint and restore training state.
    Supports both full checkpoints and old-style model-only weights.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a full checkpoint or just model weights
    if 'model_state_dict' in checkpoint:
        # Full checkpoint with all training state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'batch_idx': checkpoint['batch_idx'],
            'best_test_acc': checkpoint['best_test_acc'],
            'train_losses': checkpoint['train_losses'],
            'test_losses': checkpoint['test_losses'],
            'test_acces': checkpoint['test_acces'],
        }
    else:
        # Old-style checkpoint (only model weights)
        print("Warning: Loading old-style checkpoint (model weights only).")
        print("Optimizer and scheduler state will be reset.")
        model.load_state_dict(checkpoint)
        
        # Return default values for training state
        return {
            'epoch': 0,
            'batch_idx': 0,
            'best_test_acc': -1,
            'train_losses': [],
            'test_losses': [],
            'test_acces': [],
        }


def load_file(data, *file_paths):
    """
    Load data from JSON files.
    """
    for path in file_paths:
        with open(path, 'r') as file:
            for line in file:
                dic = json.loads(line)
                data.append(dic)
    return data


def preprocess_data(data, min_len=0, multi_class=False):
    """
    Preprocess data: extract first turn of prompt and filter by length.
    """
    def get_first_turn(prompt_str):
        try:
            return json.loads(prompt_str)[0].strip()
        except:
            return prompt_str.strip()
    
    data["prompt"] = data["prompt"].apply(get_first_turn)
    data = data.loc[data["prompt"].apply(len) >= min_len]
    data = data[["id", "model_a", "model_b", "prompt", "winner_model_a", "winner_model_b", "winner_tie"]]
    return data.to_dict(orient='records')


def preprocess_task_data(data_dict):
    """
    Preprocess data for specific tasks.
    """
    processed_data = []
    id = 0
    for task, samples in data_dict.items():
        if task == "bbh":
            for sample in samples:
                processed_data.append({
                    "id": id,
                    "task_name": task,
                    "prompt": sample["input"]
                })
                id += 1
        elif task == "math":
            for sample in samples:
                processed_data.append({
                    "id": id,
                    "task_name": task,
                    "prompt": sample["problem"]
                })
                id += 1
        elif task == "mmlu_pro":
            for sample in samples:
                processed_data.append({
                    "id": id,
                    "task_name": task,
                    "prompt": sample["question"]
                })
                id += 1
        else:
            print(f"Unknown task: {task}")
    
    return processed_data


def load_model_weights(model, checkpoint_path, device):
    """
    Load model weights from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a full checkpoint or just model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Model weights loaded successfully.")