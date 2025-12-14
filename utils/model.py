import os
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, get_linear_schedule_with_warmup

from utils.utils import save_checkpoint, load_checkpoint

class BERTBinaryClassifier(nn.Module):
    """
    Binary classifier using EmbeddingGemma for prompt encoding and model embeddings.
    """
    def __init__(self, model_name, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.encoder.config.hidden_size, hidden_dim)  # prompt, model_a, model_b embeddings
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
    
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        prompt_embed = outputs.last_hidden_state.mean(dim=1)  # [batch_size, embed_dim]
        
        x = self.dropout(prompt_embed)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.classifier(x).squeeze(-1)
        
        return logits
    

class MultiClassClassifier(nn.Module):
    """
    Multi-class classifier using EmbeddingGemma for prompt encoding and model embeddings.
    """
    def __init__(self, model_name, hidden_dim=256, num_classes=3, dropout=0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.encoder.config.hidden_size, hidden_dim)  # prompt, model_a, model_b embeddings
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        prompt_embed = outputs.last_hidden_state.mean(dim=1)  # [batch_size, embed_dim]
        
        x = self.dropout(prompt_embed)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits

def evaluator(model, test_loader, device, multi_class=False):
    """
    Evaluate the model on test data.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum") if not multi_class else nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            if multi_class:
                loss = loss_fn(logits, labels)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
            
            else:
                loss = loss_fn(logits, labels)
                binary_labels = torch.sigmoid(labels) > 0.5
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds == binary_labels).sum().item()
            
            total_loss += loss.item()
            num_samples += labels.shape[0]
    
    model.train()
    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples
    return avg_loss, accuracy


def train_loops(
    model,
    train_loader,
    test_loader,
    lr,
    weight_decay,
    num_epochs,
    save_path,
    device="cuda",
    warmup_steps=500,
    save_best_only=False,
    gradient_accumulation_steps=2,
    save_steps=500,
    max_checkpoints=5,
    resume_from_checkpoint=None,
    multi_class=False
):
    """
    Training loop with evaluation and model saving.
    Includes gradient accumulation for effective larger batch sizes.
    Supports resuming from checkpoint.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean") if not multi_class else nn.CrossEntropyLoss(reduction="mean")
    
    # Learning rate scheduler
    # Adjust total steps for gradient accumulation
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize training state
    start_epoch = 0
    start_batch_idx = 0
    best_test_acc = -1
    train_losses = []
    test_losses = []
    test_acces = []
    
    # Load checkpoint if resuming
    if resume_from_checkpoint:
        checkpoint_data = load_checkpoint(resume_from_checkpoint, model, optimizer, scheduler, device)
        if checkpoint_data:
            start_epoch = checkpoint_data['epoch']
            start_batch_idx = checkpoint_data['batch_idx']
            best_test_acc = checkpoint_data['best_test_acc']
            train_losses = checkpoint_data['train_losses']
            test_losses = checkpoint_data['test_losses']
            test_acces = checkpoint_data['test_acces']
            print(f"Resuming training from epoch {start_epoch}, batch {start_batch_idx}")
            print(f"Best test accuracy so far: {best_test_acc:.4f}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss_sum = 0.0
        num_batches = 0
        
        # Create progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in pbar:
            # Skip batches if resuming from checkpoint
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            loss = loss_fn(logits, labels)
            
            # Normalize loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Only update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss_sum += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})

            # Save checkpoint periodically
            if not save_best_only and (batch_idx + 1) % save_steps == 0:
                # Save training checkpoint (full state)
                save_checkpoint(
                    model, optimizer, scheduler, epoch, batch_idx + 1,
                    best_test_acc, train_losses, test_losses, test_acces,
                    save_path, checkpoint_name="latest_checkpoint.pth"
                )
                
                model_path = os.path.join(save_path, f"model_epoch_{epoch}_batch_{batch_idx + 1}.pth")
                torch.save(model.state_dict(), model_path)
                
                tqdm.write(f"\nSaved checkpoint and model to {save_path}")           
                checkpoint_files = sorted(
                    [f for f in os.listdir(save_path) if f.startswith("model_epoch_") and f.endswith(".pth") and not f.startswith("best_")],
                    key=lambda x: os.path.getmtime(os.path.join(save_path, x))
                )
                if len(checkpoint_files) > max_checkpoints:
                    for old_checkpoint in checkpoint_files[:-max_checkpoints]:
                        old_path = os.path.join(save_path, old_checkpoint)
                        os.remove(old_path)
                        tqdm.write(f"Removed old model weights: {old_path}")
        
        # Reset start_batch_idx after first epoch
        if epoch == start_epoch:
            start_batch_idx = 0
        
        # Update any remaining gradients at the end of epoch
        if len(train_loader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss_sum / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        test_loss, test_acc = evaluator(model, test_loader, device, multi_class=multi_class)
        test_losses.append(test_loss)
        test_acces.append(test_acc)
        
        tqdm.write(f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Best Test Acc: {best_test_acc:.4f}")
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, 0,
            best_test_acc, train_losses, test_losses, test_acces,
            save_path, checkpoint_name="latest_checkpoint.pth"
        )
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        model_path = os.path.join(save_path, f"model_loss_{avg_train_loss:.4f}_acc_{test_acc:.4f}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        tqdm.write(f"\nSaved best model to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(save_path, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    tqdm.write(f"\nSaved final model to {final_model_path}")
    
    return final_model_path
