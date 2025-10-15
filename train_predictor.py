#!/usr/bin/env python3
"""
train_predictor.py - Train attention-based expert predictor
Uses WikiText-103 train split for training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import os

# Optional wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")

class AttentionBasedExpertPredictor(nn.Module):
    """Predictor model for expert selection."""
    def __init__(self, hidden_dim=2048, n_moe_layers=22, n_experts=60, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_moe_layers = n_moe_layers  # 22 layers (MoE 2-23)
        self.n_experts = n_experts

        self.fc1 = nn.Linear(hidden_dim, 2048)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2048, n_moe_layers * n_experts)

    def forward(self, attention_output):
        # Extract dimensions properly
        if len(attention_output.shape) == 3:
            batch_size, seq_len, _ = attention_output.shape
        elif len(attention_output.shape) == 2:
            # Single sequence without batch dimension
            attention_output = attention_output.unsqueeze(1)  # Add seq_len dimension
            batch_size, seq_len, _ = attention_output.shape
        else:
            raise ValueError(f"Unexpected attention_output shape: {attention_output.shape}")

        x = self.fc1(attention_output)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # CRITICAL FIX: Maintain batch and sequence dimensions separately
        # Instead of view(-1, ...) which merges batch and seq_len, we properly reshape:
        expert_scores = x.view(batch_size, seq_len, self.n_moe_layers, self.n_experts)

        return expert_scores

class ExpertPredictionDataset(Dataset):
    """Dataset loader for training data from WikiText-103."""
    def __init__(self, data_dir, file_pattern="training_data_*.h5", max_samples=None):
        self.data_files = sorted(Path(data_dir).glob(file_pattern))
        print(f"Found {len(self.data_files)} data files")

        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in {data_dir} with pattern {file_pattern}")

        # Load metadata from first file
        with h5py.File(self.data_files[0], 'r') as f:
            self.hidden_dim = f.attrs['hidden_dim']
            self.n_moe_layers = f.attrs['n_moe_layers_to_predict']  # 22 layers
            self.n_experts = f.attrs['n_experts']
            print(f"Dataset: {f.attrs['dataset']}, Split: {f.attrs['split']}")

        # Count total samples
        self.file_offsets = [0]
        for data_file in self.data_files:
            with h5py.File(data_file, 'r') as f:
                n_samples = f.attrs['n_samples']
                self.file_offsets.append(self.file_offsets[-1] + n_samples)

        self.total_samples = self.file_offsets[-1]

        # Limit samples if max_samples is specified
        if max_samples is not None and max_samples < self.total_samples:
            self.total_samples = max_samples
            print(f"Limiting to {self.total_samples} samples (max_samples={max_samples})")
        else:
            print(f"Total samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which file contains this sample
        file_idx = 0
        for i in range(len(self.file_offsets) - 1):
            if self.file_offsets[i] <= idx < self.file_offsets[i + 1]:
                file_idx = i
                break

        local_idx = idx - self.file_offsets[file_idx]

        # Load from file
        with h5py.File(self.data_files[file_idx], 'r') as f:
            attention_output = f['attention_outputs'][local_idx]
            gating_scores = f['gating_scores'][local_idx]  # [22, 60] - full distribution

        return {
            'attention_output': torch.from_numpy(attention_output).float(),
            'gating_scores': torch.from_numpy(gating_scores).float()  # Ground truth
        }

def kl_divergence_loss(predicted_logits, true_distribution):
    """
    Compute KL divergence loss between predicted and true gating distributions.

    Args:
        predicted_logits: [batch, n_moe_layers, n_experts] - predicted logits (before softmax)
        true_distribution: [batch, n_moe_layers, n_experts] - ground truth gating scores (after softmax)

    Returns:
        loss: scalar KL divergence loss
    """
    # Apply log_softmax to predicted logits
    log_pred = torch.log_softmax(predicted_logits, dim=-1)

    # Cross-entropy: -sum(true * log(pred))
    loss = -(true_distribution * log_pred).sum(dim=-1).mean()

    return loss

def mse_loss(predicted_logits, true_distribution):
    """
    Compute MSE loss between predicted and true gating distributions.

    Args:
        predicted_logits: [batch, n_moe_layers, n_experts] - predicted logits
        true_distribution: [batch, n_moe_layers, n_experts] - ground truth gating scores

    Returns:
        loss: scalar MSE loss
    """
    # Apply softmax to predicted logits to get distribution
    pred_distribution = torch.softmax(predicted_logits, dim=-1)

    # MSE between distributions
    loss = torch.nn.functional.mse_loss(pred_distribution, true_distribution)

    return loss

def calculate_top_k_accuracy(predicted_logits, true_distribution, k=4):
    """
    Calculate top-k accuracy: fraction of true top-k experts in predicted top-k.

    Args:
        predicted_logits: [batch, n_moe_layers, n_experts] - predicted logits
        true_distribution: [batch, n_moe_layers, n_experts] - ground truth scores
        k: top-k to evaluate (default 4)

    Returns:
        accuracy: fraction of overlap in top-k sets
    """
    batch_size, n_moe_layers, n_experts = predicted_logits.shape

    # Get top-k predicted experts
    _, predicted_top_k = torch.topk(predicted_logits, k, dim=-1)  # [batch, n_moe_layers, k]

    # Get top-k true experts
    _, true_top_k = torch.topk(true_distribution, k, dim=-1)  # [batch, n_moe_layers, k]

    # Calculate intersection
    correct = 0
    total = 0

    for b in range(batch_size):
        for layer in range(n_moe_layers):
            pred_set = set(predicted_top_k[b, layer].cpu().numpy())
            true_set = set(true_top_k[b, layer].cpu().numpy())

            intersection = len(pred_set & true_set)
            correct += intersection
            total += k

    return correct / total if total > 0 else 0.0

def train_epoch(model, dataloader, optimizer, device, epoch, loss_fn='kl', use_wandb=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_accuracy = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        attention_output = batch['attention_output'].to(device)
        gating_scores = batch['gating_scores'].to(device)  # [batch, 22, 60]

        # Forward pass
        predicted_logits = model(attention_output)  # [batch, 22, 60]

        # Compute loss
        if loss_fn == 'kl':
            loss = kl_divergence_loss(predicted_logits, gating_scores)
        elif loss_fn == 'mse':
            loss = mse_loss(predicted_logits, gating_scores)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            accuracy = calculate_top_k_accuracy(predicted_logits, gating_scores, k=4)

        total_loss += loss.item()
        total_accuracy += accuracy
        n_batches += 1

        pbar.set_postfix({'loss': total_loss / n_batches, 'acc': total_accuracy / n_batches})

        # Log to wandb (batch level)
        if use_wandb and n_batches % 10 == 0:
            wandb.log({
                'batch_loss': loss.item(),
                'batch_accuracy': accuracy,
            })

    return total_loss / n_batches, total_accuracy / n_batches

def validate(model, dataloader, device, loss_fn='kl'):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            attention_output = batch['attention_output'].to(device)
            gating_scores = batch['gating_scores'].to(device)

            predicted_logits = model(attention_output)

            if loss_fn == 'kl':
                loss = kl_divergence_loss(predicted_logits, gating_scores)
            elif loss_fn == 'mse':
                loss = mse_loss(predicted_logits, gating_scores)
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")

            accuracy = calculate_top_k_accuracy(predicted_logits, gating_scores, k=4)

            total_loss += loss.item()
            total_accuracy += accuracy
            n_batches += 1

    return total_loss / n_batches, total_accuracy / n_batches

def main():
    # Configuration
    config = {
        'data_dir': 'predictor_training_data',
        'output_dir': 'predictor_checkpoints',
        'hidden_dim': 2048,
        'n_moe_layers': 22,  # Predict for MoE layers 2-23 (skip 0-1 which are GPU-resident)
        'n_experts': 60,
        'dropout': 0.1,
        'batch_size': 256,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'val_split': 0.1,
        'max_samples': 50000,  # Limit dataset size for faster training (None = use all)
        'loss_function': 'kl',  # 'kl' or 'mse'
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_wandb': WANDB_AVAILABLE,
        'wandb_project': 'fiddler-expert-predictor',
        'wandb_run_name': 'attention-based-predictor-v1'
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    # Initialize wandb
    use_wandb = config['use_wandb']
    if use_wandb:
        wandb.init(
            project=config['wandb_project'],
            name=config['wandb_run_name'],
            config=config
        )

    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*80)

    # Load dataset
    full_dataset = ExpertPredictionDataset(config['data_dir'], max_samples=config.get('max_samples'))

    # Split train/val
    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=4)

    print(f"\nTrain samples: {train_size}")
    print(f"Val samples: {val_size}")

    # Create model
    model = AttentionBasedExpertPredictor(
        hidden_dim=config['hidden_dim'],
        n_moe_layers=config['n_moe_layers'],
        n_experts=config['n_experts'],
        dropout=config['dropout']
    ).to(config['device'])

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Loss function: {config['loss_function']}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    best_val_loss = float('inf')
    training_history = []

    for epoch in range(config['num_epochs']):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*80}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, config['device'],
            epoch + 1, loss_fn=config['loss_function'], use_wandb=use_wandb
        )
        val_loss, val_acc = validate(model, val_loader, config['device'], loss_fn=config['loss_function'])

        print(f"\nTrain Loss: {train_loss:.4f}, Train Top-4 Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Top-4 Acc: {val_acc:.4f}")

        # Log to wandb (epoch level)
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })

        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'config': config
        }

        checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save(checkpoint, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config['output_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")

    # Save training history
    with open(os.path.join(config['output_dir'], 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Trained on WikiText-103 train split")
    print(f"Checkpoints saved to: {config['output_dir']}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
