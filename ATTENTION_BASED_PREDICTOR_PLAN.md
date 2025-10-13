# Attention-Based Expert Predictor Implementation Plan

## Executive Summary

This document provides a detailed plan to implement a learned predictor that uses the first layer's attention output to predict which experts will be needed in subsequent MoE layers (layers 2-23). This approach aims to improve upon pattern-based prefetching by learning complex relationships between early-layer representations and future expert selections.

**Key Innovation**: Instead of using token position patterns (current approach) or gating predictions (7.5% overlap), we use first-layer attention outputs as rich semantic features to predict expert usage across all downstream layers.

## 1. Background & Motivation

### Current System
- **Model**: Qwen1.5-MoE-A2.7B with 24 MoE layers (layers 0-23 in MoE indexing)
- **Experts**: 60 experts per layer, top-k=4 routing
- **Current prefetch**: Token position-based patterns (`expert_usage_patterns_qwen.json`)
- **Current limitation**: Gating-based prediction (layer X → X+1) has only 7.5% overlap
- **GPU-resident layers**: First 2 MoE layers (0-1) kept on GPU permanently
- **Prediction target**: MoE layers 2-23 (22 layers to predict)

### Why First Layer Attention?
1. **Rich semantic features**: Attention outputs capture token relationships and context
2. **Early availability**: Layer 0 attention completes before any MoE layers
3. **Sufficient lookahead**: Can predict 22 MoE layers (2-23) from a single forward pass
4. **Proven approach**: Attention representations are known to capture high-level semantics

## 2. Architecture Overview

### 2.1 Model Structure

```
Input: First layer attention output for each token
  ↓
Hidden Layer 1: Linear(hidden_dim → 2048) + ReLU + Dropout(0.1)
  ↓
Hidden Layer 2: Linear(2048 → n_moe_layers * n_experts)
  ↓
Output: Scores for each expert in each layer (22 layers × 60 experts = 1320 scores per token)
  ↓
Reshape: [batch, seq_len, n_moe_layers, n_experts] = [batch, seq_len, 22, 60]
```

### 2.2 Detailed Specifications

**Input Dimensions**:
- Qwen1.5-MoE-A2.7B has `hidden_dim = 2048` (verify with `model.config.hidden_size`)
- Input shape: `[batch_size, sequence_length, 2048]`

**Output Dimensions**:
- 22 MoE layers to predict (MoE layers 2-23; layers 0-1 are GPU-resident)
- 60 experts per layer
- Output shape: `[batch_size, sequence_length, 22, 60]`
- Each value is a score (logit) for that expert at that layer

**Model Capacity**:
- Parameters: ~6.9M (2048×2048 + 2048×1320 ≈ 6.9M weights + biases)
- Memory: ~28MB in fp32, ~14MB in fp16
- Inference time: <1ms per forward pass on GPU

### 2.3 PyTorch Implementation Template

```python
import torch
import torch.nn as nn

class AttentionBasedExpertPredictor(nn.Module):
    """
    Predicts expert selections for MoE layers based on first layer attention output.

    Note: Predicts for MoE layers 2-23 (22 layers). Layers 0-1 are kept on GPU permanently.
    """
    def __init__(self, hidden_dim=2048, n_moe_layers=22, n_experts=60, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_moe_layers = n_moe_layers  # 22 layers to predict (MoE layers 2-23)
        self.n_experts = n_experts

        # Two-layer MLP
        self.fc1 = nn.Linear(hidden_dim, 2048)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2048, n_moe_layers * n_experts)

    def forward(self, attention_output):
        """
        Args:
            attention_output: [batch, seq_len, hidden_dim] - First layer attention output

        Returns:
            expert_scores: [batch, seq_len, n_moe_layers, n_experts] - Logits for each expert
                          Shape: [batch, seq_len, 22, 60]
        """
        batch_size, seq_len, _ = attention_output.shape

        # Two-layer MLP
        x = self.fc1(attention_output)  # [batch, seq_len, 2048]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, seq_len, 22 * 60]

        # Reshape to per-layer expert scores
        expert_scores = x.view(batch_size, seq_len, self.n_moe_layers, self.n_experts)

        return expert_scores
```

## 3. Data Collection Pipeline

### 3.1 Overview

**Dataset**: WikiText-103-raw-v1
- **Train split**: For collecting training data (~100M tokens)
- **Test split**: Reserved for final evaluation (do NOT use for training)
- **Why WikiText**:
  - Standard benchmark for language modeling
  - Clean, well-defined train/test splits
  - Appropriate size for training neural predictors
  - Representative of diverse text domains

### 3.2 Data Collection Script: `collect_training_data.py`

**Objective**: Run the model on WikiText-103 train split and record:
1. **Input features**: First layer attention outputs [n_samples, 2048]
2. **Ground truth labels**: Full gating scores for all 60 experts across MoE layers 2-23 [n_samples, 22, 60]

**Why full gating scores (not just top-4)?**
- **Richer learning signal**: 60 continuous values vs 4 discrete indices
- **Captures relative importance**: Not just which experts, but how confident the gating is
- **Better gradients**: Continuous targets provide smoother optimization
- **Learns the distribution**: Model learns the complete expert selection distribution

**Implementation**:

```python
#!/usr/bin/env python3
"""
collect_training_data.py - Collect training data for attention-based expert predictor
Uses WikiText-103 train split only (test split reserved for evaluation)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import h5py
import numpy as np
from tqdm import tqdm
import os

class DataCollector:
    def __init__(self, model_name="Qwen/Qwen1.5-MoE-A2.7B", output_dir="predictor_training_data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Find MoE layers
        self.moe_layers = []
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts'):
                self.moe_layers.append(idx)

        print(f"Found {len(self.moe_layers)} MoE layers: {self.moe_layers}")

        # We will predict for MoE layers 2-23 (skipping 0-1 which are GPU-resident)
        self.moe_layers_to_predict = self.moe_layers[2:]  # Skip first 2
        print(f"Will predict for {len(self.moe_layers_to_predict)} MoE layers: {self.moe_layers_to_predict}")

        # Storage
        self.attention_outputs = []  # Store first layer attention outputs
        self.gating_scores_data = []  # Store full gating scores (all 60 experts)

        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture attention outputs and full gating scores."""

        # Hook for first layer attention output (Layer 0)
        def attention_hook(module, input, output):
            # output is a tuple: (hidden_states, attention_weights, ...)
            # We want hidden_states after attention
            hidden_states = output[0]  # [batch, seq_len, hidden_dim]
            self.current_attention_output = hidden_states.detach().cpu()

        # Register on layer 0's self_attn
        self.model.model.layers[0].self_attn.register_forward_hook(attention_hook)

        # Hook for MoE gating scores (ALL 60 experts, not just top-4)
        def moe_hook(layer_idx):
            def hook(module, input, output):
                hidden_states = input[0]
                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_states_flat = hidden_states.view(-1, hidden_dim)

                # Get router logits for ALL 60 experts
                router_logits = module.gate(hidden_states_flat)

                # Apply softmax to get gating scores - this is our ground truth
                # Shape: [batch * seq_len, 60]
                gating_scores = F.softmax(router_logits, dim=1, dtype=torch.float)

                # Store full gating distribution for all 60 experts
                if not hasattr(self, 'current_gating_scores'):
                    self.current_gating_scores = {}

                self.current_gating_scores[layer_idx] = gating_scores.detach().cpu()

                return output
            return hook

        # Register on all MoE layers
        for layer_idx in self.moe_layers:
            self.model.model.layers[layer_idx].mlp.register_forward_hook(moe_hook(layer_idx))

    def collect_from_text(self, text, max_length=512):
        """Collect data from a single text sample."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(self.device)

        if input_ids.shape[1] < 2:
            return  # Skip very short sequences

        # Reset storage
        self.current_attention_output = None
        self.current_gating_scores = {}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=False)

        # Verify we got the data
        if self.current_attention_output is None:
            print("Warning: No attention output captured")
            return

        if len(self.current_gating_scores) != len(self.moe_layers):
            print(f"Warning: Expected {len(self.moe_layers)} MoE outputs, got {len(self.current_gating_scores)}")
            return

        # Store the data
        batch_size, seq_len, hidden_dim = self.current_attention_output.shape

        # For each token in the sequence
        for token_idx in range(seq_len):
            # Get attention output for this token: [2048]
            attn_out = self.current_attention_output[0, token_idx, :].numpy()

            # Get FULL gating scores for this token across MoE layers 2-23
            # Shape: [22, 60] - complete distribution over all experts
            gating_scores = np.zeros((len(self.moe_layers_to_predict), 60), dtype=np.float32)

            for moe_idx, layer_idx in enumerate(self.moe_layers_to_predict):
                # Store all 60 expert scores (not just top-4)
                gating_scores[moe_idx] = self.current_gating_scores[layer_idx][token_idx].numpy()

            self.attention_outputs.append(attn_out)
            self.gating_scores_data.append(gating_scores)

    def collect_from_dataset(self, num_samples=50000, samples_per_file=1000):
        """Collect data from WikiText-103 train split ONLY."""
        print(f"Loading dataset: wikitext/wikitext-103-raw-v1 (train split)")
        print(f"Test split is RESERVED for evaluation - not used here")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

        file_idx = 0
        samples_in_current_file = 0

        for idx, example in enumerate(tqdm(dataset, desc="Collecting data", total=num_samples)):
            if idx >= num_samples:
                break

            text = example['text']
            if len(text.strip()) < 50:  # Skip very short texts
                continue

            self.collect_from_text(text)
            samples_in_current_file += 1

            # Save periodically to avoid memory issues
            if samples_in_current_file >= samples_per_file:
                self.save_data(file_idx)
                file_idx += 1
                samples_in_current_file = 0
                self.attention_outputs = []
                self.gating_scores_data = []

        # Save remaining data
        if len(self.attention_outputs) > 0:
            self.save_data(file_idx)

    def save_data(self, file_idx):
        """Save collected data to HDF5 file."""
        if len(self.attention_outputs) == 0:
            return

        output_path = os.path.join(self.output_dir, f"training_data_{file_idx:04d}.h5")
        print(f"\nSaving {len(self.attention_outputs)} samples to {output_path}")

        with h5py.File(output_path, 'w') as f:
            # Save attention outputs: [n_samples, 2048]
            attn_array = np.stack(self.attention_outputs)
            f.create_dataset('attention_outputs', data=attn_array, compression='gzip')

            # Save full gating scores: [n_samples, 22, 60]
            # This is the complete gating distribution - our ground truth labels
            gating_scores_array = np.stack(self.gating_scores_data)
            f.create_dataset('gating_scores', data=gating_scores_array, compression='gzip')

            # Metadata
            f.attrs['n_samples'] = len(self.attention_outputs)
            f.attrs['hidden_dim'] = self.attention_outputs[0].shape[0]
            f.attrs['n_moe_layers_to_predict'] = len(self.moe_layers_to_predict)
            f.attrs['moe_layers_to_predict'] = self.moe_layers_to_predict
            f.attrs['n_experts'] = 60
            f.attrs['dataset'] = 'wikitext-103-raw-v1'
            f.attrs['split'] = 'train'

        print(f"Saved successfully")

def main():
    collector = DataCollector()

    # Collect from WikiText-103 train split ONLY
    # Test split is reserved for final evaluation
    collector.collect_from_dataset(
        num_samples=50000,  # 50k samples from train split
        samples_per_file=1000
    )

    print("\n✅ Data collection complete!")
    print(f"Data saved to: {collector.output_dir}/")
    print(f"Test split of WikiText-103 reserved for evaluation")

if __name__ == "__main__":
    main()
```

### 3.3 Data Format

**HDF5 File Structure** (`training_data_XXXX.h5`):
```
/attention_outputs     [n_samples, 2048] - float32 (input features)
/gating_scores         [n_samples, 22, 60] - float32 (ground truth - FULL gating distribution)

Attributes:
  - n_samples: int
  - hidden_dim: 2048
  - n_moe_layers_to_predict: 22
  - moe_layers_to_predict: [2, 3, 4, ..., 23] (MoE layer indices)
  - n_experts: 60
  - dataset: "wikitext-103-raw-v1"
  - split: "train"
```

**Dataset Size Estimation**:
- 50k samples × 2048 float32 = ~400MB (attention outputs)
- 50k samples × 22 × 60 float32 = ~260MB (gating scores - full distribution)
- **Total**: ~660MB per 50k samples from WikiText train split

**Key Difference from Original Approach**:
- **Original**: Stored top-4 expert indices + weights [n_samples, 22, 4] - sparse, discrete
- **New (Better)**: Store all 60 expert scores [n_samples, 22, 60] - dense, continuous
- **Benefit**: Richer learning signal, better gradients, captures full distribution

## 4. Training Pipeline

### 4.1 Training Script: `train_predictor.py`

**Objective**: Train the predictor to minimize loss between predicted scores and actual gating scores for ALL 60 experts.

**Loss Function**:
```
For each token and each layer:
  - Ground truth: gating scores for all 60 experts (after softmax)
  - Predicted: predicted scores for all 60 experts (before softmax)
  - Loss: MSE or KL divergence between distributions

Options:
1. MSE Loss: Mean squared error between predicted and true scores
2. KL Divergence: KL(true_distribution || predicted_distribution)
3. Cross-Entropy: Equivalent to KL divergence with softmax

Recommendation: Use KL divergence (or cross-entropy) as it respects the
probability distribution nature of gating scores.
```

**Implementation**:

```python
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

class AttentionBasedExpertPredictor(nn.Module):
    """Predictor model (implementation from Section 2.3)"""
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
        batch_size, seq_len, _ = attention_output.shape if len(attention_output.shape) == 3 else (attention_output.shape[0], 1, attention_output.shape[1])

        if len(attention_output.shape) == 2:
            attention_output = attention_output.unsqueeze(1)

        x = self.fc1(attention_output)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        expert_scores = x.view(-1, self.n_moe_layers, self.n_experts)

        return expert_scores

class ExpertPredictionDataset(Dataset):
    """Dataset loader for training data from WikiText-103."""
    def __init__(self, data_dir, file_pattern="training_data_*.h5"):
        self.data_files = sorted(Path(data_dir).glob(file_pattern))
        print(f"Found {len(self.data_files)} data files")

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

    # KL(true || pred) = sum(true * log(true/pred))
    # = sum(true * (log(true) - log(pred)))
    # = sum(true * log(true)) - sum(true * log(pred))
    # For optimization, we only need the second term (first is constant)

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

def train_epoch(model, dataloader, optimizer, device, epoch, loss_fn='kl'):
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
        'loss_function': 'kl',  # 'kl' or 'mse'
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Load dataset
    full_dataset = ExpertPredictionDataset(config['data_dir'])

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

    # Create model
    model = AttentionBasedExpertPredictor(
        hidden_dim=config['hidden_dim'],
        n_moe_layers=config['n_moe_layers'],
        n_experts=config['n_experts'],
        dropout=config['dropout']
    ).to(config['device'])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Loss function: {config['loss_function']}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, config['device'],
            epoch + 1, loss_fn=config['loss_function']
        )
        val_loss, val_acc = validate(model, val_loader, config['device'], loss_fn=config['loss_function'])

        print(f"Train Loss: {train_loss:.4f}, Train Top-4 Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Top-4 Acc: {val_acc:.4f}")

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

    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Trained on WikiText-103 train split")

if __name__ == "__main__":
    main()
```

### 4.2 Training Configuration

**Hyperparameters**:
- Batch size: 256
- Learning rate: 1e-4 (Adam optimizer)
- Epochs: 10
- Dropout: 0.1
- Train/val split: 90/10 (from WikiText train split)
- Loss function: KL divergence (recommended) or MSE

**Expected Training Time**:
- 50k samples, batch size 256 = ~195 batches/epoch
- ~1-2 seconds per batch on V100 GPU
- ~5-10 minutes per epoch
- **Total**: ~1-2 hours for 10 epochs

**Why This Approach is Better**:
1. **Richer supervision**: Learning from full 60-dim distributions vs 4 discrete indices
2. **Better gradients**: Continuous targets provide smoother optimization landscape
3. **Captures uncertainty**: Model learns when gating is confident vs uncertain
4. **More robust**: Less sensitive to noise in top-4 boundary decisions

## 5. Evaluation on WikiText Test Set

### 5.1 Evaluation Script: `evaluate_predictor.py`

**Objective**: Evaluate the trained predictor on WikiText-103 **test split** (never seen during training).

**Metrics**:
1. **Top-4 Overlap Accuracy**: What fraction of the true top-4 experts appear in predicted top-4
2. **Jaccard Similarity**: IoU between predicted and true top-4 sets
3. **Precision@4**: Precision of top-4 predictions
4. **NDCG@4**: Normalized discounted cumulative gain

```python
#!/usr/bin/env python3
"""
evaluate_predictor.py - Evaluate predictor on WikiText-103 TEST split
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json

# Import predictor model
from train_predictor import AttentionBasedExpertPredictor

class PredictorEvaluator:
    def __init__(self, predictor_path, model_name="Qwen/Qwen1.5-MoE-A2.7B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load predictor
        checkpoint = torch.load(predictor_path, map_location=self.device)
        config = checkpoint['config']

        self.predictor = AttentionBasedExpertPredictor(
            hidden_dim=config['hidden_dim'],
            n_moe_layers=config['n_moe_layers'],
            n_experts=config['n_experts'],
            dropout=0.0
        ).to(self.device)

        self.predictor.load_state_dict(checkpoint['model_state_dict'])
        self.predictor.eval()

        print(f"Loaded predictor from epoch {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Val Top-4 Acc: {checkpoint['val_acc']:.4f}")

        # Load Qwen model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Find MoE layers
        self.moe_layers = []
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts'):
                self.moe_layers.append(idx)

        self.moe_layers_to_predict = self.moe_layers[2:]  # Skip first 2

        # Register hooks
        self._register_hooks()

        # Metrics storage
        self.metrics = {
            'top4_overlap': [],
            'jaccard': [],
            'precision_at_4': [],
            'ndcg_at_4': []
        }

    def _register_hooks(self):
        """Register hooks to capture attention and gating scores."""
        def attention_hook(module, input, output):
            self.current_attention_output = output[0].detach()

        self.model.model.layers[0].self_attn.register_forward_hook(attention_hook)

        def moe_hook(layer_idx):
            def hook(module, input, output):
                hidden_states = input[0]
                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_states_flat = hidden_states.view(-1, hidden_dim)

                router_logits = module.gate(hidden_states_flat)
                gating_scores = F.softmax(router_logits, dim=1, dtype=torch.float)

                if not hasattr(self, 'current_gating_scores'):
                    self.current_gating_scores = {}

                self.current_gating_scores[layer_idx] = gating_scores.detach()
                return output
            return hook

        for layer_idx in self.moe_layers:
            self.model.model.layers[layer_idx].mlp.register_forward_hook(moe_hook(layer_idx))

    def evaluate_sample(self, text, max_length=512):
        """Evaluate predictor on a single text sample."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(self.device)

        if input_ids.shape[1] < 2:
            return

        # Reset
        self.current_attention_output = None
        self.current_gating_scores = {}

        # Forward pass through model
        with torch.no_grad():
            _ = self.model(input_ids, use_cache=False)

        if self.current_attention_output is None:
            return

        if len(self.current_gating_scores) != len(self.moe_layers):
            return

        # Get predictor predictions
        with torch.no_grad():
            predicted_logits = self.predictor(self.current_attention_output)  # [1, seq_len, 22, 60]

        batch_size, seq_len, _ = self.current_attention_output.shape

        # Evaluate each token
        for token_idx in range(seq_len):
            for output_idx, layer_idx in enumerate(self.moe_layers_to_predict):
                # True top-4
                true_scores = self.current_gating_scores[layer_idx][token_idx]  # [60]
                _, true_top4 = torch.topk(true_scores, 4)
                true_top4_set = set(true_top4.cpu().numpy())

                # Predicted top-4
                pred_scores = predicted_logits[0, token_idx, output_idx, :]  # [60]
                _, pred_top4 = torch.topk(pred_scores, 4)
                pred_top4_set = set(pred_top4.cpu().numpy())

                # Calculate metrics
                intersection = len(true_top4_set & pred_top4_set)
                union = len(true_top4_set | pred_top4_set)

                # Top-4 overlap (out of 4)
                top4_overlap = intersection / 4.0

                # Jaccard similarity
                jaccard = intersection / union if union > 0 else 0.0

                # Precision@4
                precision = intersection / 4.0

                # NDCG@4 (simplified)
                ndcg = intersection / 4.0  # Simplified version

                self.metrics['top4_overlap'].append(top4_overlap)
                self.metrics['jaccard'].append(jaccard)
                self.metrics['precision_at_4'].append(precision)
                self.metrics['ndcg_at_4'].append(ndcg)

    def evaluate_dataset(self, num_samples=1000):
        """Evaluate on WikiText-103 TEST split."""
        print("Loading WikiText-103 TEST split...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

        for idx, example in enumerate(tqdm(dataset, desc="Evaluating", total=num_samples)):
            if idx >= num_samples:
                break

            text = example['text']
            if len(text.strip()) < 50:
                continue

            self.evaluate_sample(text)

        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS ON WIKITEXT-103 TEST SET")
        print("="*80)
        print(f"Samples evaluated: {idx + 1}")
        print(f"Total token-layer pairs: {len(self.metrics['top4_overlap'])}")
        print(f"\nTop-4 Overlap Accuracy: {np.mean(self.metrics['top4_overlap']):.4f}")
        print(f"Jaccard Similarity: {np.mean(self.metrics['jaccard']):.4f}")
        print(f"Precision@4: {np.mean(self.metrics['precision_at_4']):.4f}")
        print(f"NDCG@4: {np.mean(self.metrics['ndcg_at_4']):.4f}")

        # Save results
        results = {
            'num_samples': idx + 1,
            'num_token_layer_pairs': len(self.metrics['top4_overlap']),
            'top4_overlap_accuracy': float(np.mean(self.metrics['top4_overlap'])),
            'jaccard_similarity': float(np.mean(self.metrics['jaccard'])),
            'precision_at_4': float(np.mean(self.metrics['precision_at_4'])),
            'ndcg_at_4': float(np.mean(self.metrics['ndcg_at_4']))
        }

        with open('predictor_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to predictor_evaluation_results.json")

def main():
    evaluator = PredictorEvaluator(
        predictor_path="predictor_checkpoints/best_model.pt"
    )

    evaluator.evaluate_dataset(num_samples=1000)

if __name__ == "__main__":
    main()
```

### 5.2 Success Criteria

**Based on Top-4 Overlap Accuracy (on WikiText test set)**:

**Minimum Viable Performance**:
- Top-4 overlap > 40% (vs. 18.75% for gating-based layer X→X+1)
- Jaccard similarity > 0.30
- Justifies using the predictor over random guessing

**Target Performance**:
- Top-4 overlap > 60%
- Jaccard similarity > 0.50
- Comparable to or better than pattern-based on diverse prompts

**Stretch Goals**:
- Top-4 overlap > 75%
- Jaccard similarity > 0.65
- Strong predictor ready for production


## 6. Implementation Timeline & Milestones

### Phase 1: Data Collection (Week 1)
- [ ] Implement `collect_training_data.py`
- [ ] Test on small sample (100 examples from WikiText train)
- [ ] Run full collection on WikiText-103 train split (50k samples)
- [ ] Verify data format and integrity (HDF5 files, gating scores shape)
- **Deliverable**: Training data in `predictor_training_data/` (~660MB)

### Phase 2: Model Training (Week 2)
- [ ] Implement `train_predictor.py` with KL divergence loss
- [ ] Implement top-4 accuracy metric
- [ ] Run training (10 epochs, ~1-2 hours)
- [ ] Analyze training curves (loss decreasing, no overfitting)
- [ ] Verify top-4 accuracy > 40% on validation set
- **Deliverable**: Trained model in `predictor_checkpoints/best_model.pt`

### Phase 3: Evaluation (Week 3)
- [ ] Implement `evaluate_predictor.py`
- [ ] Run evaluation on WikiText-103 **test split** (1000 samples)
- [ ] Calculate top-4 overlap accuracy, Jaccard similarity
- [ ] Compare with baseline (18.75% for gating-based)
- [ ] Document results in `predictor_evaluation_results.json`
- **Deliverable**: Evaluation results showing predictor performance

### Phase 4: Integration with Fiddler (Week 4)
- [ ] Implement `src/fiddler/qwen_with_learned_prefetch.py`
- [ ] Test attention hook capture
- [ ] Test predictor inference integration
- [ ] Verify correctness on simple examples (output matches baseline)
- [ ] Implement `test_learned_prefetch.py`
- **Deliverable**: Working integration passing tests

### Phase 5: End-to-End Benchmarking (Week 5)
- [ ] Implement `benchmark_prediction_methods.py`
- [ ] Run benchmarks comparing baseline, pattern-based, and learned
- [ ] Measure actual inference speedup and hit rates
- [ ] Test with Fiddler CPU offloading enabled
- [ ] Document final results and recommendations
- **Deliverable**: Complete benchmarking results and production-ready implementation

## 7. Files to Create

**Core Implementation**:
1. `collect_training_data.py` - Data collection from WikiText-103 train split
2. `train_predictor.py` - Training with KL divergence loss
3. `evaluate_predictor.py` - Evaluation on WikiText-103 test split
4. `src/fiddler/qwen_with_learned_prefetch.py` - Integration with Fiddler
5. `test_learned_prefetch.py` - Testing script
6. `benchmark_prediction_methods.py` - Comparison benchmark

**Supporting Files**:
7. `predictor_training_data/` - Training data directory (HDF5 files)
8. `predictor_checkpoints/` - Model checkpoints directory
9. `predictor_evaluation_results.json` - Evaluation results
10. `ATTENTION_BASED_PREDICTOR_PLAN.md` - This document

**Documentation**:
11. Update `thoughts/20250929/guide.md` with predictor information
12. Create `PREDICTOR_RESULTS.md` with evaluation and benchmark results (after Phase 5)

## 8. Verification Checklist

Before marking implementation complete, verify:

### Data Collection
- [ ] 50k+ samples collected from WikiText-103 train split
- [ ] HDF5 files have correct format
- [ ] Attention outputs shape: [n_samples, 2048]
- [ ] Gating scores shape: [n_samples, 22, 60] (full distribution)
- [ ] All 22 MoE layers (2-23) represented
- [ ] Test split NOT used for training

### Training
- [ ] Model trains without errors
- [ ] Training loss decreases over epochs
- [ ] Validation loss decreases (no overfitting)
- [ ] Top-4 overlap accuracy > 40% on validation set
- [ ] Best model checkpoint saved with config
- [ ] KL divergence loss used (not weighted CE)

### Evaluation
- [ ] Evaluation runs on WikiText-103 test split
- [ ] Top-4 overlap accuracy calculated correctly
- [ ] Results show improvement over gating-based baseline (>18.75%)
- [ ] Jaccard similarity > 0.30
- [ ] Results saved to JSON file

### Integration
- [ ] Attention hook captures output correctly
- [ ] Predictor inference runs without errors
- [ ] Predictions have correct shape [batch, seq_len, 22, 60]
- [ ] Top-k experts extracted correctly from predictions
- [ ] Prefetch buffers populated correctly
- [ ] Generated text matches baseline (correctness check)

### Benchmarking
- [ ] Benchmark runs on all three methods (baseline, pattern, learned)
- [ ] Decode hit rate measured for learned predictor
- [ ] Actual inference speedup calculated vs. baseline
- [ ] Results documented with performance comparison table
- [ ] Integration with Fiddler CPU offload tested

### Production Readiness
- [ ] Code is clean and well-documented
- [ ] Error handling implemented (missing files, incorrect shapes, etc.)
- [ ] Memory usage is acceptable (<1GB for predictor)
- [ ] Predictor inference overhead is minimal (<5% of total time)
- [ ] Guide.md updated with usage instructions
- [ ] All scripts have proper argparse and help messages

## 9. Expected Results

### Prediction Accuracy (WikiText Test Set)
- **Top-4 Overlap**: 50-70% (vs. 18.75% for gating-based layer X→X+1)
- **Jaccard Similarity**: 0.40-0.60 (vs. 0.19 for gating-based)
- **Rationale**:
  - First layer attention captures semantic features that correlate with expert specialization
  - Full 60-dim gating distribution provides rich supervision
  - KL divergence loss optimizes for probability distribution matching

### Inference Hit Rates
- **Prefill Hit Rate**: 50-70%
  - Lower because predictions cover multiple tokens at once
  - Depends on prompt characteristics
- **Decode Hit Rate**: 70-90%
  - Higher because predictions are per-token during autoregressive generation
  - More critical for overall speedup
- **Overall Hit Rate**: 60-80%

### Performance Speedup (vs. Baseline)
- **Best Case** (hit rate >80%): 1.4-1.6x
  - Comparable to pattern-based (1.28x) but generalizes to new prompts
- **Expected Case** (hit rate 60-70%): 1.2-1.3x
  - Solid improvement with good generalization
- **Worst Case** (hit rate 40-50%): 1.05-1.1x
  - Small predictor overhead (~0.5-1ms) limits downside

### Comparison with Existing Approaches

| Metric | Baseline | Pattern-Based | Learned (Expected) |
|--------|----------|--------------|-------------------|
| **Generalization** | N/A | Poor (fixed patterns) | Good (learned features) |
| **Decode Hit Rate** | 0% | 100% (training prompts only) | 70-90% (all prompts) |
| **Speedup (BS=1)** | 1.0x | 1.28x | 1.2-1.4x |
| **Data Requirement** | None | Minimal (single prompt) | Moderate (50k samples) |
| **Training Time** | None | None | ~2 hours |
| **Overhead** | None | Negligible | <1ms predictor inference |

**Key Advantages of Learned Predictor**:
1. **Generalizes to unseen prompts** (unlike pattern-based)
2. **Better than random guessing** on diverse tasks
3. **Learns semantic relationships** between attention and expert usage
4. **Minimal overhead** (<5% of inference time)
5. **Scalable approach** (can improve with more data/better architecture)

**When to Use Each Approach**:
- **Baseline**: No GPU memory constraints
- **Pattern-Based**: Fixed set of prompts (e.g., serving specific queries)
- **Learned**: Diverse prompts, production deployment with varying workloads
- **Hybrid**: Use pattern-based when available, fall back to learned predictor

### Expected Training Curves
- **Training Loss**: Should decrease from ~3.0 to ~1.5 over 10 epochs
- **Validation Loss**: Should track training loss closely (gap <0.2 indicates good generalization)
- **Top-4 Accuracy**: Should increase from ~30% to 50-70% over training

### Failure Modes & Mitigation
1. **Low accuracy (<40%)**:
   - Try deeper architecture (3-4 layers)
   - Collect more data (100k samples)
   - Use data augmentation
2. **High inference overhead (>10%)**:
   - Quantize predictor to INT8
   - Reduce hidden dimension
   - Cache predictions across multiple forward passes
3. **Poor generalization to test set**:
   - Increase dropout
   - Add L2 regularization
   - Train on more diverse data sources

## 10. Alternative Approaches & Future Work

### Short-Term Improvements
1. **Ensemble predictor**: Combine learned + pattern-based predictions
2. **Per-layer predictors**: Train separate predictors for each layer
3. **Attention from multiple layers**: Use layers 0-2 as input features
4. **Temperature scaling**: Calibrate predicted confidence scores

### Long-Term Research Directions
1. **Transformer-based predictor**: Use 2-3 layer transformer encoder
2. **Online learning**: Update predictor based on recent inference patterns
3. **Reinforcement learning**: Train with hit rate as reward signal
4. **Joint training**: Train predictor end-to-end with the MoE model
5. **Cross-model transfer**: Train on one MoE model, apply to others

## 11. Summary

This plan provides a complete roadmap for implementing an attention-based expert predictor that:

1. **Leverages rich supervision**: Trains on full 60-dimensional gating distributions from WikiText-103
2. **Uses proper evaluation**: Separate train/test splits with top-4 overlap accuracy metric
3. **Provides practical speedup**: Expected 1.2-1.4x inference speedup with good generalization
4. **Is production-ready**: Complete implementation with testing, benchmarking, and documentation

**Key Innovations**:
- Full gating score supervision (not just top-4 indices)
- KL divergence loss for distribution matching
- WikiText-103 for standardized train/test evaluation
- Integration with existing Fiddler prefetching system

**Implementation Timeline**: 5 weeks total
- Week 1: Data collection (50k samples from WikiText-103 train)
- Week 2: Model training (10 epochs, KL divergence loss)
- Week 3: Evaluation (WikiText-103 test set)
- Week 4: Integration with Fiddler system
- Week 5: End-to-end benchmarking and optimization

**Success Criteria**:
- Top-4 overlap > 60% on WikiText test set
- Inference speedup > 1.2x vs. baseline
- Good generalization to diverse prompts

---

**End of Plan**

This plan is ready for implementation by another agent. Each section includes detailed specifications, complete code templates, and verification steps to ensure successful deployment.
