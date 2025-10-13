# Attention-Based Expert Predictor - Shared Context

## Overview

This document contains shared technical context for implementing an attention-based expert predictor for the Qwen1.5-MoE-A2.7B model. All phase implementation plans reference this document.

## System Context

### Current System
- **Model**: Qwen1.5-MoE-A2.7B with 24 MoE layers (layers 0-23)
- **Experts**: 60 experts per layer, top-k=4 routing
- **Current prefetch**: Token position-based patterns (`expert_usage_patterns_qwen.json`)
- **Current limitation**: Gating-based prediction (layer X → X+1) has only 7.5% overlap
- **GPU-resident layers**: First 2 MoE layers (0-1) kept on GPU permanently
- **Prediction target**: MoE layers 2-23 (22 layers to predict)

### Motivation

**Why First Layer Attention?**
1. **Rich semantic features**: Attention outputs capture token relationships and context
2. **Early availability**: Layer 0 attention completes before any MoE layers
3. **Sufficient lookahead**: Can predict 22 MoE layers (2-23) from a single forward pass
4. **Proven approach**: Attention representations capture high-level semantics

**Key Innovation**: Use first-layer attention outputs as features to predict expert usage across all downstream layers, improving upon pattern-based (fixed token positions) and gating-based approaches (7.5% overlap).

## Architecture Specification

### Model Structure

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

### Dimensions

**Input**:
- Qwen1.5-MoE-A2.7B: `hidden_dim = 2048`
- Input shape: `[batch_size, sequence_length, 2048]`

**Output**:
- 22 MoE layers to predict (layers 2-23; 0-1 are GPU-resident)
- 60 experts per layer
- Output shape: `[batch_size, sequence_length, 22, 60]`
- Each value is a logit score for that expert at that layer

**Model Capacity**:
- Parameters: ~6.9M (2048×2048 + 2048×1320 ≈ 6.9M weights + biases)
- Memory: ~28MB in fp32, ~14MB in fp16
- Inference time: <1ms per forward pass on GPU

### PyTorch Implementation

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

## Data Specifications

### Dataset: WikiText-103

**Source**: `wikitext/wikitext-103-raw-v1` from Hugging Face datasets

**Splits**:
- **Train split**: For collecting training data (~100M tokens)
- **Test split**: Reserved for final evaluation (NEVER use for training)

### Data Format

**Ground Truth: Full Gating Scores (Not Just Top-4)**

**Why full 60-dimensional gating scores?**
- **Richer learning signal**: 60 continuous values vs 4 discrete indices
- **Captures relative importance**: Not just which experts, but how confident the gating is
- **Better gradients**: Continuous targets provide smoother optimization
- **Learns the distribution**: Model learns the complete expert selection distribution

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
  - split: "train" or "test"
```

**Dataset Size Estimation**:
- 50k samples × 2048 float32 = ~400MB (attention outputs)
- 50k samples × 22 × 60 float32 = ~260MB (gating scores - full distribution)
- **Total**: ~660MB per 50k samples

## Training Specifications

### Loss Function

**Recommended: KL Divergence** (equivalent to cross-entropy)

```python
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
```

**Alternative: MSE Loss** (simpler but less theoretically motivated)

```python
def mse_loss(predicted_logits, true_distribution):
    pred_distribution = torch.softmax(predicted_logits, dim=-1)
    loss = torch.nn.functional.mse_loss(pred_distribution, true_distribution)
    return loss
```

### Hyperparameters

**Recommended Configuration**:
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

## Evaluation Metrics

### Top-4 Overlap Accuracy

**Definition**: Fraction of true top-4 experts that appear in predicted top-4.

```python
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
    _, predicted_top_k = torch.topk(predicted_logits, k, dim=-1)

    # Get top-k true experts
    _, true_top_k = torch.topk(true_distribution, k, dim=-1)

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
```

## Success Criteria

### Prediction Accuracy (WikiText Test Set)

**Minimum Viable Performance**:
- Top-4 overlap > 60% (vs. 18.75% for gating-based layer X→X+1)

**Target Performance**:
- Top-4 overlap > 60%
- Comparable to or better than pattern-based on diverse prompts

**Stretch Goals**:
- Top-4 overlap > 90%
- Strong predictor ready for production

### Inference Performance

**Expected Results**:
- **Prefill Hit Rate**: 50-70%
- **Decode Hit Rate**: 70-90%
- **Overall Hit Rate**: 60-80%
- **Inference Speedup**: 1.2-1.4x vs. baseline

## Technical Details

### Hook Registration for Data Collection

**Capturing First Layer Attention Output**:
```python
def attention_hook(module, input, output):
    # output is a tuple: (hidden_states, attention_weights, ...)
    hidden_states = output[0]  # [batch, seq_len, hidden_dim]
    self.current_attention_output = hidden_states.detach().cpu()

# Register on layer 0's self_attn
model.model.layers[0].self_attn.register_forward_hook(attention_hook)
```

**Capturing Full Gating Scores**:
```python
def moe_hook(layer_idx):
    def hook(module, input, output):
        hidden_states = input[0]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Get router logits for ALL 60 experts
        router_logits = module.gate(hidden_states_flat)

        # Apply softmax to get gating scores - this is our ground truth
        gating_scores = F.softmax(router_logits, dim=1, dtype=torch.float)

        if not hasattr(self, 'current_gating_scores'):
            self.current_gating_scores = {}

        self.current_gating_scores[layer_idx] = gating_scores.detach().cpu()

        return output
    return hook

# Register on all MoE layers
for layer_idx in moe_layers:
    model.model.layers[layer_idx].mlp.register_forward_hook(moe_hook(layer_idx))
```

## Directory Structure

After full implementation, the project structure will be:

```
fiddler/
├── PREDICTOR_SHARED_CONTEXT.md              # This document
├── PREDICTOR_PHASE1_DATA_COLLECTION.md      # Phase 1 implementation plan
├── PREDICTOR_PHASE2_MODEL_TRAINING.md       # Phase 2 implementation plan
├── PREDICTOR_PHASE3_EVALUATION.md           # Phase 3 implementation plan
├── PREDICTOR_PHASE4_FIDDLER_INTEGRATION.md  # Phase 4 implementation plan
├── PREDICTOR_PHASE5_BENCHMARKING.md         # Phase 5 implementation plan
├── collect_training_data.py                 # Data collection script
├── train_predictor.py                       # Training script
├── evaluate_predictor.py                    # Evaluation script
├── predictor_training_data/                 # Training data (HDF5 files)
│   ├── training_data_0000.h5
│   ├── training_data_0001.h5
│   └── ...
├── predictor_checkpoints/                   # Model checkpoints
│   ├── config.json
│   ├── best_model.pt
│   └── checkpoint_epoch_*.pt
├── predictor_evaluation_results.json        # Evaluation results
├── src/fiddler/
│   ├── qwen_with_prefetch.py               # Existing: pattern-based
│   └── qwen_with_learned_prefetch.py       # New: learned predictor
├── test_learned_prefetch.py                # Testing script
└── benchmark_prediction_methods.py         # Comparison benchmark
```

## References to Existing System

### Key Files
- `src/fiddler/qwen.py`: Baseline implementation
- `src/fiddler/qwen_with_prefetch.py`: Pattern-based prefetch (current best)
- `expert_usage_patterns_qwen.json`: Token position patterns
- `thoughts/20250929/guide.md`: Project documentation

### Performance Baselines
- **Baseline (no prefetch)**: 1.0x
- **Pattern-based prefetch**: 1.28x speedup, 100% decode hit rate (on training prompts)
- **Fiddler CPU-only**: 2.17x speedup (BS=1)
- **Fiddler+Prefetch**: 1.47x speedup (BS=1)

## Implementation Phases

1. **Phase 1: Data Collection**
   - Collect 50k samples from WikiText-103 train split
   - Store attention outputs and full gating scores in HDF5

2. **Phase 2: Model Training**
   - Train predictor with KL divergence loss
   - Achieve >40% top-4 accuracy on validation set

3. **Phase 3: Evaluation**
   - Evaluate on WikiText-103 test split
   - Document prediction accuracy metrics

4. **Phase 4: Fiddler Integration** 
   - Create `qwen_with_learned_prefetch.py`
   - Test correctness and basic functionality

5. **Phase 5: End-to-End Benchmarking**
   - Compare baseline, pattern-based, and learned approaches
   - Measure actual inference speedup and hit rates

---

**Note**: Each phase has its own detailed implementation document with complete code, step-by-step instructions, and verification checklists.
