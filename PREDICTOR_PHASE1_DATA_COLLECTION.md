# Phase 1: Data Collection

**Prerequisites**: Access to GPU, internet connection for downloading WikiText-103
**Dependencies**: See [PREDICTOR_SHARED_CONTEXT.md](PREDICTOR_SHARED_CONTEXT.md) for architecture and data format specifications

## Objective

Collect training data from WikiText-103 train split by running Qwen1.5-MoE-A2.7B and capturing:
1. First layer attention outputs (input features)
2. Full gating scores for all 60 experts across MoE layers 2-23 (ground truth labels)

## Deliverables

- `predictor_training_data/` directory containing HDF5 files
- ~50k samples (tokens) from WikiText-103 train split
- Total size: ~660MB
- Each HDF5 file contains:
  - `/attention_outputs`: [n_samples, 2048] float32
  - `/gating_scores`: [n_samples, 22, 60] float32

## Implementation

### Script: `collect_training_data.py`

Create the following script in the project root:

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

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install torch transformers datasets h5py numpy tqdm
```

### 2. Test on Small Sample (Optional but Recommended)

Before running the full collection, test on a small sample:

```python
# Modify main() to collect only 100 samples
collector.collect_from_dataset(
    num_samples=100,  # Small test
    samples_per_file=100
)
```

Run:
```bash
python collect_training_data.py
```

Expected output:
- Creates `predictor_training_data/training_data_0000.h5`
- Should complete in ~5-10 minutes
- File size: ~13MB

### 3. Verify Test Data

```python
import h5py

with h5py.File('predictor_training_data/training_data_0000.h5', 'r') as f:
    print("Datasets:", list(f.keys()))
    print("Attention outputs shape:", f['attention_outputs'].shape)
    print("Gating scores shape:", f['gating_scores'].shape)
    print("\nAttributes:")
    for key, value in f.attrs.items():
        print(f"  {key}: {value}")
```

Expected output:
```
Datasets: ['attention_outputs', 'gating_scores']
Attention outputs shape: (n_samples, 2048)
Gating scores shape: (n_samples, 22, 60)

Attributes:
  n_samples: <number>
  hidden_dim: 2048
  n_moe_layers_to_predict: 22
  moe_layers_to_predict: [2, 3, 4, ..., 23]
  n_experts: 60
  dataset: wikitext-103-raw-v1
  split: train
```

### 4. Run Full Data Collection

Once test is successful, modify main() to collect full dataset:

```python
collector.collect_from_dataset(
    num_samples=50000,  # Full collection
    samples_per_file=1000
)
```

Run:
```bash
python collect_training_data.py
```

Expected:
- **Time**: 2-4 hours (depends on GPU)
- **Output**: ~50 HDF5 files in `predictor_training_data/`
- **Total size**: ~660MB
- **Progress**: tqdm progress bar shows collection progress

### 5. Verify Full Dataset

```bash
ls -lh predictor_training_data/
```

Expected:
- Multiple files: `training_data_0000.h5`, `training_data_0001.h5`, ...
- Each file: ~13MB
- Total: ~660MB

Count total samples:
```python
import h5py
from pathlib import Path

total_samples = 0
for file_path in sorted(Path('predictor_training_data').glob('training_data_*.h5')):
    with h5py.File(file_path, 'r') as f:
        total_samples += f.attrs['n_samples']

print(f"Total samples collected: {total_samples}")
```

## Verification Checklist

Before proceeding to Phase 2, verify:

- [ ] `predictor_training_data/` directory exists
- [ ] 50k+ samples collected from WikiText-103 train split
- [ ] HDF5 files have correct format (verified with test script)
- [ ] Attention outputs shape: [n_samples, 2048]
- [ ] Gating scores shape: [n_samples, 22, 60] (full distribution)
- [ ] All 22 MoE layers (2-23) represented in metadata
- [ ] Test split NOT used for training (only train split accessed)
- [ ] Total dataset size is approximately 660MB
- [ ] No errors or warnings during collection
- [ ] All HDF5 files can be opened and read without errors

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Reduce `max_length` in `collect_from_text()`:
```python
def collect_from_text(self, text, max_length=256):  # Reduced from 512
```

### Issue: "No attention output captured"

**Solution**: Verify hook is attached to correct layer. Check model architecture:
```python
print(model.model.layers[0].self_attn)
```

### Issue: Slow collection speed

**Solutions**:
1. Use larger GPU (V100/A100)
2. Reduce `max_length` to 256
3. Collect fewer samples initially (e.g., 25k)

### Issue: HDF5 file corruption

**Solution**: Files are saved incrementally. Delete corrupted file and re-run:
```bash
rm predictor_training_data/training_data_XXXX.h5
# Collection will skip existing files
```

## Next Steps

Once data collection is complete and verified:
1. Proceed to **Phase 2: Model Training** (PREDICTOR_PHASE2_MODEL_TRAINING.md)
2. The collected data will be used to train the predictor model
3. Do NOT collect data from the test split - it is reserved for Phase 3 evaluation

## References

- See [PREDICTOR_SHARED_CONTEXT.md](PREDICTOR_SHARED_CONTEXT.md) for:
  - Architecture specifications
  - Data format details
  - Hook implementation details
  - Success criteria

---

**Estimated Completion Time**: 1 week (including testing, debugging, and full collection)
