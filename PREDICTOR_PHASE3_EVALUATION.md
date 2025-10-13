# Phase 3: Evaluation

**Prerequisites**: Completed Phase 2 (trained model available)
**Dependencies**: See [PREDICTOR_SHARED_CONTEXT.md](PREDICTOR_SHARED_CONTEXT.md) for evaluation metrics and success criteria

## Objective

Evaluate the trained predictor on WikiText-103 **test split** (never seen during training) to measure its prediction accuracy and compare against baseline approaches.

## Deliverables

- `predictor_evaluation_results.json`: Comprehensive evaluation results
- Metrics showing:
  - Top-4 overlap accuracy (target: >60%)
  - Precision@4 and NDCG@4
- Comparison with gating-based baseline (18.75% overlap)
- Analysis document summarizing findings

## Implementation

### Script: `evaluate_predictor.py`

Create the following script in the project root:

```python
#!/usr/bin/env python3
"""
evaluate_predictor.py - Evaluate predictor on WikiText-103 TEST split
IMPORTANT: Uses TEST split only - never used during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import argparse

# Import predictor model from training script
import sys
sys.path.append('.')
from train_predictor import AttentionBasedExpertPredictor

class PredictorEvaluator:
    def __init__(self, predictor_path, model_name="Qwen/Qwen1.5-MoE-A2.7B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load predictor
        print(f"Loading predictor from: {predictor_path}")
        checkpoint = torch.load(predictor_path, map_location=self.device)
        config = checkpoint['config']

        self.predictor = AttentionBasedExpertPredictor(
            hidden_dim=config['hidden_dim'],
            n_moe_layers=config['n_moe_layers'],
            n_experts=config['n_experts'],
            dropout=0.0  # No dropout during evaluation
        ).to(self.device)

        self.predictor.load_state_dict(checkpoint['model_state_dict'])
        self.predictor.eval()

        print(f"✅ Loaded predictor from epoch {checkpoint['epoch']}")
        print(f"   Training Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"   Training Val Top-4 Acc: {checkpoint['val_acc']:.4f}")

        # Load Qwen model
        print(f"\nLoading model: {model_name}")
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
        print(f"✅ Found {len(self.moe_layers)} MoE layers")
        print(f"   Evaluating predictions for layers: {self.moe_layers_to_predict}")

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
        print("\n" + "="*80)
        print("EVALUATING ON WIKITEXT-103 TEST SPLIT")
        print("="*80)
        print("Loading WikiText-103 TEST split...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

        samples_evaluated = 0
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating", total=num_samples)):
            if samples_evaluated >= num_samples:
                break

            text = example['text']
            if len(text.strip()) < 50:
                continue

            self.evaluate_sample(text)
            samples_evaluated += 1

        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Samples evaluated: {samples_evaluated}")
        print(f"Total token-layer pairs: {len(self.metrics['top4_overlap'])}")
        print(f"\n{'Metric':<25} {'Score':<10} {'Status'}")
        print("-"*80)

        top4_overlap = np.mean(self.metrics['top4_overlap'])
        jaccard = np.mean(self.metrics['jaccard'])
        precision = np.mean(self.metrics['precision_at_4'])
        ndcg = np.mean(self.metrics['ndcg_at_4'])

        # Check against success criteria
        top4_status = "✅ PASS" if top4_overlap > 0.60 else "⚠️  TARGET: >60%" if top4_overlap > 0.40 else "❌ FAIL"
        jaccard_status = "✅ PASS" if jaccard > 0.50 else "⚠️  TARGET: >50%" if jaccard > 0.30 else "❌ FAIL"

        print(f"{'Top-4 Overlap Accuracy':<25} {top4_overlap:<10.4f} {top4_status}")
        print(f"{'Jaccard Similarity':<25} {jaccard:<10.4f} {jaccard_status}")
        print(f"{'Precision@4':<25} {precision:<10.4f}")
        print(f"{'NDCG@4':<25} {ndcg:<10.4f}")

        print("\n" + "="*80)
        print("COMPARISON WITH BASELINES")
        print("="*80)
        print(f"{'Method':<30} {'Top-4 Overlap':<15} {'Notes'}")
        print("-"*80)
        print(f"{'Gating-based (layer X→X+1)':<30} {'18.75%':<15} {'Current system'}")
        print(f"{'Random guessing':<30} {'6.67%':<15} {'4/60 experts'}")
        print(f"{'Learned predictor (ours)':<30} {f'{top4_overlap*100:.2f}%':<15} {'This evaluation'}")

        improvement = (top4_overlap - 0.1875) / 0.1875 * 100
        print(f"\n{'Improvement over gating-based:':<30} {improvement:+.1f}%")

        # Save results
        results = {
            'num_samples': samples_evaluated,
            'num_token_layer_pairs': len(self.metrics['top4_overlap']),
            'metrics': {
                'top4_overlap_accuracy': float(top4_overlap),
                'jaccard_similarity': float(jaccard),
                'precision_at_4': float(precision),
                'ndcg_at_4': float(ndcg)
            },
            'baselines': {
                'gating_based': 0.1875,
                'random': 0.0667
            },
            'success_criteria': {
                'minimum_viable': {
                    'top4_overlap': 0.40,
                    'jaccard': 0.30,
                    'met': top4_overlap > 0.40 and jaccard > 0.30
                },
                'target': {
                    'top4_overlap': 0.60,
                    'jaccard': 0.50,
                    'met': top4_overlap > 0.60 and jaccard > 0.50
                },
                'stretch': {
                    'top4_overlap': 0.75,
                    'jaccard': 0.65,
                    'met': top4_overlap > 0.75 and jaccard > 0.65
                }
            }
        }

        output_file = 'predictor_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {output_file}")
        print("="*80)

        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate attention-based expert predictor')
    parser.add_argument('--predictor-path', type=str, default='predictor_checkpoints/best_model.pt',
                       help='Path to trained predictor checkpoint')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to evaluate from test set')
    args = parser.parse_args()

    evaluator = PredictorEvaluator(predictor_path=args.predictor_path)
    results = evaluator.evaluate_dataset(num_samples=args.num_samples)

if __name__ == "__main__":
    main()
```

## Step-by-Step Instructions

### 1. Verify Prerequisites

Ensure Phase 2 is complete:

```bash
ls predictor_checkpoints/best_model.pt
```

Should exist and contain the trained model.

### 2. Run Evaluation

```bash
python evaluate_predictor.py
```

**Default settings**:
- Uses `predictor_checkpoints/best_model.pt`
- Evaluates 1000 samples from WikiText-103 test split
- Outputs results to `predictor_evaluation_results.json`

**Custom settings**:
```bash
python evaluate_predictor.py \
  --predictor-path predictor_checkpoints/best_model.pt \
  --num-samples 2000
```

### 3. Expected Output

The script will print:
1. Model loading confirmation
2. Progress bar for evaluation
3. Comprehensive results table
4. Comparison with baselines
5. Success criteria assessment

**Expected evaluation time**:
- 1000 samples: ~20-30 minutes
- 2000 samples: ~40-60 minutes

### 4. Interpret Results

**Success Tiers**:

1. **Minimum Viable** (40% top-4):
   - Predictor is better than gating-based (18.75%)
   - Justifies using learned approach
   - Worth proceeding to integration

2. **Target** (60% top-4):
   - Strong predictor performance
   - Should provide good inference speedup
   - Ready for production testing

3. **Stretch** (75% top-4):
   - Excellent predictor performance
   - Likely 1.3-1.5x inference speedup
   - Production-ready

### 5. Analyze Results File

```python
import json

with open('predictor_evaluation_results.json', 'r') as f:
    results = json.load(f)

print("Evaluation Summary:")
print(f"  Samples: {results['num_samples']}")
print(f"  Token-layer pairs: {results['num_token_layer_pairs']}")
print(f"\nMetrics:")
for metric, value in results['metrics'].items():
    print(f"  {metric}: {value:.4f}")

print(f"\nSuccess Criteria:")
for tier, criteria in results['success_criteria'].items():
    status = "✅ MET" if criteria['met'] else "❌ NOT MET"
    print(f"  {tier}: {status}")
```

## Verification Checklist

Before proceeding to Phase 4, verify:

- [ ] Evaluation completed without errors
- [ ] Evaluated on WikiText-103 **test split** (NOT train split)
- [ ] Top-4 overlap accuracy > 40% (minimum viable)
- [ ] Jaccard similarity > 0.30 (minimum viable)
- [ ] Results show improvement over gating-based baseline (18.75%)
- [ ] `predictor_evaluation_results.json` created with all metrics
- [ ] Success criteria assessment included in results
- [ ] No data leakage (test set never seen during training)

## Troubleshooting

### Issue: "No such file: predictor_checkpoints/best_model.pt"

**Solution**: Complete Phase 2 first:
```bash
python train_predictor.py
```

### Issue: Low accuracy (<30%)

**Diagnosis**: Model did not train properly.

**Solutions**:
1. Check Phase 2 training results - was validation accuracy >40%?
2. Try training with different hyperparameters:
   - More epochs (20 instead of 10)
   - Larger model (4096 hidden dim)
   - More training data (100k samples)
3. Verify data quality from Phase 1

### Issue: Evaluation taking too long

**Solution**: Reduce number of samples:
```bash
python evaluate_predictor.py --num-samples 500
```

Note: Results may be less stable with fewer samples.

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce max_length in evaluate_sample:
   ```python
   def evaluate_sample(self, text, max_length=256):  # Reduced from 512
   ```

2. Process samples in smaller batches

3. Use smaller GPU or CPU:
   ```bash
   CUDA_VISIBLE_DEVICES="" python evaluate_predictor.py  # Force CPU
   ```

## Analysis Guidelines

### What Makes a Good Result?

**Top-4 Overlap Accuracy**:
- **>60%**: Excellent - predictor captures expert selection patterns well
- **40-60%**: Good - useful for inference acceleration
- **25-40%**: Marginal - may still help but limited benefit
- **<25%**: Poor - not better than gating-based (18.75%)

### Comparison Context

| Approach | Top-4 Overlap | Generalization |
|----------|---------------|----------------|
| Random | 6.67% | N/A |
| Gating layer X→X+1 | 18.75% | Poor (7.5% one layer ahead) |
| Pattern-based | 100% (training prompts) | Poor (fixed patterns) |
| **Learned (target)** | **60%+** | **Good (diverse prompts)** |

### When to Proceed to Phase 4

**Proceed if**:
- Top-4 overlap > 40%
- Improvement over baseline demonstrated

**Consider retraining if**:
- Top-4 overlap < 30%
- No improvement over baseline
- High variance in results

**Alternative approaches if failing**:
- Collect more training data (100k samples)
- Use deeper architecture (3-4 layers)
- Try different loss functions
- Use attention from multiple layers (0-2)

## Next Steps

Once evaluation shows promising results (>40% top-4 overlap):
1. Proceed to **Phase 4: Fiddler Integration** (PREDICTOR_PHASE4_FIDDLER_INTEGRATION.md)
2. The evaluated model will be integrated with the Fiddler prefetching system
3. Keep `predictor_evaluation_results.json` for documentation

If results are not satisfactory (<40% top-4 overlap):
1. Analyze failure modes
2. Return to Phase 2 with adjusted hyperparameters
3. Consider collecting more data (return to Phase 1)

## References

- See [PREDICTOR_SHARED_CONTEXT.md](PREDICTOR_SHARED_CONTEXT.md) for:
  - Evaluation metrics definitions
  - Success criteria thresholds
  - Baseline comparisons
  - Expected results

---

**Estimated Completion Time**: 1 week (including evaluation runs, analysis, and documentation)
