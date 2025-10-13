# Phase 4: Fiddler Integration

**Prerequisites**: Completed Phase 2 (trained model available)
**Dependencies**: See [PREDICTOR_SHARED_CONTEXT.md](PREDICTOR_SHARED_CONTEXT.md) for architecture details

## Objective

Integrate the trained attention-based predictor with the Fiddler prefetching system to create `qwen_with_learned_prefetch.py` that uses learned predictions instead of token position patterns.

## Deliverables

- `src/fiddler/qwen_with_learned_prefetch.py`: New implementation with learned predictor
- `test_learned_prefetch.py`: Testing script for correctness validation
- Verified correctness: output matches baseline implementation
- Working integration ready for Phase 5 benchmarking

## Architecture Overview

### Integration Strategy

The learned predictor will replace the pattern-based prediction in `qwen_with_prefetch.py`:

**Pattern-based (current)**:
```python
# Lookup experts based on token position
predicted_experts = self.expert_patterns[token_position][layer_id]
```

**Learned-based (new)**:
```python
# Predict experts based on first layer attention output
attention_output = captured_from_layer_0_attention
predicted_logits = self.predictor(attention_output)  # [batch, seq_len, 22, 60]
top_k_experts = torch.topk(predicted_logits[..., layer_idx, :], k=num_experts_to_prefetch)
```

### Key Components

1. **Predictor Loading**: Load trained model at initialization
2. **Attention Capture**: Hook to capture layer 0 attention output
3. **Expert Prediction**: Run predictor to get expert scores
4. **Prefetch Integration**: Use predictions for buffer loading
5. **Correctness**: Ensure output matches baseline

## Implementation

### File: `src/fiddler/qwen_with_learned_prefetch.py`

This file is based on `qwen_with_prefetch.py` but uses learned predictions. Key differences:

```python
#!/usr/bin/env python3
"""
qwen_with_learned_prefetch.py - Qwen with learned attention-based expert prefetching
Uses trained predictor instead of token position patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import os

# Import predictor model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from train_predictor import AttentionBasedExpertPredictor


class FiddlerQwenWithLearnedPrefetch:
    """
    Qwen1.5-MoE implementation with learned expert prefetching.
    Uses attention-based predictor instead of token position patterns.
    """

    def __init__(
        self,
        args,
        predictor_path="predictor_checkpoints/best_model.pt",
        num_experts_to_prefetch=8,
        enable_cpu_offload=False,
        latency_cpu=0.1,
        latency_gpu=10.0
    ):
        """
        Initialize model with learned prefetching.

        Args:
            args: Model arguments (model path, etc.)
            predictor_path: Path to trained predictor checkpoint
            num_experts_to_prefetch: Number of experts to prefetch per layer
            enable_cpu_offload: Whether to use Fiddler CPU offloading
            latency_cpu: CPU expert latency for Fiddler cost model
            latency_gpu: GPU expert latency for Fiddler cost model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Prefetch configuration
        self.num_experts_to_prefetch = num_experts_to_prefetch
        self.enable_cpu_offload = enable_cpu_offload
        self.latency_cpu = latency_cpu
        self.latency_gpu = latency_gpu

        # Load predictor
        print(f"Loading predictor from: {predictor_path}")
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

        print(f"✅ Loaded predictor (val_acc: {checkpoint['val_acc']:.4f})")

        # Load Qwen model
        print(f"Loading model: {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self.dtype
        )

        # Find MoE layers
        self.moe_layers = []
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts'):
                self.moe_layers.append(idx)

        self.moe_layers_to_predict = self.moe_layers[2:]  # Skip first 2 (GPU-resident)
        print(f"Found {len(self.moe_layers)} MoE layers")
        print(f"Will prefetch for {len(self.moe_layers_to_predict)} layers: {self.moe_layers_to_predict}")

        # Register attention capture hook
        self._register_attention_hook()

        # Initialize prefetch buffers (similar to pattern-based version)
        if self.num_experts_to_prefetch > 0:
            self._initialize_prefetch_buffers()

        # Initialize Fiddler CPU offload if enabled
        if self.enable_cpu_offload:
            self._initialize_fiddler()

        # Timing tracking
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.prefill_hits = 0
        self.prefill_total = 0
        self.decode_hits = 0
        self.decode_total = 0

    def _register_attention_hook(self):
        """Register hook to capture first layer attention output."""
        def attention_hook(module, input, output):
            # Capture attention output for predictor
            self.current_attention_output = output[0].detach()  # [batch, seq_len, hidden_dim]

        # Register on layer 0 self-attention
        self.model.model.layers[0].self_attn.register_forward_hook(attention_hook)

    def _initialize_prefetch_buffers(self):
        """Initialize prefetch buffers (similar to pattern-based)."""
        # Dual buffer system for even/odd layers
        # Details depend on your existing qwen_with_prefetch.py implementation
        # This is a placeholder for the buffer initialization logic
        pass

    def _initialize_fiddler(self):
        """Initialize Fiddler CPU offloading system."""
        # Fiddler initialization logic
        # Move experts to CPU based on cost model
        # Details depend on your existing Fiddler implementation
        pass

    def predict_experts_for_layer(self, layer_idx, token_idx=None):
        """
        Predict top-k experts for a given layer using learned predictor.

        Args:
            layer_idx: MoE layer index (2-23)
            token_idx: Token index (for decode phase). If None, predict for all tokens.

        Returns:
            List of expert indices to prefetch
        """
        if not hasattr(self, 'current_attention_output'):
            # Fallback: no prediction available
            return list(range(self.num_experts_to_prefetch))

        attention_output = self.current_attention_output  # [batch, seq_len, hidden_dim]

        # Run predictor
        with torch.no_grad():
            predicted_logits = self.predictor(attention_output)  # [batch, seq_len, 22, 60]

        # Map layer_idx to predictor output index
        # predictor predicts for layers 2-23, so layer 2 → output_idx 0
        output_idx = layer_idx - 2  # Adjust for layers 2-23

        if token_idx is not None:
            # Decode phase: predict for specific token
            logits = predicted_logits[0, token_idx, output_idx, :]  # [60]
        else:
            # Prefill phase: predict for all tokens, aggregate
            logits = predicted_logits[0, :, output_idx, :].mean(dim=0)  # [60]

        # Get top-k experts
        _, top_k_indices = torch.topk(logits, k=self.num_experts_to_prefetch)

        return top_k_indices.cpu().tolist()

    def generate(self, text, output_token=20, input_token=None):
        """
        Generate text with learned prefetching.

        Returns:
            (prefill_time, decode_time, prefill_hit_rate, decode_hit_rate)
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # Reset timing
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.prefill_hits = 0
        self.prefill_total = 0
        self.decode_hits = 0
        self.decode_total = 0

        # Generate
        start_time = time.time()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=output_token,
                do_sample=False,
                num_beams=1,
                eos_token_id=None  # Force exact token count
            )

        total_time = time.time() - start_time

        # Calculate hit rates
        prefill_hit_rate = self.prefill_hits / self.prefill_total if self.prefill_total > 0 else 0.0
        decode_hit_rate = self.decode_hits / self.decode_total if self.decode_total > 0 else 0.0

        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return self.prefill_time, self.decode_time, prefill_hit_rate, decode_hit_rate

    def tokenize(self, text):
        """Tokenize text."""
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids
        position_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0)
        return input_ids, position_ids


# Note: This is a simplified template. Full implementation should:
# 1. Integrate with existing qwen_with_prefetch.py buffer management
# 2. Track hit rates properly
# 3. Handle prefill vs decode phases
# 4. Integrate with Fiddler CPU offload if enabled
# 5. Match the interface of existing implementations for compatibility
```

### File: `test_learned_prefetch.py`

Testing script to verify correctness:

```python
#!/usr/bin/env python3
"""
test_learned_prefetch.py - Test learned prefetch correctness
Verifies that learned predictor produces same outputs as baseline
"""

import torch
import argparse
import sys
sys.path.append('src/fiddler')

from qwen import FiddlerQwen  # Baseline
from qwen_with_learned_prefetch import FiddlerQwenWithLearnedPrefetch


class Args:
    """Simple args container"""
    def __init__(self):
        self.model_path = "Qwen/Qwen1.5-MoE-A2.7B"


def test_correctness():
    """Test that learned prefetch produces same output as baseline."""
    args = Args()

    print("="*80)
    print("CORRECTNESS TEST: Learned Prefetch vs Baseline")
    print("="*80)

    # Test input
    test_input = "The quick brown fox jumps over the lazy dog."
    output_tokens = 10

    # Run baseline
    print("\n1. Running baseline...")
    baseline_model = FiddlerQwen(args)
    baseline_model.tokenize(test_input)  # Tokenize first
    inputs = baseline_model.tokenizer(test_input, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")

    with torch.no_grad():
        baseline_output = baseline_model.model.generate(
            input_ids,
            max_new_tokens=output_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=None
        )

    baseline_text = baseline_model.tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    print(f"Baseline output: {baseline_text}")

    # Run learned prefetch
    print("\n2. Running learned prefetch...")
    learned_model = FiddlerQwenWithLearnedPrefetch(
        args,
        predictor_path="predictor_checkpoints/best_model.pt",
        num_experts_to_prefetch=8,
        enable_cpu_offload=False
    )

    with torch.no_grad():
        learned_output = learned_model.model.generate(
            input_ids,
            max_new_tokens=output_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=None
        )

    learned_text = learned_model.tokenizer.decode(learned_output[0], skip_special_tokens=True)
    print(f"Learned output: {learned_text}")

    # Compare
    print("\n3. Comparing outputs...")
    if torch.equal(baseline_output, learned_output):
        print("✅ PASS: Outputs match exactly")
        return True
    else:
        print("❌ FAIL: Outputs differ")
        print(f"Baseline tokens: {baseline_output[0].tolist()}")
        print(f"Learned tokens: {learned_output[0].tolist()}")
        return False


def test_hit_rate():
    """Test that prefetching achieves reasonable hit rates."""
    args = Args()

    print("\n" + "="*80)
    print("HIT RATE TEST")
    print("="*80)

    test_input = "The quick brown fox jumps over the lazy dog."
    output_tokens = 20

    model = FiddlerQwenWithLearnedPrefetch(
        args,
        predictor_path="predictor_checkpoints/best_model.pt",
        num_experts_to_prefetch=8,
        enable_cpu_offload=False
    )

    prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
        test_input,
        output_token=output_tokens
    )

    print(f"\nPrefill hit rate: {prefill_hit_rate*100:.1f}%")
    print(f"Decode hit rate: {decode_hit_rate*100:.1f}%")

    # Reasonable hit rates based on evaluation
    min_decode_hit_rate = 0.40  # Should be >40% based on top-4 accuracy

    if decode_hit_rate >= min_decode_hit_rate:
        print(f"✅ PASS: Decode hit rate >= {min_decode_hit_rate*100}%")
        return True
    else:
        print(f"⚠️  WARNING: Decode hit rate < {min_decode_hit_rate*100}%")
        print("   This may indicate integration issues")
        return False


def main():
    print("Testing learned prefetch integration...")

    # Test 1: Correctness
    correctness_pass = test_correctness()

    # Test 2: Hit rate
    hit_rate_pass = test_hit_rate()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Correctness: {'✅ PASS' if correctness_pass else '❌ FAIL'}")
    print(f"Hit Rate: {'✅ PASS' if hit_rate_pass else '⚠️  WARNING'}")

    if correctness_pass and hit_rate_pass:
        print("\n✅ All tests passed! Ready for Phase 5 benchmarking.")
        return 0
    elif correctness_pass:
        print("\n⚠️  Correctness passed but hit rate low. Proceed with caution.")
        return 0
    else:
        print("\n❌ Tests failed. Debug integration before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
```

## Step-by-Step Instructions

### 1. Study Existing Implementation

Before implementing, review the existing pattern-based implementation:

```bash
less src/fiddler/qwen_with_prefetch.py
```

Key components to understand:
- Buffer management (dual buffer A/B)
- Hit rate tracking
- Prefill vs decode phase handling
- Integration with Fiddler CPU offload

### 2. Implement `qwen_with_learned_prefetch.py`

Based on the template above and `qwen_with_prefetch.py`:

1. Copy buffer management logic from `qwen_with_prefetch.py`
2. Replace pattern lookup with learned prediction
3. Ensure attention capture hook works correctly
4. Maintain the same interface (generate, tokenize, etc.)

### 3. Implement Test Script

Create `test_learned_prefetch.py` with correctness and hit rate tests.

### 4. Run Tests

```bash
python test_learned_prefetch.py
```

Expected:
- Correctness test: PASS (outputs match)
- Hit rate test: >40% decode hit rate

### 5. Debug if Needed

If tests fail, check:
- Attention hook captures correct output
- Predictor outputs correct shape [batch, seq_len, 22, 60]
- Layer index mapping (layer 2 → output_idx 0)
- Top-k extraction works correctly

## Verification Checklist

Before proceeding to Phase 5, verify:

- [ ] `src/fiddler/qwen_with_learned_prefetch.py` created
- [ ] `test_learned_prefetch.py` created and runs
- [ ] Correctness test passes (output matches baseline)
- [ ] Decode hit rate > 40% achieved
- [ ] Attention hook captures layer 0 output correctly
- [ ] Predictor inference runs without errors
- [ ] Buffer management works (no memory leaks)
- [ ] Interface matches existing implementations
- [ ] Can run with and without Fiddler CPU offload
- [ ] Code is documented and clean

## Troubleshooting

### Issue: Attention hook not capturing output

**Solution**: Verify hook attachment:
```python
# Check hook is registered
print(self.model.model.layers[0].self_attn._forward_hooks)
```

Ensure output[0] is the hidden states (not attention weights).

### Issue: Shape mismatch in predictor

**Solution**: Check shapes at each step:
```python
print(f"Attention output: {attention_output.shape}")  # Should be [1, seq_len, 2048]
print(f"Predicted logits: {predicted_logits.shape}")  # Should be [1, seq_len, 22, 60]
```

### Issue: Low hit rate (<20%)

**Diagnosis**: Integration problem, not prediction problem.

**Solutions**:
1. Verify predictions are being used correctly
2. Check buffer loading with predicted experts
3. Ensure prefetch timing is correct (predict for layer N+2 during layer N)

### Issue: Different output from baseline

**Diagnosis**: Determinism issue or prefetch affecting computation.

**Solution**: Ensure:
- Same random seed
- Same model dtype
- Prefetch doesn't modify expert selection (only pre-loads)

## Integration with Fiddler

If `enable_cpu_offload=True`, the learned predictor should work with Fiddler's CPU offloading:

1. Predictor predicts which experts are needed
2. Fiddler decides which to keep on GPU vs CPU based on cost model
3. Prefetch loads GPU-resident experts from predictions
4. CPU-resident experts loaded on-demand

Both systems work together:
- **Predictor**: Determines which experts likely needed
- **Fiddler**: Determines optimal placement (CPU vs GPU)

## Next Steps

Once integration is complete and tests pass:
1. Proceed to **Phase 5: Benchmarking** (PREDICTOR_PHASE5_BENCHMARKING.md)
2. Compare learned approach with baseline and pattern-based
3. Measure actual inference speedup

## References

- See [PREDICTOR_SHARED_CONTEXT.md](PREDICTOR_SHARED_CONTEXT.md) for:
  - Architecture details
  - Hook implementation
  - Expected performance
- See existing `src/fiddler/qwen_with_prefetch.py` for:
  - Buffer management
  - Hit rate tracking
  - Interface requirements

---

**Estimated Completion Time**: 1 week (including implementation, testing, debugging, and verification)
