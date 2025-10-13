# Phase 5: End-to-End Benchmarking

**Estimated Time**: 1 week
**Prerequisites**: Completed Phase 4 (integrated and tested)
**Dependencies**: See [PREDICTOR_SHARED_CONTEXT.md](PREDICTOR_SHARED_CONTEXT.md) for expected results

## Objective

Compare the learned predictor approach against baseline and pattern-based prefetching to measure actual inference speedup, hit rates, and overall performance across different batch sizes and configurations.

## Deliverables

- `benchmark_prediction_methods.py`: Comprehensive benchmark script
- Benchmark results showing:
  - Inference speedup vs baseline
  - Hit rates (prefill and decode)
  - Comparison with pattern-based approach
- Performance analysis document
- Production deployment recommendations

## Implementation

### Script: `benchmark_prediction_methods.py`

Create comprehensive benchmark comparing all approaches:

```python
#!/usr/bin/env python3
"""
benchmark_prediction_methods.py - Compare prediction methods
Compares: Baseline, Pattern-based, Learned, and combinations with Fiddler
"""

import torch
import time
import json
from datetime import datetime
from pathlib import Path
import sys
sys.path.append('src/fiddler')

from qwen import FiddlerQwen  # Baseline
from qwen_with_prefetch import FiddlerQwenWithPrefetch  # Pattern-based
from qwen_with_learned_prefetch import FiddlerQwenWithLearnedPrefetch  # Learned


class Args:
    """Simple args container"""
    def __init__(self):
        self.model_path = "Qwen/Qwen1.5-MoE-A2.7B"


def benchmark_configuration(model_class, config_name, test_inputs, output_tokens=20, **model_kwargs):
    """
    Benchmark a single configuration.

    Args:
        model_class: Model class to instantiate
        config_name: Name for this configuration
        test_inputs: List of test prompts
        output_tokens: Number of tokens to generate
        **model_kwargs: Arguments for model initialization

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {config_name}")
    print(f"{'='*80}")

    args = Args()
    model = model_class(args, **model_kwargs)

    results = {
        'config_name': config_name,
        'model_kwargs': model_kwargs,
        'samples': []
    }

    # Run on each test input
    for idx, test_input in enumerate(test_inputs):
        print(f"\n[{idx+1}/{len(test_inputs)}] Processing: {test_input[:50]}...")

        try:
            prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
                test_input,
                output_token=output_tokens
            )

            total_time = prefill_time + decode_time
            tokens_per_sec = output_tokens / decode_time if decode_time > 0 else 0

            sample_result = {
                'input': test_input,
                'prefill_time': prefill_time,
                'decode_time': decode_time,
                'total_time': total_time,
                'tokens_per_sec': tokens_per_sec,
                'prefill_hit_rate': prefill_hit_rate,
                'decode_hit_rate': decode_hit_rate
            }

            results['samples'].append(sample_result)

            print(f"  Total: {total_time:.3f}s | Decode: {decode_time:.3f}s | "
                  f"Tok/s: {tokens_per_sec:.1f} | Decode hit: {decode_hit_rate*100:.1f}%")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results['samples'].append({
                'input': test_input,
                'error': str(e)
            })

    # Calculate averages
    valid_samples = [s for s in results['samples'] if 'error' not in s]

    if valid_samples:
        results['avg_prefill_time'] = sum(s['prefill_time'] for s in valid_samples) / len(valid_samples)
        results['avg_decode_time'] = sum(s['decode_time'] for s in valid_samples) / len(valid_samples)
        results['avg_total_time'] = sum(s['total_time'] for s in valid_samples) / len(valid_samples)
        results['avg_tokens_per_sec'] = sum(s['tokens_per_sec'] for s in valid_samples) / len(valid_samples)
        results['avg_prefill_hit_rate'] = sum(s['prefill_hit_rate'] for s in valid_samples) / len(valid_samples)
        results['avg_decode_hit_rate'] = sum(s['decode_hit_rate'] for s in valid_samples) / len(valid_samples)

        print(f"\n📊 AVERAGES:")
        print(f"  Total time: {results['avg_total_time']:.3f}s")
        print(f"  Decode time: {results['avg_decode_time']:.3f}s")
        print(f"  Tokens/sec: {results['avg_tokens_per_sec']:.1f}")
        print(f"  Prefill hit rate: {results['avg_prefill_hit_rate']*100:.1f}%")
        print(f"  Decode hit rate: {results['avg_decode_hit_rate']*100:.1f}%")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


def run_full_benchmark():
    """Run comprehensive benchmark across all configurations."""

    # Test inputs - diverse to test generalization
    test_inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "In machine learning, a neural network is a computational model inspired by biological neural networks.",
        "The Eiffel Tower is located in Paris, France and was completed in 1889.",
        "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement.",
        "Python is a high-level, interpreted programming language known for its simplicity and readability.",
    ]

    output_tokens = 20

    # Configurations to benchmark
    configurations = [
        # 1. Baseline (no prefetch, no offload)
        {
            'model_class': FiddlerQwen,
            'config_name': 'Baseline',
            'kwargs': {}
        },

        # 2. Pattern-based prefetch (no offload)
        {
            'model_class': FiddlerQwenWithPrefetch,
            'config_name': 'Pattern-Prefetch-8',
            'kwargs': {
                'num_experts_to_prefetch': 8,
                'enable_cpu_offload': False,
                'pattern_file': 'expert_usage_patterns_qwen.json'
            }
        },

        # 3. Learned prefetch (no offload)
        {
            'model_class': FiddlerQwenWithLearnedPrefetch,
            'config_name': 'Learned-Prefetch-8',
            'kwargs': {
                'num_experts_to_prefetch': 8,
                'enable_cpu_offload': False,
                'predictor_path': 'predictor_checkpoints/best_model.pt'
            }
        },

        # 4. Fiddler CPU-only (no prefetch)
        {
            'model_class': FiddlerQwen,
            'config_name': 'Fiddler-CPU',
            'kwargs': {
                'enable_cpu_offload': True,
                'latency_cpu': 0.1,
                'latency_gpu': 10.0
            }
        },

        # 5. Fiddler + Pattern prefetch
        {
            'model_class': FiddlerQwenWithPrefetch,
            'config_name': 'Fiddler+Pattern-Prefetch-8',
            'kwargs': {
                'num_experts_to_prefetch': 8,
                'enable_cpu_offload': True,
                'latency_cpu': 0.1,
                'latency_gpu': 10.0,
                'pattern_file': 'expert_usage_patterns_qwen.json'
            }
        },

        # 6. Fiddler + Learned prefetch
        {
            'model_class': FiddlerQwenWithLearnedPrefetch,
            'config_name': 'Fiddler+Learned-Prefetch-8',
            'kwargs': {
                'num_experts_to_prefetch': 8,
                'enable_cpu_offload': True,
                'latency_cpu': 0.1,
                'latency_gpu': 10.0,
                'predictor_path': 'predictor_checkpoints/best_model.pt'
            }
        },
    ]

    # Run benchmarks
    all_results = []

    for config in configurations:
        result = benchmark_configuration(
            config['model_class'],
            config['config_name'],
            test_inputs,
            output_tokens=output_tokens,
            **config['kwargs']
        )
        all_results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"prediction_methods_benchmark_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Generate comparison table
    generate_comparison_table(all_results, output_dir)

    return all_results


def generate_comparison_table(all_results, output_dir):
    """Generate comparison table and analysis."""

    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON")
    print("="*100)

    # Find baseline for speedup calculation
    baseline_result = next((r for r in all_results if r['config_name'] == 'Baseline'), None)

    if not baseline_result or 'avg_decode_time' not in baseline_result:
        print("Warning: Baseline results not found, cannot calculate speedups")
        return

    baseline_decode_time = baseline_result['avg_decode_time']

    # Table header
    print(f"\n{'Configuration':<30} {'Decode (s)':<12} {'Speedup':<10} {'Tok/s':<10} {'Decode Hit':<12}")
    print("-"*100)

    comparison_data = []

    for result in all_results:
        if 'avg_decode_time' not in result:
            continue

        config_name = result['config_name']
        decode_time = result['avg_decode_time']
        speedup = baseline_decode_time / decode_time if decode_time > 0 else 0
        tok_s = result['avg_tokens_per_sec']
        decode_hit = result['avg_decode_hit_rate'] * 100

        comparison_data.append({
            'config': config_name,
            'decode_time': decode_time,
            'speedup': speedup,
            'tok_s': tok_s,
            'decode_hit_rate': decode_hit
        })

        print(f"{config_name:<30} {decode_time:<12.3f} {speedup:<10.2f}x {tok_s:<10.1f} {decode_hit:<12.1f}%")

    # Save comparison table
    comparison_file = output_dir / "comparison_table.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    # Analysis
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)

    # Find best configuration
    best_speedup = max(comparison_data, key=lambda x: x['speedup'])
    best_hit_rate = max(comparison_data, key=lambda x: x['decode_hit_rate'])

    print(f"\n🏆 Best speedup: {best_speedup['config']} ({best_speedup['speedup']:.2f}x)")
    print(f"🎯 Best decode hit rate: {best_hit_rate['config']} ({best_hit_rate['decode_hit_rate']:.1f}%)")

    # Compare learned vs pattern-based
    learned_result = next((r for r in comparison_data if 'Learned-Prefetch-8' in r['config']), None)
    pattern_result = next((r for r in comparison_data if 'Pattern-Prefetch-8' in r['config']), None)

    if learned_result and pattern_result:
        print(f"\n📊 Learned vs Pattern-based:")
        print(f"  Speedup: {learned_result['speedup']:.2f}x vs {pattern_result['speedup']:.2f}x")
        print(f"  Decode hit rate: {learned_result['decode_hit_rate']:.1f}% vs {pattern_result['decode_hit_rate']:.1f}%")

        if learned_result['decode_hit_rate'] > pattern_result['decode_hit_rate'] * 0.8:
            print(f"  ✅ Learned predictor achieves competitive hit rate!")
        else:
            print(f"  ⚠️  Learned predictor hit rate lower than expected")

    # Save analysis
    analysis = {
        'best_speedup': {
            'config': best_speedup['config'],
            'speedup': best_speedup['speedup']
        },
        'best_hit_rate': {
            'config': best_hit_rate['config'],
            'hit_rate': best_hit_rate['decode_hit_rate']
        },
        'learned_vs_pattern': {
            'learned_speedup': learned_result['speedup'] if learned_result else None,
            'pattern_speedup': pattern_result['speedup'] if pattern_result else None,
            'learned_hit_rate': learned_result['decode_hit_rate'] if learned_result else None,
            'pattern_hit_rate': pattern_result['decode_hit_rate'] if pattern_result else None,
        }
    }

    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✅ Analysis saved to: {analysis_file}")


def main():
    print("="*100)
    print("PREDICTION METHODS BENCHMARK")
    print("="*100)
    print("\nComparing:")
    print("  1. Baseline (no optimization)")
    print("  2. Pattern-based prefetch")
    print("  3. Learned prefetch")
    print("  4. Fiddler CPU offload")
    print("  5. Fiddler + Pattern prefetch")
    print("  6. Fiddler + Learned prefetch")
    print("="*100)

    results = run_full_benchmark()

    print("\n" + "="*100)
    print("BENCHMARK COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
```

## Step-by-Step Instructions

### 1. Verify Prerequisites

Ensure all previous phases are complete:

```bash
# Phase 2: Trained model
ls predictor_checkpoints/best_model.pt

# Phase 3: Evaluation results
ls predictor_evaluation_results.json

# Phase 4: Integration
ls src/fiddler/qwen_with_learned_prefetch.py
python test_learned_prefetch.py  # Should pass
```

### 2. Prepare Test Environment

Clear GPU memory:
```bash
nvidia-smi  # Check current usage
```

Ensure no other processes using GPU.

### 3. Run Benchmark

```bash
python benchmark_prediction_methods.py
```

**Expected runtime**: 30-60 minutes for all configurations

**What happens**:
- Loads each configuration sequentially
- Runs 5 diverse test prompts per configuration
- Measures decode time, hit rates, tokens/sec
- Generates comparison tables and analysis

### 4. Review Results

Results saved to `prediction_methods_benchmark_YYYYMMDD_HHMMSS/`:
- `benchmark_results.json`: Raw results
- `comparison_table.json`: Comparison table
- `analysis.json`: Key findings

### 5. Analyze Performance

Expected patterns:

**Speedup (decode time)**:
- Baseline: 1.0x
- Pattern-Prefetch: ~1.3x (100% hit on training prompts)
- Learned-Prefetch: 1.2-1.3x (good generalization)
- Fiddler-CPU: ~2.2x (BS=1)
- Fiddler+Pattern: ~1.5x
- Fiddler+Learned: ~1.4-1.5x

**Decode Hit Rate**:
- Baseline: 0%
- Pattern-Prefetch: ~100% (on training prompts), <20% (on new prompts)
- Learned-Prefetch: 50-70% (generalizes to new prompts)
- Others: Similar to their prefetch component

## Verification Checklist

Phase 5 is complete when:

- [ ] Benchmark runs successfully on all configurations
- [ ] Results show learned predictor achieves >50% decode hit rate
- [ ] Learned approach shows 1.2-1.4x speedup over baseline
- [ ] Learned predictor generalizes to new prompts (unlike pattern-based)
- [ ] Comparison table generated
- [ ] Analysis document created
- [ ] Performance meets or exceeds minimum viable criteria from Phase 3
- [ ] Results are reproducible

## Expected Results

### Success Criteria

**Minimum Viable**:
- Learned predictor: >40% decode hit rate
- Speedup: >1.1x over baseline
- Better than random (6.67%) and competitive with gating (18.75%)

**Target**:
- Learned predictor: >60% decode hit rate
- Speedup: 1.2-1.4x over baseline
- Good generalization to diverse prompts

**Stretch**:
- Learned predictor: >70% decode hit rate
- Speedup: 1.4-1.6x over baseline
- Comparable to pattern-based on training prompts

### Comparison with Existing Work

| Metric | Baseline | Pattern | Learned | Fiddler | Fiddler+Learned |
|--------|----------|---------|---------|---------|-----------------|
| Decode Hit Rate | 0% | 100% (fixed) | 60-70% | 0% | 60-70% |
| Speedup (BS=1) | 1.0x | 1.28x | 1.2-1.3x | 2.17x | 1.4-1.5x |
| Generalization | N/A | Poor | Good | N/A | Good |
| Production Ready | ✅ | ❌ | ✅ | ✅ | ✅ |

## Troubleshooting

### Issue: All configurations show similar performance

**Diagnosis**: Benchmark may not be measuring correctly.

**Solutions**:
1. Verify hit rates are being tracked
2. Check decode time measurement is correct
3. Ensure prefetching is actually happening

### Issue: Learned predictor slower than baseline

**Diagnosis**: Predictor overhead or integration issue.

**Solutions**:
1. Profile predictor inference time (should be <1ms)
2. Check if predictor is running on GPU (not CPU)
3. Ensure predictions are cached (not recomputed each token)

### Issue: Low hit rate (<40%) for learned predictor

**Diagnosis**: Integration issue or prediction quality problem.

**Solutions**:
1. Verify Phase 3 evaluation showed >40% accuracy
2. Check expert predictions are being used correctly
3. Debug with print statements showing predicted vs actual experts

## Production Deployment Recommendations

Based on benchmark results:

### Single Request (BS=1)
- **Best**: Fiddler CPU-only (2.17x speedup)
- **Alternative**: Fiddler + Learned (1.4-1.5x, better generalization)

### Small Batches (BS=2-16)
- **Best**: Fiddler + Learned prefetch
- **Benefit**: Combines CPU offload with predictive prefetch

### Large Batches (BS=32+)
- **Best**: Learned prefetch (no CPU offload)
- **Reason**: GPU parallelism scales better

### Diverse Workloads
- **Best**: Learned prefetch
- **Advantage**: Generalizes to unseen prompts (unlike pattern-based)

### Fixed Query Set
- **Best**: Pattern-based prefetch
- **Advantage**: 100% hit rate on training prompts

## Final Deliverables

At the end of Phase 5:

1. **Code**:
   - `benchmark_prediction_methods.py`: Comprehensive benchmark
   - All supporting scripts from previous phases

2. **Results**:
   - `prediction_methods_benchmark_YYYYMMDD_HHMMSS/`: Complete benchmark results
   - Comparison tables and analysis

3. **Documentation**:
   - Performance analysis
   - Deployment recommendations
   - Lessons learned

4. **Production-Ready**:
   - `src/fiddler/qwen_with_learned_prefetch.py`: Ready for deployment
   - Verified correctness and performance

## Next Steps

After Phase 5 completion:

1. **Update guide.md** with:
   - Learned predictor documentation
   - Benchmark results
   - Deployment recommendations

2. **Create summary document**:
   - Implementation journey (Phases 1-5)
   - Key findings
   - Future work

3. **Consider improvements**:
   - Ensemble (learned + pattern)
   - Online learning
   - Per-layer predictors

4. **Deploy to production**:
   - Choose configuration based on workload
   - Monitor performance
   - Collect feedback

## References

- See [PREDICTOR_SHARED_CONTEXT.md](PREDICTOR_SHARED_CONTEXT.md) for:
  - Expected performance metrics
  - Comparison baselines
  - Success criteria

- See existing benchmark scripts:
  - `benchmark_prefetch_configs.py`: Pattern-based benchmark
  - `benchmark_batch_size_sweep.py`: Batch size analysis

---

**Estimated Completion Time**: 1 week (including benchmark runs, analysis, and documentation)

**Total Project Time**: 5 weeks (all phases)

**Final Status**: Production-ready learned expert predictor with verified performance and comprehensive documentation
