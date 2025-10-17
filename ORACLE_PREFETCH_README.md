# Oracle Prefetch - Perfect Expert Prediction

This feature provides "oracle" (perfect) expert prefetching to establish the **upper bound performance** achievable with ideal expert prediction.

## Overview

The oracle prefetch configurations use pre-collected gating decisions from actual model runs to achieve 100% prediction accuracy. This shows the maximum possible speedup achievable with perfect expert prediction.

## How It Works

1. **Collection Phase**: Run prompts through the model and capture the exact expert selections made by the gating function
2. **Prefetch Phase**: Use the pre-collected decisions to prefetch exactly the experts that will be needed
3. **Result**: 100% hit rate (upper bound performance)

## Usage

### Step 1: Collect Oracle Gating Decisions

Run the collection script to capture gating decisions for your benchmark prompts:

```bash
python collect_oracle_gating_decisions.py
```

This will:
- Load the Qwen model
- Run the same prompts used in the benchmark
- Capture gating decisions at each layer and token position
- Save to `oracle_gating_decisions.json` (~100MB for full benchmark)

**Configuration in collection script:**
- Batch sizes: 1, 2, 4, 8, 16 (matching benchmark)
- Output tokens: 20 (matching benchmark)
- Trials: 3 (matching benchmark)

**Expected duration**: 2-4 hours (depends on batch sizes and number of prompts)

### Step 2: Run Benchmark with Oracle Configurations

The benchmark now includes two oracle configurations:

```python
configurations = [
    # ... existing configurations ...
    {
        'name': 'Oracle-Prefetch',
        'use_fiddler_mode': False,
        'kwargs': {
            'num_experts_to_prefetch': 4,  # Qwen uses top-4
            'enable_cpu_offload': False,
            'oracle_path': 'oracle_gating_decisions.json'
        }
    },
    {
        'name': 'Fiddler+Oracle-Prefetch',
        'use_fiddler_mode': False,
        'kwargs': {
            'num_experts_to_prefetch': 4,  # Qwen uses top-4
            'enable_cpu_offload': True,
            'latency_cpu': 0.1,
            'latency_gpu': 10.0,
            'oracle_path': 'oracle_gating_decisions.json'
        }
    }
]
```

Run the benchmark:

```bash
python benchmark_prediction_methods.py
```

### Step 3: Analyze Results

The oracle configurations will show:

**Expected Hit Rates:**
- **Oracle-Prefetch**: 100% (perfect prediction)
- **Fiddler+Oracle-Prefetch**: 100% (perfect prediction with CPU offload)

**Performance Interpretation:**
- **Oracle-Prefetch** shows the best possible performance without CPU offload
- **Fiddler+Oracle-Prefetch** shows the best possible performance with CPU offload
- The difference between learned/pattern-based and oracle shows the **gap to perfect prediction**

## File Formats

### Oracle Decisions File (`oracle_gating_decisions.json`)

```json
{
  "model": "Qwen/Qwen1.5-MoE-A2.7B",
  "n_experts": 60,
  "top_k": 4,
  "moe_layers": [2, 3, ..., 23],
  "gating_decisions": {
    "bs1_seed0_prompt0": {
      "2": {  // layer_idx
        "0": [12, 34, 45, 56],  // token_pos: [expert_ids]
        "1": [23, 34, 45, 57],
        ...
      },
      ...
    },
    ...
  }
}
```

**Key format**: `bs{batch_size}_seed{trial}_prompt{idx}`
- `batch_size`: Size of the batch (1, 2, 4, 8, 16)
- `trial`: Trial number (0, 1, 2 matching benchmark seeds)
- `idx`: Index within the batch (0 to batch_size-1)

## Implementation Details

### Oracle Prefetch Class

**Location**: `src/fiddler/qwen_with_oracle_prefetch.py`

**Key Features:**
- Extends `FiddlerQwenWithPrefetch` base class
- Loads pre-collected gating decisions from JSON
- Supports batched generation by unioning experts from all batch elements
- Achieves 100% hit rate by prefetching exactly what's needed

**Usage in code:**
```python
from fiddler.qwen_with_oracle_prefetch import FiddlerQwenWithOraclePrefetch

model = FiddlerQwenWithOraclePrefetch(
    args,
    oracle_path='oracle_gating_decisions.json',
    num_experts_to_prefetch=4,
    enable_cpu_offload=False
)

# For single prompt
result = model.generate(
    "The capital of France is",
    output_token=20,
    prompt_keys="bs1_seed0_prompt0"
)

# For batched prompts
result = model.generate(
    ["prompt1", "prompt2"],
    output_token=20,
    prompt_keys=["bs2_seed0_prompt0", "bs2_seed0_prompt1"]
)
```

### Collection Script

**Location**: `collect_oracle_gating_decisions.py`

**How it works:**
1. Hooks into MoE layer gating to capture expert selections
2. Runs prompts through model with token-by-token generation
3. Tracks layer index and token position for each decision
4. Saves all decisions to JSON file

## Comparison with Other Methods

| Method | Hit Rate | What It Shows |
|--------|----------|---------------|
| **Baseline** | N/A | On-demand loading performance |
| **Pattern-Based** | ~20% | Pattern prediction from token position |
| **Learned** | ~47% | Attention-based neural prediction |
| **Oracle** | **100%** | **Upper bound with perfect prediction** |

## Notes

- **Oracle is not practical for deployment** - it requires knowing the future (what experts will be needed)
- **Oracle shows the ceiling** - the maximum possible benefit from prefetching
- **Gap analysis**: Compare learned vs oracle to see remaining optimization potential
- **Fiddler+Oracle**: Shows best possible performance with CPU offloading strategy

## Files Created

- `collect_oracle_gating_decisions.py` - Collection script
- `src/fiddler/qwen_with_oracle_prefetch.py` - Oracle prefetch implementation
- `oracle_gating_decisions.json` - Collected gating decisions (generated by script)
- `ORACLE_PREFETCH_README.md` - This documentation

## Quick Start

```bash
# 1. Collect oracle decisions (2-4 hours)
python collect_oracle_gating_decisions.py

# 2. Run benchmark with oracle configurations
python benchmark_prediction_methods.py

# 3. Check results
cat phase5_benchmark_*/ANALYSIS.md
```

Look for "Oracle-Prefetch" and "Fiddler+Oracle-Prefetch" in the results to see upper bound performance!
