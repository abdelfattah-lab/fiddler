# Fiddler MoE Optimization Project - Agent Guide

## Guidelines
You're a genius world class researcher and software engineer. You can achieve any goal. You do not stop until the goal is fully achieved and you do not take shortcuts that compromise the reliability of the results.
Always update guide.md to prepare it for another agent to look at it and understand the full state of the system and keep it concise. At the end of that, add all files changed (that are relevant) including guide.md to git and suggest a commit message but let me do the git commit.
Don't stop till you achieve the goal in a reliable way without shortcuts or workarounds. Make sure you validate the results and the correctness.

## Current Goal

Investigate and fix the Oracle Prefetch implementation - the recent fix is not satisfactory. When running the benchmark (results are in phase5_benchmark_20251018_171454), it seems like only batch 1 is working properly. Here are my suggested steps to resolve the issues:
Make a new script that runs only the Oracle Prefetch, the script should run the collection first and then the measurements. You should look at the efficiency and make sure that it is 100% for both all decode steps and prefill for all batch sizes. Efficiency being defined as the number of prefetched experts that were actually used divided by the total number of prefetched experts. Make sure you validate the correctness of your implementation and don't give up till you have a reliable and working Oracle Prefetch implementation that achieves 100% efficiency for all batch sizes.

### Context
The Oracle-Prefetch implementation was recently "fixed" to achieve 98.7% decode hit rate, but this result is suspect and likely not a genuine fix. Need to investigate why the oracle prefetch is not working as expected and implement a proper solution.

## 📊 Project Status

**Branch**: `predictor_review`
**Model**: Qwen/Qwen1.5-MoE-A2.7B (60 experts, top-4 selection, 24 MoE layers)

### Recent Work - Oracle Prefetch Implementation (INCOMPLETE/PROBLEMATIC)

⚠️ **Status**: The recent "fix" claiming 98.7% decode hit rate is **NOT WORKING** as intended. Results are questionable.

**What Was Attempted (2025-10-18)**:
Multiple attempts were made to fix Oracle-Prefetch to achieve perfect (100%) hit rate:

1. **First Attempt**: Fixed token position tracking bugs
   - Modified `qwen_with_prefetch.py` to properly track token positions during prefill/decode
   - Fixed `collect_oracle_gating_decisions.py` to record correct token positions
   - Result: Still didn't achieve reliable oracle performance

2. **Second Attempt**: Fixed model loading method mismatch
   - **Problem Identified**: Oracle data was collected using `device_map="auto"` loading, but FiddlerQwen uses custom expert management
   - Different loading methods caused different numerical behavior → different expert selections
   - **Solution**: Created `collect_oracle_with_fiddler.py` that uses FiddlerQwen base class for data collection
   - Claimed to achieve 98.7% decode hit rate

**Why This is Suspicious**:
- Oracle (perfect prediction) should achieve **100%** hit rate, not 98.7%
- Multiple "fixes" applied without clear root cause analysis
- Data was re-collected multiple times with different approaches
- The approach keeps changing, suggesting the underlying issue is not understood
- Results need validation with full benchmark run (not done yet)

**Key Files Involved**:
- `src/fiddler/qwen_with_oracle_prefetch.py` - Oracle prefetch implementation
- `collect_oracle_with_fiddler.py` - Data collection using FiddlerQwen loading
- `oracle_gating_decisions.json` - Current oracle data (only BS=1)
- `oracle_gating_decisions_old_buggy.json` - Previous attempt's data

**What Needs Investigation**:
1. Why isn't oracle achieving 100% hit rate if it has perfect information?
2. Is there a fundamental architecture mismatch?
3. Does the buffer management system have issues?
4. Are there bugs in how oracle predictions are being used?
5. Need to run full benchmark to validate actual performance

## 🏗️ Stable System Components

### ✅ Learned Prefetch System (WORKING WELL)

**Status**: Production-ready with 46.8% top-4 prediction accuracy

**Key Achievements**:
1. **Phase 1-2**: Successfully collected 3.3M samples and trained attention-based predictor
   - Training data: WikiText-103 (45k train / 5k validation)
   - Model: 6.90M parameter MLP
   - Training: BCE loss for direct expert selection optimization
   - Result: 46.8% top-4 accuracy (exceeds 40% target)

2. **Phase 4**: Integration with Fiddler system complete
   - `src/fiddler/qwen_with_learned_prefetch.py` - Captures layer 0 attention, uses trained predictor
   - Async predictor execution in separate CUDA stream (no main thread blocking)
   - Supports batched generation with frequency-based expert aggregation
   - Compatible with Fiddler CPU offloading

3. **Phase 5**: Comprehensive benchmarking completed
   - Correctness validation: All configurations produce identical outputs
   - Hit rate tracking: Separate for prefill (30-40%) and decode (40-55%)
   - Batched generation: Properly supports BS=1,2,4,8,16 with non-zero hit rates
   - Results: `phase5_benchmark_20251014_145424/` (most recent reliable benchmark)

### ✅ Pattern-Based Prefetch System (WORKING)

**Status**: Production-ready baseline

- Uses token position patterns collected from actual runs
- Simpler than learned predictor but lower accuracy (18.75%)
- Stored in `expert_usage_patterns_qwen.json`
- Good baseline for comparison

### ✅ Fiddler CPU Offloading (WORKING)

**Status**: Production-ready

- Dynamic CPU/GPU partitioning with greedy cost model
- Best for BS=1 (2.17x speedup vs baseline)
- Configurable latency parameters: `latency_cpu=0.1`, `latency_gpu=10.0`
- Properly integrated with both prefetch systems

### 📋 Completed Development Phases

| Phase | Status | Deliverable |
|-------|--------|-------------|
| Phase 1: Data Collection | ✅ Complete | `predictor_training_data/` (3.3M samples) |
| Phase 2: Model Training | ✅ Complete | `predictor_checkpoints/best_model.pt` (46.8% accuracy) |
| Phase 3: Evaluation | ✅ Complete | Validated on test set |
| Phase 4: Integration | ✅ Complete | `qwen_with_learned_prefetch.py` |
| Phase 5: Benchmarking | ✅ Complete | `phase5_benchmark_20251014_145424/` |
| Oracle Prefetch | ⚠️ Incomplete | Hit rate issues, needs investigation |

### 🎯 System Performance Summary

**Validated Configurations** (from Phase 5 benchmarks):
1. **Baseline**: On-demand expert loading
2. **Pattern-Prefetch**: Token position-based prefetching (8 experts)
3. **Learned-Prefetch**: Attention-based predictor (46.8% accuracy)
4. **Fiddler**: CPU offloading with dynamic partitioning
5. **Fiddler+Learned-Prefetch**: Combined approach (best overall)

**Key Performance Results**:
- **BS=1**: Fiddler CPU-only achieves 2.17x speedup (21.2 tok/s)
- **BS=2-16**: Fiddler+Learned-Prefetch provides 1.06-1.15x speedup over Fiddler-only
- **BS=32+**: Pattern-Prefetch-only best for large batches (46.8 tok/s)
- **Learned vs Pattern**: Learned predictor achieves 46.8% accuracy vs 18.75% for pattern-based

**Technical Insights**:
- Gating-based prediction has only 7.5% expert overlap between adjacent layers
- Token position-based prediction is superior to gating-based
- Learned predictor significantly outperforms pattern-based prediction
- GPU parallelism scales better than CPU at higher batch sizes

## 🏗️ Core Implementations

### Model Files (`src/fiddler/`)

**`qwen.py`** - Baseline implementation (single GPU buffer, on-demand loading)

**`qwen_with_prefetch.py`** - Unified implementation supporting all 4 configurations

**`qwen_with_learned_prefetch.py`** - ✨ NEW: Learned prefetch using attention-based predictor:
- Uses trained predictor instead of token position patterns
- Captures layer 0 attention output for prediction
- Achieves ~46.8% prediction accuracy (vs 18.75% for pattern-based)
- Supports all Fiddler features (CPU offloading, dual buffers)

**Configuration Details for `qwen_with_prefetch.py`**:
- **Configuration via parameters**:
  - Baseline: `num_experts_to_prefetch=0, enable_cpu_offload=False`
  - Prefetch: `num_experts_to_prefetch>0, enable_cpu_offload=False`
  - Fiddler: `num_experts_to_prefetch=0, enable_cpu_offload=True`
  - Fiddler+Prefetch: `num_experts_to_prefetch>0, enable_cpu_offload=True`
- **Prefetch**: Dual buffer (A/B), pattern-based prediction, async CUDA stream
- **Fiddler**: Dynamic CPU/GPU partitioning with greedy cost model algorithm
- Returns: `(prefill_time, decode_time, prefill_hit_rate, decode_hit_rate)`

### Benchmarking & Analysis Tools

- **`benchmark_prefetch_configs.py`**: Full benchmark suite (single input)
- **`benchmark_batch_size_sweep.py`**: Batch size sweep (1-32)
- **`benchmark_prediction_methods.py`**: ✨ Phase 5 comprehensive benchmark (learned predictor with diverse batches)
- **`quick_test.py`**: Fast 3-token correctness validation
- **`analyze_gating_prediction_accuracy.py`**: Gating prediction analysis

## 🔧 Technical Architecture

### Buffer System
- **Baseline**: Single GPU buffer (one expert at a time)
- **Prefetch**: Dual buffer (A/B) for even/odd layers, async stream, cache tracking
- **Fiddler**: Dynamic partitioning with cost model: `cpu_cost = num_tokens * latency_cpu`, `gpu_cost = latency_gpu`

### Prefetch Strategy
- **Collection mode** (first run): Records patterns to `expert_usage_patterns_qwen.json`
- **Prediction mode**: Prefetches layer N+2 experts during layer N execution based on token position
- Pattern structure: `token_pos → layer_id → [expert_ids]`

### Timing Tracking
- **Prefill phase**: `sequence_length > 1` (prompt processing)
- **Decode phase**: `sequence_length == 1` (autoregressive generation)
- Return: `(prefill_time, decode_time, prefill_hit_rate, decode_hit_rate)`

## 📋 Interface Requirements

All implementations must support:
```python
class YourImplementation:
    def __init__(self, args, **kwargs):
        # Initialize model

    def generate(self, text, output_token=20, input_token=None):
        # Return (prefill_time, decode_time, prefill_hit_rate, decode_hit_rate)

    def tokenize(self, text):
        # Return (input_ids, position_ids)

    def mixtral_forward(self, input_ids, position_ids, is_decode):
        # Core inference - return logits tensor
```

## ⚠️ Critical Configuration

### Testing Settings
- `cpu_offload=0`, `max_experts_gpu=0`, `beam_width=1`
- `eos_token_id=None` (forces exact token count for consistent benchmarking)

### Memory Management
- Use `torch.cuda.empty_cache()` between model loads
- Pinned CPU memory required for async transfers

### Key Implementation Details
- **Dtype consistency**: Expert buffers use `dtype=self.dtype` (bfloat16)
- **Async synchronization**: Event synchronization required before using prefetched experts (critical for correctness)
- **MoE semantics**: One-hot expert masking, index_add accumulation, shared expert
- **GPU-resident layers**: Layers 0-1 kept on GPU permanently

### Fiddler Cost Model
- **Qwen optimal**: `latency_cpu=0.1`, `latency_gpu=10.0` (hardware-specific)
- Use `profile_qwen_expert_costs.py` to measure actual costs for your hardware

## 🚀 Usage

### Benchmarking
```bash
# Single input benchmark
python benchmark_prefetch_configs.py

# Batch size sweep
python benchmark_batch_size_sweep.py

# Results in: prefetch_benchmark_YYYYMMDD_HHMMSS/ or batch_size_sweep_YYYYMMDD_HHMMSS/
```

### Testing
```bash
# Correctness validation
python quick_test.py FiddlerQwen
python quick_test.py FiddlerQwenWithPrefetch
```

### Profiling
```bash
# Generate Nsight profiles
python profile_qwen_prefetch.py

# Analyze
nsight-sys qwen_prefetch_profile_*/qwen_prefetch_prediction.nsys-rep
nsys stats --report nvtx_sum qwen_prefetch_profile_*/qwen_prefetch_prediction.nsys-rep
```

## 📁 Key Files

### Core
- `src/fiddler/qwen.py`: Baseline implementation
- `src/fiddler/qwen_with_prefetch.py`: Pattern-based prefetch + Fiddler implementation
- `src/fiddler/qwen_with_learned_prefetch.py`: ✨ Learned prefetch + Fiddler implementation

### Predictor Files
- `train_predictor.py`: Predictor model training script
- `collect_training_data.py`: Training data collection
- `test_learned_prefetch.py`: Integration validation tests
- `predictor_checkpoints/best_model.pt`: Trained predictor (46.8% accuracy)
- `predictor_training_data/`: Training data (3.3M samples)

### Tools
- `benchmark_prefetch_configs.py`: Single input benchmark
- `benchmark_batch_size_sweep.py`: Batch size sweep
- `benchmark_prediction_methods.py`: Phase 5 comprehensive benchmark
- `quick_test.py`: Correctness validation
- `analyze_gating_prediction_accuracy.py`: Gating prediction analysis
- `profile_qwen_expert_costs.py`: Hardware cost profiling

### Data
- `expert_usage_patterns_qwen.json`: Prefetch patterns
- `batch_size_sweep_20251009_183710/`: Previous benchmark results with ANALYSIS.md
- `phase5_benchmark_20251013_132306/`: ✨ Phase 5 benchmark results (learned predictor)
- `gating_analysis/`: Gating prediction accuracy analysis (validates pattern-based approach)

## 🎯 Deployment Recommendations

### Configuration by Batch Size

| Batch Size | Strategy | Configuration | Performance |
|------------|----------|---------------|-------------|
| **1** | Fiddler CPU-only | `enable_cpu_offload=True, num_experts_to_prefetch=0` | 21.2 tok/s |
| **2-16** | Fiddler+Prefetch | `enable_cpu_offload=True, num_experts_to_prefetch=8` | 5.7-18.6 tok/s |
| **32+** | Prefetch-only | `enable_cpu_offload=False, num_experts_to_prefetch=8` | 46.8 tok/s |

### For New Hardware
1. Profile costs: `python profile_qwen_expert_costs.py`
2. Tune `latency_cpu` and `latency_gpu` parameters
3. Run batch size sweep to find crossover point
4. Validate correctness: `python quick_test.py FiddlerQwenWithPrefetch`

### Model-Specific Notes
- Qwen's small experts favor CPU at BS=1, GPU at BS≥2
- Larger expert models (e.g., Mixtral) may have different crossover points
- Pattern-based prefetching (token position) outperforms gating-based (7.5% overlap)
