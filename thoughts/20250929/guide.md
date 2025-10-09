# Fiddler MoE Optimization Project - Agent Guide

## Guidelines

Always update guide.md to prepare it for another agent to look at it and understand the full state of the system and keep it concise. At the end of that, add all files changed (that are relevant) including guide.md to git and suggest a commit message but let me do the git commit.

## 📊 Project Status & Final Results

**STATUS**: ✅ Qwen MoE optimization fully implemented and benchmarked

The project successfully implemented and benchmarked **4 optimization configurations** for Qwen1.5-MoE-A2.7B:
1. **Baseline**: On-demand expert loading
2. **Prefetch**: Pattern-based expert prefetching (8 experts)
3. **Fiddler**: CPU offloading with dynamic partitioning
4. **Fiddler+Prefetch**: Combined approach

**Branch**: `qwen2_with_fiddler`

### Final Benchmark Results (October 9, 2025)

| Configuration | Total Time | Speedup | Prefill Hit % | Decode Hit % | CPU/GPU Split |
|--------------|-----------|---------|---------------|--------------|---------------|
| **Baseline** | 2.446s | 1.000x | 8.7% | 8.3% | - |
| **Prefetch-8** | 1.915s | **1.278x** | 44.1% | **100.0%** | - |
| **Fiddler** | 1.125s | **2.174x** 🏆 | 100.0% | 100.0% | 100% CPU / 0% GPU |
| **Fiddler+Prefetch** | 1.663s | **1.471x** | 100.0% | 100.0% | 13.3% CPU / 86.7% GPU |

### Key Findings

1. ✅ **Fiddler CPU Offloading is Optimal**: 2.17x speedup (BEST)
   - 100% CPU execution avoids GPU transfer overhead
   - Qwen's small experts execute efficiently on CPU

2. ✅ **Prefetching Works**: 1.28x speedup with 100% decode hit rate
   - Expert pattern prediction enables perfect prefetching during decode
   - Modest speedup due to Qwen's small expert sizes

3. ✅ **Fiddler+Prefetch Works**: 1.47x speedup
   - Improves over Prefetch alone (1.47x > 1.28x)
   - Demonstrates correct implementation of both techniques

4. ✅ **All Configurations Produce Identical Outputs**: Correctness verified

**Scientific Insight**: For Qwen, CPU execution is significantly faster than GPU execution even with prefetching. This is model-specific - Qwen has smaller experts that benefit less from GPU parallelism and CPU execution avoids GPU kernel launch overhead.

## 🏗️ Core Implementations

### Model Files (`src/fiddler/`)

**`qwen.py`** - Baseline implementation
- All model layers on GPU except experts (CPU)
- Single GPU expert buffer for on-demand loading
- Pinned CPU memory for async transfers
- Returns: `(prefill_time, decode_time, prefill_hit_rate, decode_hit_rate)`

**`qwen_with_prefetch.py`** - Unified prefetch + Fiddler implementation
- Supports 4 configurations via parameters:
  - Baseline: `num_experts_to_prefetch=0, enable_cpu_offload=False`
  - Prefetch: `num_experts_to_prefetch>0, enable_cpu_offload=False`
  - Fiddler: `num_experts_to_prefetch=0, enable_cpu_offload=True`
  - Fiddler+Prefetch: `num_experts_to_prefetch>0, enable_cpu_offload=True`
- **Prefetch features**:
  - Dual buffer system (A/B) for alternating layer prefetch
  - Collection mode: Records expert usage patterns
  - Prediction mode: Prefetches based on learned patterns
  - Layer+2 prefetch triggering
  - Async CUDA stream for non-blocking transfers
- **Fiddler features**:
  - Dynamic CPU/GPU partitioning using cost model
  - Greedy partitioning algorithm for 64 experts
  - Parallel CPU/GPU execution
  - Configurable cost parameters: `latency_cpu`, `latency_gpu`

**Legacy files** (reference only):
- `mixtral.py` - Original Mixtral baseline
- `mixtral_with_prefetch.py` - Mixtral prefetch version

### Benchmarking Tools

**`benchmark_prefetch_configs.py`** - Comprehensive benchmark suite
- Tests prefetch with 0-16 experts
- Measures prefill vs decode speedup separately
- Generates performance plots and CSV results
- Tracks hit rates and timing statistics

**`quick_test.py`** - Fast validation tool
- 3-token generation test for correctness
- Quick performance check

## 🔧 Technical Architecture

### Buffer System

**Baseline**: Single GPU expert buffer (one expert at a time)

**Prefetch**: Dual buffer system
- **Buffer A**: Even MoE layers (0, 2, 4, ...)
- **Buffer B**: Odd MoE layers (1, 3, 5, ...)
- Configurable slots per buffer
- Async prefetch stream separate from compute stream
- Cache tracking: `{(layer_idx, expert_idx): buffer_slot}`

**Fiddler**: Dynamic CPU/GPU partitioning
- Cost model: `cpu_cost = num_tokens * latency_cpu`, `gpu_cost = latency_gpu`
- Greedy partitioning minimizes: `max(sum_cpu_costs, sum_gpu_costs)`
- Expert location tracking: `expert_loc[layer, expert]` (0=CPU, 1=GPU)

### Prefetch Strategy

**Collection Mode** (first run):
- Records expert usage per token position
- Pattern file: `expert_usage_patterns_qwen.json`
- Structure: `token_pos → layer_id → [expert_ids]`

**Prediction Mode** (subsequent runs):
- Predicts needed experts based on token position
- Prefetches top-k experts for layer N+2 during layer N execution
- Async transfer allows GPU to continue computing
- Cache lookup before on-demand loading

### Timing and Hit Rate Tracking

Both implementations track execution by phase:
- **Prefill phase**: `sequence_length > 1` (initial prompt processing)
- **Decode phase**: `sequence_length == 1` (autoregressive generation)
- Returned from `generate()`: `(prefill_time, decode_time, prefill_hit_rate, decode_hit_rate)`

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

## ⚠️ Critical Configuration Notes

### Testing Configuration
All testing must use identical settings:
- `cpu_offload=0`
- `max_experts_gpu=0`
- `beam_width=1`
- `eos_token_id=None` (forces exact token count for consistent benchmarking)

### Memory Management
- Use `torch.cuda.empty_cache()` between model loads
- Pinned CPU memory required for async transfers
- Expert parameters pinned during initialization

### Pattern Files
- Baseline: No pattern file needed
- Prefetch: Uses `expert_usage_patterns_qwen.json`
- First run (collection mode) generates pattern file
- Subsequent runs (prediction mode) use learned patterns

### Key Implementation Details
- **Dtype consistency**: Expert buffers created with `dtype=self.dtype` (bfloat16)
- **Precision fix**: Explicit dtype conversion before accumulation prevents precision loss
- **Async synchronization**: Event synchronization required before using prefetched experts (critical for correctness)
- **MoE semantics**: One-hot expert masking, index_add accumulation, shared expert
- **GPU-resident layers**: Layers 0-1 kept on GPU permanently for efficiency

### Fiddler Cost Model Parameters
- **Qwen optimal values**: `latency_cpu=0.1`, `latency_gpu=10.0`
- These values are hardware-specific and may need tuning
- Use `profile_qwen_expert_costs.py` to measure actual costs

## 🚀 Usage Instructions

### Running Benchmarks

```bash
# Full benchmark: 0-16 experts with prefill/decode analysis
python benchmark_prefetch_configs.py

# Results saved to: prefetch_benchmark_YYYYMMDD_HHMMSS/
# - benchmark_results.csv: Detailed timing data
# - benchmark_summary.json: Full results with baseline
# - prefetch_speedup_analysis.png: Visualization plots
```

### Quick Testing

```bash
# Test baseline
python quick_test.py FiddlerQwen

# Test prefetch (first run = collection, second run = prediction)
python quick_test.py FiddlerQwenWithPrefetch
```

### Profiling

```bash
# Generate Nsight profiles
python profile_qwen_prefetch.py

# Analyze with Nsight Systems GUI
nsight-sys qwen_prefetch_profile_*/qwen_prefetch_prediction.nsys-rep

# Command-line stats
nsys stats --report nvtx_sum qwen_prefetch_profile_*/qwen_prefetch_prediction.nsys-rep
```

## 📁 Key Files

### Core Implementation
- **`src/fiddler/qwen.py`**: Baseline with on-demand expert loading
- **`src/fiddler/qwen_with_prefetch.py`**: Unified prefetch + Fiddler implementation
  - Added async synchronization (lines 318-324) for correctness with 4+ experts
  - Fiddler CPU offloading with greedy partitioning
  - Separate prefill/decode hit rate tracking

### Benchmarking & Tools
- **`benchmark_prefetch_configs.py`**: Comprehensive benchmark suite
- **`quick_test.py`**: Fast validation tool
- **`profile_qwen_expert_costs.py`**: Cost parameter profiling script

### Data Files
- **`expert_usage_patterns_qwen.json`**: Expert usage patterns for prefetching

### Latest Results
- **`prefetch_benchmark_20251009_182449/`**: Final benchmark results
  - All 4 configurations tested
  - Complete CSV/JSON results
  - Visualization plots

## 🎯 Recommendations

**For Production**:
- Use **Fiddler CPU offloading alone**: 2.17x speedup
- Configuration: `enable_cpu_offload=True, num_experts_to_prefetch=0`
- Cost parameters: `latency_cpu=0.1, latency_gpu=10.0`

**For Different Hardware**:
- Profile actual costs using `profile_qwen_expert_costs.py`
- Adjust `latency_cpu` and `latency_gpu` accordingly
- Test all 4 configurations to find optimal approach

**Model-Specific Insight**:
- Qwen's small experts favor CPU execution over GPU
- Larger expert models (e.g., Mixtral) may benefit more from prefetching
- Always validate correctness with `quick_test.py` before benchmarking
