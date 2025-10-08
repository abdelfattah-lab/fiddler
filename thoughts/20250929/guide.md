# Fiddler MoE Optimization Project - Agent Guide

## Guidelines

Always update guide.md to prepare it for another agent to look at it and understand the full state of the system and keep it concise. At the end of that, add all files changed (that are relevant) including guide.md to git and suggest a commit message but let me do the git commit.

## Current Goal

You're an expert researcher and software engineer and can get tasks done very efficiently. I want to fix the fact that 4+ experts prefetched have different output than other configurations in prefetch_qwen. I also want to resolve the prefetch hit rate. Ideally, you would be able to resolve those while maintaining the high speedup achieved at those configurations. Validate everything and do your best to get this right.

## Completed Steps

✅ COMPLETED: Separate prefill and decode hit rate tracking has been implemented. Both `qwen.py` and `qwen_with_prefetch.py` now return separate hit rates for prefill and decode phases.

## 🎯 Project Status

**STATUS**: ✅ Qwen MoE prefetching system fully operational with comprehensive performance analysis

The Fiddler MoE optimization project has successfully implemented and benchmarked expert prefetching for Qwen1.5-MoE-A2.7B. Both baseline and prefetch implementations generate correct outputs and achieve significant speedup through async expert loading.

**Primary Branch**: `predictor_vs_fiddler`

**Architecture**: CPU-to-GPU expert management with pattern-based prefetching
- Model layers on GPU, experts on CPU (pinned memory)
- Single expert buffer for baseline on-demand loading
- Dual buffer system (A/B) for prefetch with async transfers
- Layer+2 prefetch prediction based on learned token-position patterns

## 📊 Latest Benchmark Results (Prefill vs Decode Analysis)

**Benchmark**: `prefetch_benchmark_20251006_234734/`

### Baseline Performance (No Prefetch)
- **Prefill**: 0.404s
- **Decode**: 1.522s
- **Total**: 2.036s

### Optimal Configurations (0-16 Experts Tested)

**Best Overall Performance:**
1. **8 experts**: 2.889x total speedup
   - Prefill: 1.136x (0.356s)
   - Decode: 4.842x (0.314s)
   - Hit rate: 63.5%

2. **4 experts**: 2.885x total speedup
   - Prefill: 1.136x (0.356s)
   - Decode: 4.818x (0.316s)
   - Hit rate: 50.8%

3. **6 experts**: 2.875x total speedup
   - Prefill: 1.122x (0.360s)
   - Decode: 4.862x (0.313s)
   - Hit rate: 57.2%

### Key Insights
- **Prefetching primarily benefits decode phase**: Up to 4.86x speedup
- **Prefill phase sees modest gains**: ~13% improvement at best
- **Sweet spot**: 4-8 experts prefetched for optimal performance/memory tradeoff
- **Diminishing returns**: Beyond 8 experts, overhead outweighs benefits
- **Why it works**: Async memory transfers overlap with GPU compute, hiding transfer latency

### Speedup Mechanism
Transfer/compute overlap is the key to speedup:
- **Baseline (serial)**: GPU waits for each expert transfer
- **Prefetch (parallel)**: Experts loaded ahead of time, transfers hidden behind compute
- **Result**: Critical path dominated by compute, not memory transfers

## 🏗️ Core Implementations

### Model Files (`src/fiddler/`)

**`qwen.py`** - Baseline Fiddler implementation
- All model layers on GPU except experts (CPU)
- Single GPU expert buffer for on-demand loading
- Pinned CPU memory for async transfers
- Timing tracking for prefill/decode phases
- Generates correct output, validated

**`qwen_with_prefetch.py`** - Prefetch implementation
- Extends baseline with pattern-based prefetching
- Dual buffer system (A/B) for alternating layer prefetch
- Collection mode: Records expert usage patterns → `expert_usage_patterns_qwen.json`
- Prediction mode: Prefetches experts based on learned patterns
- Layer+2 prefetch triggering (prefetch for N+2 while executing N)
- Async CUDA stream for non-blocking transfers
- Timing tracking for prefill/decode phases

**Legacy Implementations:**
- `mixtral.py` - Original Mixtral baseline
- `mixtral_with_prefetch.py` - Mixtral prefetch version
- `qwen_single_buffer.py` - Research prototype (not used)

### Benchmarking Tools

**`benchmark_prefetch_configs.py`** - Comprehensive benchmark suite
- Tests prefetch with 0-16 experts
- Measures prefill vs decode speedup separately
- Generates performance plots and CSV results
- Tracks hit rates and timing statistics
- Latest run: `prefetch_benchmark_20251006_234734/`

**`quick_test.py`** - Fast validation tool
- 3-token generation test for correctness
- Quick performance check
- Usage:
  ```bash
  python quick_test.py FiddlerQwen              # Baseline
  python quick_test.py FiddlerQwenWithPrefetch  # Prefetch
  ```

### Profiling Tools

**`profile_qwen_prefetch.py`** - Nsight Systems profiling
- Generates collection vs prediction mode profiles
- NVTX markers for expert operations
- Memory transfer analysis
- Latest profiles: `qwen_prefetch_profile_*/`

## 🔧 Technical Architecture

### Buffer System

**Baseline (qwen.py):**
- Single GPU expert buffer (one expert at a time)
- Expert state dict copied from CPU to buffer on-demand
- Pinned CPU memory for faster transfers

**Prefetch (qwen_with_prefetch.py):**
- **Buffer A**: Even MoE layers (0, 2, 4, ...)
- **Buffer B**: Odd MoE layers (1, 3, 5, ...)
- Configurable slots per buffer (num_experts_to_prefetch)
- Async prefetch stream separate from compute stream
- Prefetch cache tracking: `{(layer_idx, expert_idx): buffer_slot}`

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

Both implementations track MoE layer execution time and hit rates by phase:
- **Prefill phase**: `sequence_length > 1` (initial prompt processing)
- **Decode phase**: `sequence_length == 1` (autoregressive generation)
- Times accumulated across all MoE layers
- Hit rates tracked separately for prefill and decode phases
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

## ⚠️ Critical Notes

### Testing Configuration
All testing must use identical settings:
- `cpu_offload=0`
- `max_experts_gpu=0`
- `beam_width=1`

### Memory Management
- Use `torch.cuda.empty_cache()` between model loads
- Pinned CPU memory required for async transfers
- Expert parameters pinned during initialization

### Pattern Files
- Baseline: No pattern file needed
- Prefetch: Uses `expert_usage_patterns_qwen.json`
- First run (collection mode) generates pattern file
- Subsequent runs (prediction mode) use learned patterns

### Correctness Verification
- Expected output: `"The capital of France is ______.\nParis\nLondon\nBerlin\nRome\n答案:\nA"`
- Both baseline and prefetch produce identical outputs
- Forward pass logits match within dtype tolerance

### Key Implementation Details
- **Dtype consistency**: Expert buffers created with `dtype=self.dtype` (bfloat16)
- **Precision fix**: Explicit dtype conversion before accumulation prevents precision loss
- **MoE semantics**: One-hot expert masking, index_add accumulation, shared expert
- **GPU-resident layers**: Layers 0-1 kept on GPU permanently for efficiency

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
nsight-sys qwen_prefetch_profile_*/qwen_prefetch_collection.nsys-rep
nsight-sys qwen_prefetch_profile_*/qwen_prefetch_prediction.nsys-rep

# Command-line stats
nsys stats --report nvtx_sum qwen_prefetch_profile_*/qwen_prefetch_prediction.nsys-rep
```

## 📁 Important Files Modified

### Core Implementation
- `src/fiddler/qwen.py` - Baseline with prefill/decode timing and hit rates
  - Returns separate hit rates for prefill and decode (same value for both in baseline)
- `src/fiddler/qwen_with_prefetch.py` - Prefetch with prefill/decode timing and hit rates
  - Added `get_prefill_hit_rate()` method to `PrefetchMetrics` class
  - Phase tracking in `_moe_forward_with_management()` using `metrics.set_phase()`
  - `generate()` returns separate `prefill_hit_rate` and `decode_hit_rate`
  - `get_prefetch_stats()` includes separate hit rate metrics

### Benchmarking
- `benchmark_prefetch_configs.py` - Updated for separate prefill/decode analysis
  - Modified `run_baseline()` to return separate hit rates
  - Modified `run_prefetch_config()` to return separate hit rates
  - Updated `plot_results()` with 3-panel visualization including separate hit rate plot
  - Enhanced summary output with phase-specific rankings and hit rates
  - CSV and JSON outputs include separate hit rate columns

### Results
- `prefetch_benchmark_20251006_234734/` - Latest benchmark results
  - `benchmark_results.csv` - Per-config timing data
  - `benchmark_summary.json` - Full results
  - `prefetch_speedup_analysis.png` - Visualization

---

**✅ Status**: Prefill/decode speedup and hit rate analysis completed. System achieves up to 4.86x decode speedup and 1.14x prefill speedup with 4-8 experts prefetched. Hit rates are now tracked separately for prefill and decode phases.
