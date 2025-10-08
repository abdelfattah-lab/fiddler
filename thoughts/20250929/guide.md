# Fiddler MoE Optimization Project - Agent Guide

## Guidelines

Always update guide.md to prepare it for another agent to look at it and understand the full state of the system and keep it concise. At the end of that, add all files changed (that are relevant) including guide.md to git and suggest a commit message but let me do the git commit.

## Current Goal

✅ **COMPLETED** (October 7, 2025): Fixed the correctness bug with 4+ experts by adding proper async synchronization. All configurations 0-16 now work correctly with excellent speedup (up to 1.428x decode speedup) and high hit rates (100% decode hit rate for configs 4+).

## Completed Tasks

✅ COMPLETED: **Fixed correctness bug with 4+ experts** (October 7, 2025)
- **Root cause**: Missing synchronization for async CUDA transfers
  - When prefetching 4+ experts, all needed experts were loaded asynchronously
  - GPU accessed expert parameters before async transfers completed
  - Result: Incorrect computation and wrong outputs
- **Solution**: Added explicit event synchronization in src/fiddler/qwen_with_prefetch.py:318-324
  - Synchronize with prefetch stream before using prefetched expert
  - Maintains parallelism: async transfer happens during previous layer's compute
  - Only sync when actually needed (just-in-time synchronization)
- **Impact**: All configs 0-16 now produce correct outputs
- **Performance**: Config 4 achieves 1.428x decode speedup with 100% decode hit rate
- **Validation**: Full correctness testing and benchmarking completed

✅ COMPLETED: **Root cause analysis of hit rate plateau** (October 7, 2025)
- Identified TWO critical issues causing low hit rates:
  1. **Incomplete pattern file**: Pattern file only contained 3 token positions instead of 21
     - Root cause: Early EOS token generation stopped collection after ~14 tokens
     - Fix: Set `eos_token_id=None` in both qwen.py and qwen_with_prefetch.py to force exact token count
     - Created `regenerate_patterns.py` script to collect complete patterns (21 positions)
  2. **Correctness bug with 4+ experts**: Model generates INCORRECT outputs when prefetching 4+ experts
     - Configs 0-3: Generate correct output matching baseline ✓
     - Configs 4-16: Generate wrong output (e.g., "ParisHuman Resource Management..." instead of "Paris\nLondon\nBerlin\nRome...") ✗
     - This explains hit rate plateau: patterns are correct, but model with 4+ experts uses different experts due to wrong generation
     - Hypothesis: Stale cache entries, insufficient async transfer synchronization, or buffer management issues
     - **STATUS**: Bug identified but not yet fixed - requires deeper investigation

✅ COMPLETED: **Hit rate analysis across 0-16 experts** (October 7, 2025)
- Fresh pattern collection run completed
- Comprehensive benchmark with 17 configurations (0-16 experts)
- **Key finding**: Decode hit rate plateaus at ~55%, not 100% as initially expected
- Prefill hit rate grows from 9% to 84% with more experts
- Decode hit rate remains stable at 55% regardless of expert count (4-16 experts)
- Optimal configuration: 5 experts (2.945x speedup, 55.2% decode hit rate)

✅ COMPLETED: Separate prefill and decode hit rate tracking has been implemented. Both `qwen.py` and `qwen_with_prefetch.py` now return separate hit rates for prefill and decode phases.

## 🎯 Project Status

**STATUS**: ✅✅✅ Qwen MoE prefetching system fully operational with EXCELLENT performance

The Fiddler MoE optimization project has successfully implemented and benchmarked expert prefetching for Qwen1.5-MoE-A2.7B. **Critical async synchronization bug fixed** - all configurations now produce correct outputs and achieve up to 1.428x decode speedup with 100% decode hit rate.

**Primary Branch**: `predictor_vs_fiddler`

**Architecture**: CPU-to-GPU expert management with pattern-based prefetching
- Model layers on GPU, experts on CPU (pinned memory)
- Single expert buffer for baseline on-demand loading
- Dual buffer system (A/B) for prefetch with async transfers
- Layer+2 prefetch prediction based on learned token-position patterns

## 📊 Latest Benchmark Results

**Benchmark**: `prefetch_benchmark_20251007_233447/` ✅ **All configs working correctly!**

### Baseline Performance (No Prefetch)
- **Prefill**: 0.404s
- **Decode**: 2.052s
- **Total**: 2.600s

### ✅ ALL Configurations Working (Correctness Verified)

**Best Configurations by Total Speedup:**

1. **8 experts**: 1.326x total speedup ⭐⭐⭐
   - Prefill: 1.140x (0.354s) | Hit rate: 55.3%
   - Decode: 1.425x (1.440s) | Hit rate: 100.0%
   - **Perfect decode hit rate!**

2. **7 experts**: 1.325x total speedup ⭐⭐
   - Prefill: 1.139x (0.354s) | Hit rate: 49.6%
   - Decode: 1.419x (1.446s) | Hit rate: 100.0%

3. **5 experts**: 1.324x total speedup ⭐
   - Prefill: 1.134x (0.356s) | Hit rate: 38.2%
   - Decode: 1.423x (1.442s) | Hit rate: 100.0%

4. **4 experts**: 1.323x total speedup (BEST decode speedup!)
   - Prefill: 1.131x (0.357s) | Hit rate: 33.1%
   - Decode: 1.428x (1.437s) | Hit rate: 100.0%
   - **Previously broken, now works perfectly!**

**All other configs (0-3, 6, 9-16)** also work correctly with speedups ranging from 1.07x to 1.32x.

### Key Insights
- ✅ **All configs 0-16 produce correct outputs** - correctness bug fully fixed!
- ✅ **100% decode hit rate** for configs 4-16 (up from 26-27% when broken)
- ✅ **1.428x decode speedup** achieved with config 4 (up from 1.285x with config 3)
- ✅ **Async synchronization fix** unlocked full performance potential
- **Performance sweet spot**: 4-8 experts for best total speedup
- **Decode hit rate scaling**: Configs 4+ achieve perfect 100% decode hit rate
- **Prefill hit rate scaling**: Improves from 33% to 86% as expert count increases

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

### Core Implementation (October 7, 2025 - Correctness Fix)
- `src/fiddler/qwen_with_prefetch.py` - **CRITICAL BUG FIX: Added async synchronization**
  - Lines 318-324: Added event synchronization before using prefetched experts
  - Ensures async transfers complete before GPU accesses expert parameters
  - Fixes correctness bug with 4+ experts while maintaining parallelism
  - Result: All configs 0-16 now work correctly with 100% decode hit rate

### Core Implementation (October 7, 2025 - EOS Fix)
- `src/fiddler/qwen.py` - Baseline with prefill/decode timing and hit rates
  - Added `eos_token_id=None` to force exact token count for consistent benchmarking
  - Returns separate hit rates for prefill and decode (same value for both in baseline)
- `src/fiddler/qwen_with_prefetch.py` - Prefetch with prefill/decode timing and hit rates
  - Added `eos_token_id=None` to force exact token count
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

### Debugging Scripts (October 7, 2025)
- `regenerate_patterns.py` - Script to regenerate complete expert usage patterns
  - Forces collection of all 21 token positions (1 prefill + 20 decode)
  - Used to fix incomplete pattern file issue
  - Run with: `python regenerate_patterns.py`
- `debug_pattern_collection.py` - Debug script for understanding pattern collection
- `test_correctness.py` - Test script to verify output correctness across configs

### Results
- `expert_usage_patterns_qwen.json` - Complete pattern file with 21 token positions ✓
- `prefetch_benchmark_20251007_233447/` - **LATEST: All configs working!** ✅✅✅
  - All configs 0-16 produce correct outputs
  - Best config: 8 experts with 1.326x speedup and 100% decode hit rate
  - Config 4: 1.428x decode speedup (best decode performance)
  - Full CSV and JSON results + visualization plots included
- `prefetch_benchmark_20251007_022723/` - Previous benchmark (with correctness bug)
  - Showed correctness bug for configs 4+
  - Best working config: 3 experts with 77% decode hit rate
- `prefetch_benchmark_20251007_020140/` - Earlier benchmark (incomplete patterns)
  - Had hit rate plateau at 55% due to incomplete pattern file

---

**✅ Status**: All issues RESOLVED (October 7, 2025). System fully operational!
1. ✅ **FIXED**: Incomplete pattern file (only 3 positions) - fixed by forcing exact token count with `eos_token_id=None`
2. ✅ **FIXED**: Correctness issue with 4+ experts - fixed by adding async synchronization (qwen_with_prefetch.py:318-324)

**Best configurations**:
- **Overall speedup**: 8 experts achieves 1.326x total speedup with 100% decode hit rate
- **Decode speedup**: 4 experts achieves 1.428x decode speedup with 100% decode hit rate
- **All configs 0-16 produce correct outputs** and achieve good performance (1.07x-1.33x speedup)
