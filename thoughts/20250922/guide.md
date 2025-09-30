# Fiddler MoE Optimization Project - Agent Guide

## Guidelines

Always update guide.md to prepare it for another agent to look at it and understand the full state of the system and keep it concise. At the end of that, add all files changed (that are relevant) including guide.md to git and suggest a commit message but let me do the git commit.


## Current Goal

✅ **COMPLETED** - Added pinned memory to baseline Qwen for fair comparison

## Previous Goal

✅ **COMPLETED** - Added configurable prefetch system with benchmark script

## Current Status (2025-09-30 - Pinned Memory Baseline)

**New Feature**: ✅ Added pinned memory to baseline FiddlerQwen

**Key Achievement**: **1.33x speedup** from pinned memory alone (2.908s → 2.182s)

**Implementation**:
- Pinned all CPU expert parameters in `_pin_cpu_experts()` (4,320 parameters)
- Updated `_get_expert_for_execution()` to ensure pinned memory during state_dict transfers
- Same fix as prefetch: force re-pinning of state_dict() copies before GPU transfer

**Results**:
- Baseline without pinned: 2.908s (previous benchmark)
- Baseline with pinned: 2.182s (current benchmark)
- Speedup: **1.33x** (33.3% faster)
- Plot: `pinned_baseline_benchmark_20250930_185406/pinned_memory_speedup.png`

**Updated Prefetch Speedup** (vs new pinned baseline):
- Previous: 4.285x vs 2.908s baseline = 0.679s
- New: 0.679s / 2.182s = **3.21x vs pinned baseline** (still significant!)
- This shows prefetch provides 3.21x speedup beyond what pinned memory alone achieves

## Previous Status (2025-09-30 - Configurable Prefetch)

**Feature**: ✅ Configurable prefetch with benchmark automation

**Implementation**:
- Added `num_experts_to_prefetch` parameter (0-16) to FiddlerQwenWithPrefetch
- Dynamic buffer allocation based on configuration
- Benchmark script tests all configurations and plots results

**Achievement**: ✅ 4.285x speedup vs old baseline (2.908s → 0.679s) with 52.1% hit rate (7 experts)

**Key Achievement**: Successfully enabled pinned memory transfers using async CUDA streams

## Investigation Complete

**ROOT CAUSE FOUND**: `state_dict()` was creating non-pinned copies of parameters, even though original params were pinned.

**FIX APPLIED**: Force re-pinning of all tensors from state_dict() before GPU transfer (lines 458-462 in qwen_with_prefetch.py)

**VERIFICATION**:
- Profile shows 1,533 Pinned H2D transfers (vs 0 before)
- cudaMemcpyAsync used on separate stream (stream 13)
- 1.28x measured speedup confirms async benefit

## Recent Work (2025-09-30)

✅ **Added configurable prefetch system** (2025-09-30):
1. **Parameter**: Added `num_experts_to_prefetch` to control prefetch behavior (0-16 experts)
2. **Dynamic buffers**: Buffer size now adapts to configuration (was fixed at 4)
3. **Benchmark automation**: Created `benchmark_prefetch_configs.py` script that:
   - Tests all configurations from 0 to 16 experts
   - Runs baseline comparison
   - Generates speedup and hit rate plots
   - Saves results to CSV and JSON
4. **Usage**: Model can now be instantiated with custom prefetch config:
   ```python
   model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=4)
   ```

**Purpose**: Systematically explore the prefetch configuration space to find optimal number of experts to prefetch per layer, balancing hit rate vs memory bandwidth.

**Benchmark Results** (from latest run):
- **Baseline**: 2.908s
- **Best configuration**: 7 experts → **4.285x speedup** (0.679s, 52.1% hit rate)
- Top 5 configurations:
  1. 7 experts: 4.285x speedup (52.1% hit rate)
  2. 4 experts: 4.263x speedup (42.6% hit rate)
  3. 5 experts: 4.257x speedup (45.5% hit rate)
  4. 9 experts: 4.255x speedup (57.5% hit rate)
  5. 6 experts: 4.247x speedup (48.6% hit rate)

**Key Findings**:
- Sweet spot around 4-9 experts per layer
- 7 experts provides best balance of hit rate and performance
- Beyond 10 experts, performance slightly degrades (diminishing returns)
- Results saved to: `prefetch_benchmark_20250930_174024/`

## Previous Work (2025-09-30)

✅ **Fixed async pinned memory transfers**:
1. **Problem**: Despite pinning parameters, Nsight showed "Pageable" transfers
2. **Root cause**: `state_dict()` creates copies that lose pinned memory property
3. **Solution**: Force re-pin all tensors from state_dict() before transfer (lines 458-462)
4. **Additional fixes**:
   - Wrapped transfers in `torch.no_grad()` to prevent autograd synchronization
   - Used `param.data.copy_(src, non_blocking=True)` for true async copy
   - Ensured dtype conversion preserves pinned memory property

**Results**:
- ✅ 1.28x speedup vs baseline (1.08s vs 0.82s for prediction run)
- ✅ Hit rate: 20.7% (unchanged)
- ✅ Pinned transfers: 1,533 pinned H2D operations (was 0 before fix)
- ✅ Async streams: Transfers on stream 13, compute on stream 7
- ✅ Output correct: "The capital of France is ______.\nParis"

**Key insight**: PyTorch's `state_dict()` method creates NEW tensor objects that don't inherit the pinned memory property from the original parameter, even though `is_pinned()` returns True. Must explicitly re-pin!

✅ **PyTorch overlap test** (2025-09-30):
1. Created `parallel_memory_compute_fixed_torch.py` - pure PyTorch version
2. Removed all Numba dependencies, uses only PyTorch operations
3. Uses dual buffers with separate transfer/compute streams
4. Event-based synchronization between streams (no blocking synchronize)
5. Profile generated: `pytorch_parallel_overlap.nsys-rep` (~226ms runtime)

**Purpose**: Determine if PyTorch can achieve compute/memory overlap at all, or if the framework adds hidden synchronization that prevents it.

✅ **GPU-resident layers optimization** (2025-09-30):
1. Keep first 2 MoE layers (0-1) permanently on GPU (line 142-153)
2. Direct GPU access for layers 0-1 bypassing prefetch logic (line 254-263)
3. Reasoning: Layer N predicts N+2, so layers 0-1 never get prefetched

**Results**:
- ✅ Correct output: "The capital of France is ______.\nParis"
- ✅ Hit rate improved: 11.8% → 20.7%
- ✅ GPU memory: ~8 experts × 2 layers = 16 experts on GPU (vs 4 in buffers)
- ⏱️ Speedup: 0.87x (still slower than baseline, needs profiling)

✅ **Fixed async prefetching implementation** (2025-09-30):
1. Removed forced event wait blocking parallelism (line 251-252)
2. Moved CPU operations outside stream context (line 410-414)
3. Moved prefetch trigger after expert processing (line 290-298)
4. Removed hardcoded layer restrictions (line 340)

**Previous results**:
- Hit rate: 11.8% (before GPU-resident optimization)
- Profile: `qwen_prefetch_profile_20250930_133058/`
- ❌ Still no compute/memory overlap observed in Nsight

## Key Files

- **Main implementation**: `src/fiddler/qwen_with_prefetch.py` - Async prefetching with dual buffers
- **Baseline**: `src/fiddler/qwen.py` - CPU-to-GPU on-demand loading
- **Test script**: `quick_test.py` - Fast correctness/performance validation

## Implementation Architecture

**Qwen Model Setup**:
- Base model + non-expert layers: GPU
- MoE layers 0-1 experts: GPU (permanently resident, ~16 experts)
- MoE layers 2-23 experts: CPU (60 experts × 22 layers)
- Expert buffers: GPU (dual buffer A/B for alternating layers)
- Prefetch strategy: Layer N predicts experts for Layer N+2

**Prefetching Mechanism**:
- Dual buffers: Buffer A (even layers), Buffer B (odd layers)
- Async loading: Separate CUDA stream (`self.prefetch_stream`)
- Pattern learning: Collection mode → `expert_usage_patterns_qwen.json` → Prediction mode

## Testing & Profiling

### Quick Testing
```bash
# Test baseline (on-demand loading)
python quick_test.py FiddlerQwen

# Test prefetch - first run collects patterns, second run uses them
python quick_test.py FiddlerQwenWithPrefetch

# Expected output: "The capital of France is ______.\nParis"

# Run benchmark across all prefetch configurations (0-16 experts)
python benchmark_prefetch_configs.py
# Creates prefetch_benchmark_<timestamp>/ directory with:
# - benchmark_results.csv (detailed results)
# - benchmark_summary.json (summary stats)
# - prefetch_speedup_analysis.png (speedup and hit rate plots)
```

### Profiling with Nsight Systems

#### Quick Profile of Best Configuration (7 experts)
```bash
# Profile the optimal configuration (4.285x speedup)
timestamp=$(date +%Y%m%d_%H%M%S)
profile_dir="qwen_7experts_profile_${timestamp}"
mkdir -p "${profile_dir}"

nsys profile \
  --output="${profile_dir}/qwen_7experts" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=none \
  python profile_7_experts.py

# Profile saved to: ${profile_dir}/qwen_7experts.nsys-rep
```

#### Profile Any Configuration
```bash
# Create a custom profiling script
# Set num_experts_to_prefetch to desired value (0-16)

timestamp=$(date +%Y%m%d_%H%M%S)
profile_dir="qwen_Nexperts_profile_${timestamp}"
mkdir -p "${profile_dir}"

nsys profile \
  --output="${profile_dir}/qwen_profile" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=none \
  python <your_script.py>

# Replace <your_script.py> with script that instantiates:
# model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=N)
```

#### View Profiles

**GUI (Recommended)**:
```bash
# View in Nsight Systems GUI
nsight-sys qwen_7experts_profile_*/qwen_7experts.nsys-rep

# Or use full path
nsight-sys /home/afa55/Projects/fiddler/qwen_7experts_profile_20250930_181810/qwen_7experts.nsys-rep
```

**CLI Analysis**:
```bash
# Memory transfer statistics
nsys stats --report cuda_gpu_mem_time_sum <profile.nsys-rep>

# NVTX marker statistics (shows prefetch hits/misses)
nsys stats --report nvtx_sum <profile.nsys-rep>

# CUDA kernel statistics
nsys stats --report cuda_gpu_kern_sum <profile.nsys-rep>

# Filter for specific markers
nsys stats --report nvtx_sum <profile.nsys-rep> | grep "ASYNC_EXPERT_LOAD"
nsys stats --report nvtx_sum <profile.nsys-rep> | grep "EXPERT_LOAD_ON_DEMAND"
nsys stats --report nvtx_sum <profile.nsys-rep> | grep "PREFETCH_HIT"
```

#### Profile Locations
All profiles are saved to directories with pattern:
- `qwen_7experts_profile_<timestamp>/` - 7-expert configuration profiles
- `qwen_prefetch_profile_<timestamp>/` - General prefetch profiles
- Each directory contains:
  - `*.nsys-rep` - Main Nsight Systems report file
  - `*.sqlite` - SQLite database (auto-generated from .nsys-rep)
  - `PROFILE_SUMMARY.md` - Human-readable summary (if created)

### NVTX Markers (visible in Nsight GUI)
- `PREFETCH_TRIGGER_AFTER_LAYER`: When prefetch is triggered for layer+2
- `ASYNC_SINGLE_EXPERT_LOAD`: Async expert loading operations
- `PREFETCH_HIT`: Using prefetched expert from cache
- `EXPERT_LOAD_ON_DEMAND`: Cache miss requiring synchronous load

## Known Issues

### No Compute/Memory Overlap
Despite async implementation with separate CUDA stream:
- Memory transfers don't overlap with compute in Nsight profiles
- Previous investigation found 6,541 sync calls vs 6 in simple parallel baseline
- PyTorch/transformers framework may add hidden synchronization points

### Investigation Artifacts
- `parallel_memory_compute_fixed.py` - Simple program showing good overlap (1.5x speedup)
- `true_parallel_baseline.nsys-rep` - Profile showing successful parallelism
- `test_minimal_pinned.py` - Verified pinned memory works with simple tensors
- `test_param_copy_pinned.py` - Verified `param.copy_()` uses pinned path
- `test_exact_copy_pattern.py` - Verified exact pattern from our code
- `diagnose_pinned_memory.py` - Diagnostic showing parameters ARE pinned
- Multiple `qwen_prefetch_profile_*/` directories - Various profiling attempts

## Files Changed

### Modified (Current Session):
1. **src/fiddler/qwen.py**:
   - Added `_pin_cpu_experts()` method (lines 138-155)
   - Pins all CPU expert parameters for faster transfers (4,320 parameters)
   - Updated `_get_expert_for_execution()` (lines 267-301)
   - Force re-pinning of state_dict() copies before GPU transfer
   - Ensures dtype conversion preserves pinned memory property

2. **thoughts/20250922/guide.md**:
   - Updated with pinned memory baseline results
   - Added 1.33x speedup comparison
   - Recalculated prefetch speedup vs new pinned baseline (3.21x)

### Created (Current Session):
1. **benchmark_pinned_baseline.py**:
   - Benchmark script for pinned memory baseline
   - Runs 5 iterations for stable measurements
   - Saves results to JSON

2. **plot_pinned_speedup.py**:
   - Visualization script for pinned vs non-pinned comparison
   - Generates bar charts showing 1.33x speedup

3. **pinned_baseline_benchmark_20250930_185406/**:
   - Benchmark results directory
   - Contains benchmark_results.json
   - Contains pinned_memory_speedup.png plot

### Modified (Previous Session):
1. **src/fiddler/qwen_with_prefetch.py**:
   - Added `num_experts_to_prefetch` parameter (lines 115-120)
   - Dynamic buffer allocation based on config (lines 163-186)
   - Updated async loading to load N experts (lines 431-500)
   - Updated sync loading to load N experts (lines 502-535)

### Created (Previous Session):
1. **benchmark_prefetch_configs.py**:
   - Automated benchmark script for all prefetch configurations
   - Tests 0-16 experts prefetch settings
   - Generates performance plots and CSV results
   - Found optimal configuration: 7 experts (4.285x speedup vs old baseline)

2. **profile_7_experts.py**:
   - Profiling script for optimal 7-expert configuration
   - Used with Nsight Systems for detailed analysis

3. **qwen_7experts_profile_20250930_181810/**:
   - Nsight Systems profile of best configuration
   - Contains .nsys-rep file and PROFILE_SUMMARY.md

### Created (diagnostics - can be deleted):
- `diagnose_pinned_memory.py`
- `test_minimal_pinned.py`
- `test_param_copy_pinned.py`
- `test_exact_copy_pattern.py`
- Various `.nsys-rep` profile files

## Suggested Commit Message

```
Add pinned memory to baseline Qwen for fair comparison

Added pinned memory support to the baseline FiddlerQwen implementation
to ensure fair comparison with the prefetch system. Pinned memory alone
provides 1.33x speedup over regular memory transfers.

Changes:
- Added _pin_cpu_experts() method to pin all CPU expert parameters
- Updated _get_expert_for_execution() to ensure pinned transfers:
  * Force re-pinning of state_dict() copies (same fix as prefetch)
  * Preserve pinned memory during dtype conversion
  * Handle 4,320 expert parameters across 24 MoE layers
- Created benchmark_pinned_baseline.py for performance measurement
- Created plot_pinned_speedup.py for visualization

Results:
- Baseline without pinned: 2.908s (previous)
- Baseline with pinned: 2.182s (current)
- Speedup from pinned memory: 1.33x (33.3% faster)

Updated prefetch speedup calculation:
- Previous: 4.285x vs 2.908s baseline
- Corrected: 3.21x vs 2.182s pinned baseline
- Prefetch still provides significant speedup beyond pinned memory

This establishes a fairer baseline for evaluating the prefetch system's
contribution, as both implementations now use pinned memory for CPU-GPU
transfers.

Files changed:
- src/fiddler/qwen.py
- benchmark_pinned_baseline.py (new)
- plot_pinned_speedup.py (new)
- thoughts/20250922/guide.md
```

## Historical Notes

<details>
<summary>Previous Work (click to expand)</summary>

### Completed Milestones
1. ✅ CPU-to-GPU expert management (baseline)
2. ✅ Dual buffer prefetch system (Buffer A/B)
3. ✅ Pattern learning and prediction
4. ✅ Async prefetch with separate stream
5. ✅ Output correctness verified
6. ✅ Removed synchronization barriers

### Architecture Evolution
- Started with on-demand loading (qwen.py)
- Added prefetch prediction (qwen_with_prefetch.py)
- Implemented dual buffers for alternating layers
- Added async loading with dedicated CUDA stream
- Multiple iterations to remove sync points

### Test Configuration
All tests use: `cpu_offload=0`, `max_experts_gpu=0`, `beam_width=1`

</details>
