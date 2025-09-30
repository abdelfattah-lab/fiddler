# Fiddler MoE Optimization Project - Agent Guide

## Guidelines

Always update guide.md to prepare it for another agent to look at it and understand the full state of the system and keep it concise. At the end of that, add all files changed (that are relevant) including guide.md to git and suggest a commit message but let me do the git commit.

## Current Status (2025-09-30)

**Implementation**: ✅ Pinned CPU memory for MoE experts in `src/fiddler/qwen_with_prefetch.py`

**Result**: ✅ 1.55x speedup vs baseline with 20.7% hit rate

## Current Goal

**COMPLETED**: Pinned CPU memory successfully improves performance. We now have overlapping memory transfers with compute.

Next steps: Further analysis of compute/memory overlap in Nsight GUI to understand remaining bottlenecks.

## Recent Work

✅ **Pinned CPU memory implementation** (2025-09-30):
1. Added `_pin_expert_memory()` method to pin all CPU expert parameters
2. Pinned 3,960 parameters (22 MoE layers × 60 experts × 3 params each)
3. Enabled async H2D transfers without blocking
4. Profile: `qwen_prefetch_profile_20250930_153222/`

**Results**:
- ✅ 1.55x speedup vs baseline (1.00s vs 1.55s)
- ✅ Hit rate: 20.7% (same as before, as expected)
- ✅ Memory transfers: 2.55s total H2D time (vs 2.36s without pinning)
- ✅ Net improvement: Transfers take longer but overlap with compute reduces wall time
- ✅ Output correct: "The capital of France is ______.\nParis"

**Key insight**: Total memory transfer time increased, but wall-clock time decreased - this confirms successful compute/memory overlap.

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
```

### Profiling with Nsight Systems
```bash
# Generate profiles (collection + prediction modes)
python profile_qwen_prefetch.py

# View in GUI
nsight-sys qwen_prefetch_profile_*/qwen_prefetch_collection.nsys-rep
nsight-sys qwen_prefetch_profile_*/qwen_prefetch_prediction.nsys-rep

# CLI analysis
nsys stats --report cuda_gpu_mem_time_sum <profile.nsys-rep>
nsys stats --report nvtx_sum <profile.nsys-rep>
```

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
- Multiple `qwen_prefetch_profile_*/` directories - Various profiling attempts

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
