# Fiddler MoE Optimization Project - Agent Guide

## Guidelines

Always update guide.md to prepare it for another agent to look at it and understand the full state of the system and keep it concise. At the end of that, add all files changed (that are relevant) including guide.md to git and suggest a commit message but let me do the git commit.

## Current Status (2025-09-30)

**Implementation**: ✅ Async prefetching fixes applied successfully in `src/fiddler/qwen_with_prefetch.py`

**Problem**: Despite fixing synchronization issues, memory transfers still don't overlap with compute in profiling.

## Next Goal

**Keep first 2 MoE layers permanently on GPU** - Since these layers are never prefetched (we predict layer+2), keeping layers 0-1 on GPU may improve performance without affecting prefetch logic.

## Recent Work

✅ **Fixed async prefetching implementation** (2025-09-30):
1. Removed forced event wait blocking parallelism (line 251-252)
2. Moved CPU operations outside stream context (line 410-414)
3. Moved prefetch trigger after expert processing (line 290-298)
4. Removed hardcoded layer restrictions (line 340)

**Results**:
- ✅ Correct output: "The capital of France is ______.\nParis"
- ✅ Hit rate: 11.8% (pattern learning phase)
- ✅ Profile: `qwen_prefetch_profile_20250930_133058/`
- ❌ Still no compute/memory overlap observed in Nsight

## Key Files

- **Main implementation**: `src/fiddler/qwen_with_prefetch.py` - Async prefetching with dual buffers
- **Baseline**: `src/fiddler/qwen.py` - CPU-to-GPU on-demand loading
- **Test script**: `quick_test.py` - Fast correctness/performance validation

## Implementation Architecture

**Qwen Model Setup**:
- Base model + non-expert layers: GPU
- All experts (60 experts × 24 MoE layers): CPU
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
