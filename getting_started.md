# Getting Started with Fiddler MoE Optimization

## Overview

Fiddler is a MoE (Mixture of Experts) inference optimization system that uses intelligent prefetching to accelerate expert loading from CPU to GPU. This guide will help you set up and run the repository on a new machine.

## Quick Summary

**Achievement**: 3.21x speedup vs pinned baseline using 7-expert prefetching (0.679s vs 2.182s)

**Models Supported**:
- Qwen2.5-7B-Instruct (primary - 24 MoE layers, 60 experts per layer)
- Mixtral-8x7B-v0.1 (legacy)

**Key Implementations**:
- `src/fiddler/qwen.py` - Baseline with on-demand CPU→GPU expert loading
- `src/fiddler/qwen_with_prefetch.py` - Prefetch system with configurable expert count (0-16)

## System Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on CUDA 12.x)
- At least 24GB GPU memory (for Qwen2.5-7B)
- 32GB+ system RAM recommended

### Software
- Linux (tested on Ubuntu with kernel 6.8.0)
- Python 3.10+
- NVIDIA drivers with CUDA 12.x support
- NVIDIA Nsight Systems (optional - for profiling)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fiddler
```

### 2. Create Python Environment

**Option A: Using conda (recommended)**
```bash
conda create -n fiddler python=3.10
conda activate fiddler
```

**Option B: Using venv**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch==2.8.0` - PyTorch with CUDA support
- `transformers==4.56.2` - HuggingFace transformers
- `accelerate==1.10.1` - For model parallelism
- `nvtx==0.2.13` - For profiling markers (optional)
- `matplotlib==3.10.6` - For plotting benchmarks (optional)

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.8.0
CUDA: True
```

## Quick Start

### 1. Basic Test

Test the baseline and prefetch implementations:

```bash
# Test baseline (on-demand loading)
python quick_test.py FiddlerQwen

# Test prefetch - first run collects patterns, second run uses them
python quick_test.py FiddlerQwenWithPrefetch
python quick_test.py FiddlerQwenWithPrefetch  # Run twice
```

**Expected output:**
- First run: "The capital of France is ______.\nParis" (or similar)
- Prefetch should show ~3x speedup on second run

### 2. Benchmark Different Configurations

Run comprehensive benchmark testing prefetch with 0-16 experts:

```bash
python benchmark_prefetch_configs.py
```

This will:
- Test all prefetch configurations (0-16 experts per layer)
- Run baseline comparison
- Generate performance plots in `prefetch_benchmark_<timestamp>/`
- Save results to CSV and JSON
- Show optimal configuration (typically 7 experts)

**Expected results:**
- Baseline: ~2.2s
- Best prefetch (7 experts): ~0.68s (3.21x speedup, 52% hit rate)

### 3. Profile with Nsight Systems (Optional)

For detailed performance analysis:

```bash
# Create profile directory
timestamp=$(date +%Y%m%d_%H%M%S)
profile_dir="qwen_profile_${timestamp}"
mkdir -p "${profile_dir}"

# Run profiling
nsys profile \
  --output="${profile_dir}/profile" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  python quick_test.py FiddlerQwenWithPrefetch

# View in GUI
nsight-sys "${profile_dir}/profile.nsys-rep"
```

## Architecture Overview

### Model Setup
- **Base model + non-expert layers**: GPU
- **MoE layers 0-1 experts**: GPU (permanently resident, ~120 experts)
- **MoE layers 2-23 experts**: CPU (60 experts × 22 layers = 1,320 experts)
- **Expert buffers**: GPU (dual buffer A/B for alternating layers)
- **Prefetch strategy**: Layer N predicts experts for Layer N+2

### Prefetching Mechanism
1. **Dual buffers**: Buffer A (even layers), Buffer B (odd layers)
2. **Async loading**: Separate CUDA stream for non-blocking transfers
3. **Pattern learning**:
   - First run: Collect expert usage patterns → `expert_usage_patterns_qwen.json`
   - Second run: Use patterns to prefetch experts
4. **Hit rate tracking**: Monitor cache effectiveness with NVTX markers

## Key Files

### Main Implementations
- `src/fiddler/qwen.py` - Baseline implementation with pinned memory
- `src/fiddler/qwen_with_prefetch.py` - Prefetch implementation extending baseline
- `quick_test.py` - Fast correctness/performance validation
- `benchmark_prefetch_configs.py` - Comprehensive benchmarking script

### Test Scripts
- `quick_test.py` - Quick validation (2-3 tokens)
- `benchmark_prefetch_configs.py` - Full benchmark (0-16 expert configs)
- `benchmark_batch_size_fiddler.py` - Batch size analysis
- `profile_7_experts.py` - Profiling for optimal config

### Documentation
- `thoughts/20250922/guide.md` - Detailed agent guide and history
- `getting_started.md` - This file
- `README.md` - Project overview (if exists)

## Usage Examples

### Custom Prefetch Configuration

```python
import sys
sys.path.append('src')

from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch

class Args:
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1

args = Args()

# Create model with custom prefetch config (0-16 experts)
model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=7)

# Generate text
prefill_time, decode_time, hit_rate = model.generate(
    "The capital of France is",
    output_token=20
)

print(f"Hit rate: {hit_rate:.1%}")
print(f"Output: {model.last_generated_text}")
```

### Baseline (No Prefetch)

```python
from fiddler.qwen import FiddlerQwen

args = Args()
model = FiddlerQwen(args)

prefill_time, decode_time, hit_rate = model.generate(
    "The capital of France is",
    output_token=20
)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size (default is 1)
- Use smaller model or fewer prefetch experts
- Check GPU memory: `nvidia-smi`

### Slow First Run
- First run collects patterns (expected to be slow)
- Run twice to see prefetch benefits
- Pattern file: `expert_usage_patterns_qwen.json`

### Import Errors
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or modify sys.path in script
import sys
sys.path.insert(0, 'src')
```

### NVTX Not Found (Optional)
```bash
pip install nvtx
```
NVTX is only needed for profiling markers - core functionality works without it.

## Expected Performance

### Qwen2.5-7B-Instruct (20 tokens)

| Configuration | Time | Speedup | Hit Rate | Memory |
|---------------|------|---------|----------|--------|
| Baseline (no prefetch) | 2.182s | 1.00x | 0% | Base |
| Prefetch (0 experts) | 1.857s | 1.18x | 0% | Base |
| Prefetch (4 experts) | 0.683s | 3.19x | 42.6% | +Small |
| **Prefetch (7 experts)** | **0.679s** | **3.21x** | **52.1%** | +Medium |
| Prefetch (9 experts) | 0.684s | 3.19x | 57.5% | +Medium |
| Prefetch (16 experts) | 0.709s | 3.08x | 70.5% | +Large |

**Optimal**: 7 experts provides best speedup-to-memory ratio

## Next Steps

1. **Run benchmarks** to reproduce results
2. **Profile your workload** with Nsight Systems
3. **Tune prefetch config** for your specific use case
4. **Explore different prompts** to understand hit rate variations
5. **Review** `thoughts/20250922/guide.md` for detailed implementation notes

## Contributing

When making changes:
1. Test with `quick_test.py` first
2. Run full benchmark to validate performance
3. Update `thoughts/20250922/guide.md` with findings
4. Profile with Nsight Systems if changing core logic

## References

- **Project docs**: `thoughts/20250922/guide.md`
- **Nsight Systems**: [NVIDIA Developer](https://developer.nvidia.com/nsight-systems)
- **Qwen Model**: [HuggingFace](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)
- **Mixtral Model**: [HuggingFace](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

## Support

For issues or questions:
1. Check `thoughts/20250922/guide.md` for known issues
2. Review profiling results in Nsight Systems
3. Verify CUDA/GPU setup with `nvidia-smi`
4. Check Python environment with `pip list`
