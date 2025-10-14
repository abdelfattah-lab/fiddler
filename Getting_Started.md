# Getting Started with Fiddler MoE Optimization

This guide will help you set up the Fiddler MoE Optimization project and reproduce the latest benchmark results.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Model Download](#model-download)
4. [Quick Start & Verification](#quick-start--verification)
5. [Reproducing Latest Results](#reproducing-latest-results)
6. [Understanding the Results](#understanding-the-results)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on GPUs with 24GB+ VRAM)
- **RAM**: At least 32GB system RAM recommended
- **Storage**: ~20GB free space for model and checkpoints

### Software
- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for PyTorch 2.1.2)
- **Git**: For cloning the repository

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fiddler
```

### 2. Create Python Virtual Environment

It's strongly recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
# venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

Install core dependencies:

```bash
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `accelerate==0.26.1` - For efficient model loading
- `transformers==4.36.2` - HuggingFace transformers library

Install additional dependencies for benchmarking and data collection:

```bash
pip install matplotlib numpy h5py datasets
```

### 4. Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8
```

---

## Model Download

The project uses the **Qwen/Qwen1.5-MoE-A2.7B** model from HuggingFace. The model will be automatically downloaded on first run.

### Option 1: Automatic Download (Recommended)

The model will download automatically when you run any script that uses it. First run will take 5-10 minutes depending on your internet speed.

### Option 2: Pre-download Model

To download the model beforehand:

```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B'); \
AutoTokenizer.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B')"
```

### Model Size
- Model weights: ~5GB
- HuggingFace cache location: `~/.cache/huggingface/hub/`

---

## Quick Start & Verification

### 1. Verify Installation

Run a quick correctness test to ensure everything is working:

```bash
python quick_test.py FiddlerQwen
```

Expected output:
```
Testing FiddlerQwen with 3 tokens...
✅ Correctness test passed!
Generated text: The capital of France is Paris, which...
```

### 2. Test Learned Prefetch (requires predictor checkpoint)

If you have the trained predictor checkpoint:

```bash
python quick_test.py FiddlerQwenWithLearnedPrefetch
```

### 3. Run a Quick Single-Configuration Benchmark

Test baseline configuration:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from fiddler.qwen import FiddlerQwen

class Args:
    def __init__(self):
        self.model = 'Qwen/Qwen1.5-MoE-A2.7B'
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1

model = FiddlerQwen(Args())
prefill_t, decode_t, prefill_hr, decode_hr = model.generate(
    'The capital of France is',
    output_token=20
)
print(f'✅ Baseline working!')
print(f'Total time: {prefill_t + decode_t:.3f}s')
print(f'Decode time: {decode_t:.3f}s')
"
```

---

## Reproducing Latest Results

The latest results are from **Phase 5 Benchmark** with enhanced prediction accuracy tracking and correctness checking.

### Overview

The Phase 5 benchmark compares 4 configurations:
1. **Baseline**: On-demand expert loading (no optimization)
2. **Fiddler**: CPU offloading with dynamic partitioning
3. **Learned-Prefetch**: Learned attention-based expert prefetching
4. **Fiddler+Learned-Prefetch**: Combined approach (Fiddler + Learned predictor)

### Prerequisites

Ensure you have the trained predictor checkpoint:

```bash
ls predictor_checkpoints/best_model.pt
```

If missing, you'll need to either:
- Obtain the checkpoint from the original setup, or
- Train a new predictor (see [Training the Predictor](#training-the-predictor) below)

### Run Full Phase 5 Benchmark

**WARNING**: This benchmark takes 1-2 hours to complete!

```bash
python benchmark_prediction_methods.py
```

The benchmark will:
1. Run correctness check (ensures all configs produce identical outputs)
2. Test 4 configurations × 5 batch sizes (1, 2, 4, 8, 16) × 3 trials = 60 experiments
3. Generate comprehensive visualizations and analysis

### Output

Results are saved to a timestamped directory: `phase5_benchmark_YYYYMMDD_HHMMSS/`

Files created:
- `correctness_check.json` - Correctness validation results
- `benchmark_results.json` - Raw benchmark data
- `benchmark_results.csv` - CSV format for analysis
- `phase5_benchmark_results.png` - 12-subplot visualization
- `ANALYSIS.md` - Detailed text analysis with all metrics

### Quick Validation Run

To test the setup without waiting 2 hours, modify the script to run fewer trials:

```bash
# Edit benchmark_prediction_methods.py and change:
# batch_sizes = [1, 2, 4, 8, 16]  →  batch_sizes = [1, 2]
# num_trials = 3  →  num_trials = 1

python benchmark_prediction_methods.py
```

---

## Understanding the Results

### Key Metrics

The benchmark tracks several metrics separately for **prefill** and **decode** phases:

1. **Total Time**: Overall inference time (prefill + decode)
2. **Prefill Time**: Time to process the input prompt
3. **Decode Time**: Time to generate output tokens (this is what matters for latency)
4. **Prediction Accuracy**:
   - **Prefill**: How many prefetched experts were used during prompt processing
   - **Decode**: How many prefetched experts were used during generation
   - Higher = better predictions = less wasted memory transfers
5. **Tokens/Second**: Throughput metric (higher = better)
6. **Speedup vs Baseline**: How much faster compared to baseline (>1.0 = faster)

### Interpreting ANALYSIS.md

The analysis report has 6 sections:

**Section 0: Correctness Check**
- ✅ Status: PASSED means all configurations produce identical outputs (required!)
- ❌ Status: FAILED means outputs differ (results are invalid)

**Section 1: Overall Comparison**
- Shows which batch sizes Fiddler+Learned beats Fiddler
- Peak speedup indicates best performance gain

**Section 2: Prefill Phase Analysis**
- Prefill is typically fast (few tokens to process)
- Less critical than decode phase for latency

**Section 3: Decode Phase Analysis**
- **Most important section** - decode time dominates inference
- Look for speedup > 1.0x vs baseline

**Section 4: Prediction Accuracy Analysis**
- **Excellent**: >80% accuracy
- **Good**: 50-80% accuracy
- **Moderate**: 30-50% accuracy
- **Poor**: <30% accuracy

**Section 5: Comprehensive Summary**
- All metrics in one table for each configuration

### Expected Results (from guide.md)

From the completed Phase 5 benchmark (`phase5_benchmark_20251014_145424/`):

| Batch Size | Best Configuration | Key Metric |
|------------|-------------------|------------|
| 1-4 | Fiddler (CPU-only) | 21.9-28.5 tok/s |
| 8-16 | Fiddler+Learned | 28.0-41.7 tok/s |
| 16 | Fiddler+Learned | **1.129x speedup** over Fiddler |

**Key Finding**: Fiddler+Learned-Prefetch achieves significant speedups (1.13x) at large batch sizes (8-16) where it matters most.

---

## Training the Predictor

If you need to train the predictor from scratch (or retrain):

### Phase 1: Collect Training Data

Collect attention outputs and gating scores from WikiText-103:

```bash
python collect_training_data.py
```

- Duration: 5-7 hours
- Output: `predictor_training_data/` directory (~660MB)
- Target: 50,000 samples

Monitor progress:
```bash
python check_collection_progress.py
tail -f data_collection.log
```

### Phase 2: Train Predictor

Train the attention-based expert predictor:

```bash
python train_predictor.py
```

- Duration: 30-60 minutes (depends on GPU)
- Output: `predictor_checkpoints/best_model.pt`
- Target: >40% validation accuracy

Monitor training:
```bash
python monitor_training.py
tail -f training.log
```

### Phase 3: Validate Integration

Test the trained predictor:

```bash
python test_learned_prefetch.py
```

Expected:
- ✅ Correctness test passes
- ✅ Decode hit rate ≥40%

---

## Troubleshooting

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in benchmark (test BS=1, 2, 4 only)
2. Clear GPU cache between runs:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
3. Close other GPU-using processes:
   ```bash
   nvidia-smi  # Check GPU usage
   ```

### Model Download Fails

**Symptom**: Connection timeout or 404 errors

**Solutions**:
1. Check internet connection
2. Try with HuggingFace token (if model requires authentication):
   ```bash
   huggingface-cli login
   ```
3. Manual download from HuggingFace website and place in cache

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'fiddler'`

**Solution**: Ensure you're running from the project root:
```bash
cd /path/to/fiddler  # Project root
python benchmark_prediction_methods.py  # Not from subdirectory
```

### Predictor Checkpoint Missing

**Symptom**: `FileNotFoundError: predictor_checkpoints/best_model.pt`

**Solutions**:
1. If you have the checkpoint, ensure it's in the correct location
2. Train a new predictor (see [Training the Predictor](#training-the-predictor))
3. For testing without predictor, use baseline or Fiddler-only configs

### Benchmark Takes Too Long

**Solutions**:
1. Reduce trials: Change `num_trials = 3` to `num_trials = 1`
2. Test fewer batch sizes: Change `batch_sizes = [1, 2, 4, 8, 16]` to `[1, 4]`
3. Use quick validation script (create based on benchmark but with fewer iterations)

### Different Results Than Expected

**Check**:
1. Correctness test passes (Section 0 in ANALYSIS.md)
2. GPU thermal throttling (check `nvidia-smi`)
3. Other processes using GPU (affects timing measurements)
4. CUDA version matches (2.1.2 requires CUDA 11.8)

### Prediction Accuracy is 0%

**Symptom**: Decode hit rate shows 0% for learned prefetch configs

**Solutions**:
1. Verify predictor checkpoint loads correctly
2. Check that attention hook is registering (should see "Captured attention output" in logs)
3. Ensure batch generation is using `generate()` method (not manual forward passes)

---

## Additional Resources

### Project Structure

```
fiddler/
├── src/fiddler/              # Core implementation
│   ├── qwen.py               # Baseline (FiddlerQwen)
│   ├── qwen_with_prefetch.py # Pattern-based prefetch + Fiddler
│   └── qwen_with_learned_prefetch.py  # Learned prefetch + Fiddler
├── benchmark_prediction_methods.py    # Phase 5 benchmark
├── collect_training_data.py           # Data collection for predictor
├── train_predictor.py                 # Predictor training
├── test_learned_prefetch.py           # Integration tests
├── quick_test.py                      # Quick correctness validation
├── predictor_checkpoints/             # Trained predictor models
│   └── best_model.pt                  # Best checkpoint (46.8% accuracy)
├── predictor_training_data/           # Training data (HDF5 files)
├── phase5_benchmark_YYYYMMDD_HHMMSS/  # Benchmark results
└── thoughts/20251014/guide.md         # Development guide

```

### Configuration Reference

**Baseline** (no optimization):
```python
model = FiddlerQwen(args)  # Default: cpu_offload=0, max_experts_gpu=0
```

**Fiddler** (CPU offloading):
```python
args.cpu_offload = 1  # or args.use_fiddler_mode = True
model = FiddlerQwen(args)
```

**Learned-Prefetch** (attention-based prediction):
```python
model = FiddlerQwenWithLearnedPrefetch(
    args,
    num_experts_to_prefetch=8,
    enable_cpu_offload=False,
    predictor_path='predictor_checkpoints/best_model.pt'
)
```

**Fiddler+Learned-Prefetch** (combined):
```python
model = FiddlerQwenWithLearnedPrefetch(
    args,
    num_experts_to_prefetch=8,
    enable_cpu_offload=True,
    latency_cpu=0.1,
    latency_gpu=10.0,
    predictor_path='predictor_checkpoints/best_model.pt'
)
```

### Key Papers and References

- **Fiddler**: Efficient CPU offloading for MoE models
- **Qwen**: Mixture of Experts architecture from Alibaba
- **Attention-Based Prediction**: Uses layer 0 attention to predict expert usage in layers 2-23

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check `thoughts/20251014/guide.md` for detailed development history
2. Review `ANALYSIS.md` in any benchmark output directory for metric explanations
3. Examine correctness check results: `phase5_benchmark_*/correctness_check.json`
4. Check CUDA/GPU status: `nvidia-smi`

---

## Quick Reference Commands

```bash
# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
python quick_test.py FiddlerQwen

# Run full benchmark (1-2 hours)
python benchmark_prediction_methods.py

# Train predictor from scratch (6-8 hours total)
python collect_training_data.py  # 5-7 hours
python train_predictor.py        # 30-60 minutes
python test_learned_prefetch.py  # Validate

# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring
```

---

**Last Updated**: October 14, 2025
**Project Status**: Phase 5 Complete - Production Ready
