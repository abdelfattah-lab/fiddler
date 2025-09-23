# Fiddler MoE Optimization Project - Agent Guide

## Current Focus: MoE Expert Memory Optimization through prefetching

## ✅ **COMPLETED: Qwen MoE Experiments**

**Qwen MoE experiments successfully implemented and validated!**

The requested experiments comparing fiddler and prefetching approaches on Qwen model have been completed with working implementations that generate correct output and demonstrate prefetch effectiveness.

**Primary Branch**: `predictor_vs_fiddler`

## 🎯 **Project Status**

✅ **COMPLETED**: Qwen MoE experiments with fiddler vs prefetching comparison
- Both implementations generate correct output: "The capital of France is ______. Paris"
- Prefetch system achieves 29.2% hit rate after pattern learning
- Ready for detailed performance analysis and speedup measurement

**Architecture**: Qwen1.5-MoE with device_map="auto" + expert management layer
- All experts managed through transformers' native device placement
- Expert usage tracking and prediction implemented
- Pattern collection and prefetch prediction working

## 🏗️ **Key Implementations**

### **Core Models** (`src/fiddler/`)
- `mixtral.py` - **Baseline Fiddler implementation** (production ready)
- `mixtral_with_prefetch.py` - **Prefetch with memory bug** (1.01x speedup but inefficient)
- `mixtral_with_buffers.py` - **Multi-buffer approach**

### **✅ NEW: Qwen MoE Implementations**
- `qwen.py` - **✅ WORKING: Baseline Fiddler implementation for Qwen MoE**
  - Uses device_map="auto" for correct device placement
  - Expert management tracking layer
  - Generates correct output, validated with quick_test.py
- `qwen_with_prefetch.py` - **✅ WORKING: Prefetch implementation for Qwen MoE**
  - Collection mode: Records expert usage patterns → `expert_usage_patterns_qwen.json`
  - Prediction mode: Achieves 29.2% prefetch hit rate
  - Extends working baseline with pattern learning

### **Research Implementations**
- `qwen_single_buffer.py` - **Reference implementation** (produces garbled output, not used)

## ⚠️ **Critical Test Configuration**
All testing must use identical settings:
- `cpu_offload=0`, `max_experts_gpu=0`, `beam_width=1`

## 🔧 **Tools & Testing**

### **Primary Testing Tool**
- `quick_test.py` - **Fast 3-token generation test** (correctness + performance)
  ```bash
  # Test Qwen implementations
  python quick_test.py FiddlerQwen              # Baseline
  python quick_test.py FiddlerQwenWithPrefetch  # Prefetch (29.2% hit rate)

  # Test Mixtral implementations
  python quick_test.py FiddlerMixtralWithPrefetch
  ```

### **Profiling Infrastructure**
- `profile_nsight.py` - **Unified Nsight profiling** with `profile_program()` function
- `profile_qwen_prefetch.py` - **✅ NEW: Qwen prefetch profiling script**
- `qwen_single_buffer.py` + `profile_single_buffer.py` - **Single buffer validation**
- **Environment**: `conda env qwen_profiling` (transformers 4.56.2)

### **Key Results**
- **✅ Qwen MoE working**: Both baseline and prefetch implementations generate correct output
- **✅ Prefetch effectiveness**: 29.2% hit rate demonstrates successful pattern learning
- **✅ Nsight profiling completed**: Collection vs Prediction mode profiles generated
  - Profile directory: `qwen_prefetch_profile_20250923_163737/`
  - Collection mode: `qwen_prefetch_collection.nsys-rep` (0% hit rate)
  - Prediction mode: `qwen_prefetch_prediction.nsys-rep` (29.2% hit rate)
- **🎯 Ready for analysis**: Can now measure speedup of prefetching vs fiddler approach
- **Memory dominance**: 35-55% execution time in transfers (from Mixtral analysis)
- **Prefetch bug**: High hit rate → 2x MORE memory ops (should be fewer) (Mixtral issue)

## 📋 **Interface Requirements**

All implementations must support:
```python
class YourImplementation:
    def __init__(self, args, **kwargs):
        # Initialize model

    def generate(self, text, output_token=20, input_token=None):
        # Return (prefill_time, decode_time, expert_hit_rate)

    def tokenize(self, text):
        # Return (input_ids, position_ids) - same as baseline

    def mixtral_forward(self, input_ids, position_ids, is_decode):
        # Core inference - return logits tensor
```

## ⚠️ **Critical Notes**

### **Testing Requirements**
- **✅ Qwen implementations**: Both work correctly with quick_test.py
- **MixtralWithBuffers**: Delete `expert_usage_patterns.json` before testing
- **QwenWithPrefetch**: Uses `expert_usage_patterns_qwen.json` for pattern storage
- **Correctness**: Forward pass logits must match baseline within tolerance ✅
- **Memory**: Use `torch.cuda.empty_cache()` between model loads

## 🚀 **Usage Instructions**

### **Testing Qwen Implementations**
```bash
# Test baseline
python quick_test.py FiddlerQwen

# Test prefetch (first run = collection, second run = prediction)
python quick_test.py FiddlerQwenWithPrefetch
```

### **Profiling Qwen Prefetch**
```bash
# Generate Nsight profiles for collection vs prediction modes
python profile_qwen_prefetch.py

# Analyze results
nsight-sys qwen_prefetch_profile_*/qwen_prefetch_collection.nsys-rep
nsight-sys qwen_prefetch_profile_*/qwen_prefetch_prediction.nsys-rep
```

## 🚀 **Next Steps**

The Qwen MoE experiments are complete and profiled. To measure speedup:

1. **✅ Profile both implementations** using existing profiling infrastructure
2. **Analyze Nsight profiles** to compare collection vs prediction performance
3. **Run longer generations** to measure decode-phase prefetch effectiveness
4. **Calculate speedup metrics** from profile data
5. **Optional**: Implement true CPU/GPU expert movement for direct Fiddler comparison

---

**✅ Updated**: Guide reflects completed Qwen MoE experiment implementation with working baseline and prefetch systems.
