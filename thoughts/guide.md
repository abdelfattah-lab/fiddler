NL# Fiddler Mixtral Optimization Project - Development Template Guide

## 🎯 **PROJECT OVERVIEW**
This project optimizes Mixture of Experts (MoE) inference for Mixtral 8x7B models. The goal is to achieve speedups over the baseline expert caching approach through innovative optimization strategies.

## 🏗️ **CORE ARCHITECTURE**

### **Baseline Implementation**: `src/fiddler/mixtral.py`
The foundation for all optimization work - a proven implementation using static expert caching:

- **Static Expert Caching**: Pre-loads popular experts on GPU based on profiling data
- **On-demand Loading**: Uses expert placeholder with `load_state_dict()` for cache misses
- **Memory Efficient**: Only holds subset of experts in GPU memory
- **Dual Processing Modes**: Supports both CPU offloading (`cpu_offload=1`) and GPU-only (`cpu_offload=0`) operation

Note that in our tests, we're currently nerfing the baseline a bit by setting max_experts_gpu to 0 so that it has to load all experts when needed. This helps with our current experiments.

#### Key Components:
- `FiddlerMixtral` class - Main model wrapper
- `expert_placeholder` - GPU-resident template for loading CPU experts
- `popular_experts` - Profiled list of most frequently used experts by layer/index
- `set_expert_loc()` - Determines which experts to cache on GPU
- `mixtral_forward()` - Core inference with expert routing and execution

## 🧪 **TESTING INFRASTRUCTURE**

### **Quick Test Script**: `quick_test.py` ⭐ **ONLY TESTING TOOL NEEDED**
Ultra-fast testing for rapid development and iteration:

#### Features:
- **Lightning fast**: Generates only 2 tokens for speed
- **Correctness verification**: Compares output with baseline FiddlerMixtral
- **Performance measurement**: Reports timing and speedup
- **Dead simple**: Single command line argument

#### Usage:
```bash
python quick_test.py MixtralWithBuffers
python quick_test.py FiddlerMixtralWithPredictor
```

#### Example Output:
```
🚀 Quick test: MixtralWithBuffers
Loading MixtralWithBuffers...
MixtralWithBuffers: 'Paris, the' (1.234s)
Loading FiddlerMixtral...
FiddlerMixtral: 'Paris, the' (1.456s)
✅ MATCH! Speedup: 1.18x
```

## 📁 **PROJECT STRUCTURE**

```
src/fiddler/
├── mixtral.py                      # ⭐ BASELINE - Start here for all new work
├── __init__.py                     # Package initialization
└── [your_new_implementation.py]   # Your optimization attempts

Testing & Validation:
├── quick_test.py                         # ⭐ ONLY TESTING TOOL NEEDED
└── [other analysis scripts]

Documentation:
└── thoughts/20250915/template_guide.md   # This guide
```

## 🎯 **DEVELOPMENT WORKFLOW**

### **Step 1: Understand the Baseline**
1. Read and understand `src/fiddler/mixtral.py:12-688`
2. Focus on the `mixtral_forward()` method (`src/fiddler/mixtral.py:504-681`)
3. Understand expert routing and execution logic

### **Step 2: Create Your Implementation**
1. Copy or inherit from `FiddlerMixtral`
2. Implement your optimization strategy
3. Ensure the same interface: `__init__(args)` and `generate()` method
4. Return same format: `(prefill_time, decode_time, expert_hit_rate)`

### **Step 3: Test Your Implementation**
1. Run quick test: `python quick_test.py YourImplementation`
2. **Verify correctness**: Output must match baseline FiddlerMixtral
3. **Check performance**: Aim for speedup > 1.0x

## 🔧 **KEY SUCCESS PATTERNS**

### **Memory Management**
- Use `torch.cuda.empty_cache()` between model loads
- Implement proper GPU memory cleanup in destructors
- Test with limited GPU memory scenarios

### **Expert Handling**
- Preserve expert routing logic from baseline
- Maintain compatibility with beam search (`beam_width` parameter)
- Handle both CPU offloading modes if applicable

### **Testing Integration**
- Support the `max_experts_gpu` parameter for controlled testing
- Maintain `last_generated_text` attribute for output comparison
- Ensure deterministic behavior for reproducible testing

## ⚠️ **CRITICAL REQUIREMENTS**

### **Interface Compatibility**
Your implementation must support:
```python
class YourImplementation:
    def __init__(self, args, **kwargs):  # Accept additional parameters
        # Initialize your model

    def generate(self, text, output_token=20, input_token=None):
        # Return (prefill_time, decode_time, expert_hit_rate)

    def tokenize(self, text):
        # Return (input_ids, position_ids) - same as baseline

    def mixtral_forward(self, input_ids, position_ids, is_decode):
        # Core inference - return logits tensor
```

### **Testing Requirements**
- **Correctness**: Forward pass logits must match FiddlerMixtral within tolerance
- **Generation**: Must produce valid text outputs
- **Performance**: Should maintain or improve upon baseline speed
- **Memory**: Must not cause out-of-memory errors

## 🎯 **OPTIMIZATION OPPORTUNITIES**

Based on previous work, consider these proven strategies:

### **Memory Optimization**
- Efficient expert loading/unloading patterns
- Memory pooling and reuse strategies
- CUDA stream management for non-blocking operations

### **Compute Optimization**
- Parallel expert execution
- Prefetching and buffering strategies
- Advanced caching beyond static expert placement

### **System-Level Optimization**
- Threading for compute/memory overlap
- Event-based synchronization
- Resource pooling (events, streams, buffers)

## 📊 **BENCHMARKING GUIDELINES**

### **Performance Targets**
- **Correctness**: 100% test pass rate
- **Speed**: Aim for >1.0x speedup vs baseline FiddlerMixtral
- **Memory**: Stay within available GPU memory limits
- **Reliability**: Consistent performance across multiple runs

### **Fair Comparison**
- Use identical test prompts and parameters
- Same expert caching configuration when possible
- Account for warm-up effects in timing measurements
- Report both prefill and decode phase performance

## 🚀 **GETTING STARTED**

1. **Setup**: Ensure CUDA environment is properly configured
2. **Baseline**: Run `FiddlerMixtral` tests to establish baseline performance
3. **Implementation**: Create your optimization in a new file
4. **Testing**: Update test configuration and validate correctness
5. **Iteration**: Profile, optimize, and repeat

## 💡 **SUCCESS CRITERIA**

Your implementation is ready when:
- ✅ `python quick_test.py YourImplementation` shows ✅ MATCH!
- ✅ Performance meets or exceeds baseline FiddlerMixtral (speedup > 1.0x)
- ✅ Code is well-documented and maintainable
- ✅ Memory usage is reasonable and stable

---

**This template provides a clean foundation for Mixtral optimization work. Focus on one optimization strategy at a time, validate thoroughly, and build incrementally on the proven baseline.**