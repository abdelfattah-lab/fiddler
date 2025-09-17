#!/usr/bin/env python3
"""
Enhanced profiling script with configuration validation.
Profile both baseline FiddlerMixtral and FiddlerMixtralWithPrefetch implementations
using verified identical configurations to obtain valid performance data.

CRITICAL: This script validates and prints all configuration settings to ensure
both implementations use identical parameters, preventing invalid comparisons.
"""

import torch
import sys
import os
import time
import json
from datetime import datetime
from argparse import Namespace

# Add src to path
sys.path.insert(0, 'src')

from fiddler.mixtral import FiddlerMixtral
from fiddler.mixtral_with_prefetch import FiddlerMixtralWithPrefetch

# Force CUDA initialization
torch.cuda.init()
torch.cuda.set_device(0)
test_tensor = torch.tensor([1.0]).cuda()
print(f"✅ CUDA initialized successfully on device: {torch.cuda.get_device_name(0)}")
del test_tensor
torch.cuda.empty_cache()

def create_test_args():
    """Create standardized test arguments for both implementations"""
    args = Namespace()
    args.model = "mistralai/Mixtral-8x7B-v0.1"
    args.max_experts_gpu = 0  # Force CPU offloading for consistent test
    args.cpu_offload = 0      # CRITICAL: Must be 0 for both implementations
    args.beam_width = 1       # CRITICAL: Must be 1 for both implementations
    return args

def validate_and_print_config(args, implementation_name):
    """
    Validate and print all critical configuration settings.

    Args:
        args: Namespace with configuration parameters
        implementation_name: Name of the implementation being tested

    Returns:
        dict: Configuration dictionary for logging
    """
    config = {
        'implementation': implementation_name,
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'cpu_offload': args.cpu_offload,
        'max_experts_gpu': args.max_experts_gpu,
        'beam_width': args.beam_width,
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
        'cuda_memory_allocated': torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0,
        'torch_version': torch.__version__,
    }

    print("=" * 80)
    print(f"🔧 CONFIGURATION VALIDATION - {implementation_name.upper()}")
    print("=" * 80)
    print(f"Implementation: {config['implementation']}")
    print(f"Timestamp: {config['timestamp']}")
    print(f"Model: {config['model']}")
    print(f"CPU Offload: {config['cpu_offload']} (CRITICAL: Must be 0)")
    print(f"Max Experts GPU: {config['max_experts_gpu']} (CRITICAL: Must be 0)")
    print(f"Beam Width: {config['beam_width']} (CRITICAL: Must be 1)")
    print(f"CUDA Device: {config['cuda_device']}")
    print(f"CUDA Memory Allocated: {config['cuda_memory_allocated']:,} bytes")
    print(f"PyTorch Version: {config['torch_version']}")
    print("=" * 80)

    # Validate critical settings
    critical_errors = []
    if config['cpu_offload'] != 0:
        critical_errors.append(f"cpu_offload={config['cpu_offload']}, must be 0")
    if config['max_experts_gpu'] != 0:
        critical_errors.append(f"max_experts_gpu={config['max_experts_gpu']}, must be 0")
    if config['beam_width'] != 1:
        critical_errors.append(f"beam_width={config['beam_width']}, must be 1")

    if critical_errors:
        print("❌ CRITICAL CONFIGURATION ERRORS:")
        for error in critical_errors:
            print(f"   - {error}")
        print("❌ PROFILING ABORTED - Fix configuration errors first")
        sys.exit(1)

    print("✅ Configuration validation passed - all critical settings correct")
    print()

    return config

def save_config_log(config, profile_dir="profiles"):
    """Save configuration to timestamped log file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{profile_dir}/{timestamp}", exist_ok=True)

    config_file = f"{profile_dir}/{timestamp}/config_{config['implementation']}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"📁 Configuration saved to: {config_file}")
    return f"{profile_dir}/{timestamp}"

def run_baseline_profile():
    """Profile the baseline FiddlerMixtral implementation with configuration validation"""
    print("🔍 Profiling baseline FiddlerMixtral...")

    # Clear GPU memory
    torch.cuda.empty_cache()

    args = create_test_args()

    # CRITICAL: Validate and print configuration
    config = validate_and_print_config(args, "FiddlerMixtral")
    profile_dir = save_config_log(config)

    print("🏗️ Loading baseline model...")
    model = FiddlerMixtral(args)

    # Verify model configuration after initialization
    print(f"📋 Model loaded - verifying internal configuration:")
    if hasattr(model, 'cpu_offload'):
        print(f"   - Model cpu_offload: {model.cpu_offload}")
        if model.cpu_offload != 0:
            print(f"❌ ERROR: Model cpu_offload={model.cpu_offload}, expected 0")
            sys.exit(1)
    if hasattr(model, 'max_experts_gpu'):
        print(f"   - Model max_experts_gpu: {model.max_experts_gpu}")
        if model.max_experts_gpu != 0:
            print(f"❌ ERROR: Model max_experts_gpu={model.max_experts_gpu}, expected 0")
            sys.exit(1)

    # Test prompt - longer than quick_test for better profiling
    test_prompt = "The capital of France is Paris, which is known for its beautiful architecture and rich cultural heritage. The Eiffel Tower stands majestically in the heart of the city, attracting millions of visitors each year who come to admire"

    print("🔥 Running warmup generation...")
    model.generate(test_prompt, output_token=5)

    # Profile run with more tokens for better analysis
    print("🚀 Starting profiled generation (15 tokens)...")
    start_time = time.time()

    # This is the section that will be profiled by nsys
    prefill_time, decode_time, hit_rate = model.generate(test_prompt, output_token=15)

    end_time = time.time()

    results = {
        'total_time': end_time - start_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'hit_rate': hit_rate,
        'generated_text': model.last_generated_text
    }

    print(f"✅ Baseline completed:")
    print(f"   - Total time: {results['total_time']:.3f}s")
    print(f"   - Prefill: {results['prefill_time']:.3f}s")
    print(f"   - Decode: {results['decode_time']:.3f}s")
    print(f"   - Hit rate: {results['hit_rate']:.2%}")
    print(f"   - Generated: '{results['generated_text']}'")

    # Save results
    results_file = f"{profile_dir}/results_baseline.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to: {results_file}")

    del model
    torch.cuda.empty_cache()
    return profile_dir

def run_prefetch_profile():
    """Profile the FiddlerMixtralWithPrefetch implementation with configuration validation"""
    print("🔍 Profiling FiddlerMixtralWithPrefetch...")

    # Clear GPU memory
    torch.cuda.empty_cache()

    args = create_test_args()

    # CRITICAL: Validate and print configuration
    config = validate_and_print_config(args, "FiddlerMixtralWithPrefetch")
    profile_dir = save_config_log(config)

    print("🏗️ Loading prefetch model...")
    model = FiddlerMixtralWithPrefetch(args)

    # Verify model configuration after initialization
    print(f"📋 Model loaded - verifying internal configuration:")
    if hasattr(model, 'cpu_offload'):
        print(f"   - Model cpu_offload: {model.cpu_offload}")
        if model.cpu_offload != 0:
            print(f"❌ ERROR: Model cpu_offload={model.cpu_offload}, expected 0")
            sys.exit(1)
    if hasattr(model, 'max_experts_gpu'):
        print(f"   - Model max_experts_gpu: {model.max_experts_gpu}")
        if model.max_experts_gpu != 0:
            print(f"❌ ERROR: Model max_experts_gpu={model.max_experts_gpu}, expected 0")
            sys.exit(1)

    # Test prompt - same as baseline
    test_prompt = "The capital of France is Paris, which is known for its beautiful architecture and rich cultural heritage. The Eiffel Tower stands majestically in the heart of the city, attracting millions of visitors each year who come to admire"

    print("🔥 Running warmup generation to build prefetch patterns...")
    model.generate(test_prompt, output_token=5)

    # Profile run with more tokens
    print("🚀 Starting profiled generation (15 tokens)...")
    start_time = time.time()

    # This is the section that will be profiled by nsys
    prefill_time, decode_time, hit_rate = model.generate(test_prompt, output_token=15)

    end_time = time.time()

    results = {
        'total_time': end_time - start_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'hit_rate': hit_rate,
        'generated_text': model.last_generated_text
    }

    # Add prefetch-specific metrics
    if hasattr(model, 'prefetch_metrics'):
        results['prefetch_hit_rate'] = model.prefetch_metrics.get_overall_hit_rate()
        print(f"   - Prefetch hit rate: {results['prefetch_hit_rate']:.2%}")

    print(f"✅ Prefetch completed:")
    print(f"   - Total time: {results['total_time']:.3f}s")
    print(f"   - Prefill: {results['prefill_time']:.3f}s")
    print(f"   - Decode: {results['decode_time']:.3f}s")
    print(f"   - Hit rate: {results['hit_rate']:.2%}")
    print(f"   - Generated: '{results['generated_text']}'")

    # Save results
    results_file = f"{profile_dir}/results_prefetch.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to: {results_file}")

    del model
    torch.cuda.empty_cache()
    return profile_dir

def compare_configurations(baseline_dir, prefetch_dir):
    """
    Compare configurations between baseline and prefetch runs to ensure they are identical.

    Args:
        baseline_dir: Directory containing baseline configuration
        prefetch_dir: Directory containing prefetch configuration
    """
    try:
        with open(f"{baseline_dir}/config_FiddlerMixtral.json", 'r') as f:
            baseline_config = json.load(f)
        with open(f"{prefetch_dir}/config_FiddlerMixtralWithPrefetch.json", 'r') as f:
            prefetch_config = json.load(f)

        print("=" * 80)
        print("🔍 CONFIGURATION COMPARISON")
        print("=" * 80)

        critical_params = ['cpu_offload', 'max_experts_gpu', 'beam_width', 'model']
        comparison_passed = True

        for param in critical_params:
            baseline_val = baseline_config.get(param)
            prefetch_val = prefetch_config.get(param)

            status = "✅" if baseline_val == prefetch_val else "❌"
            print(f"{status} {param}: Baseline={baseline_val}, Prefetch={prefetch_val}")

            if baseline_val != prefetch_val:
                comparison_passed = False

        if comparison_passed:
            print("✅ Configuration comparison PASSED - both implementations use identical settings")
        else:
            print("❌ Configuration comparison FAILED - implementations have different settings")
            print("❌ PROFILING RESULTS ARE INVALID")
            return False

        print("=" * 80)
        return True

    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find configuration file: {e}")
        return False

def main():
    if len(sys.argv) not in [2, 4]:
        print("Usage: python profile_implementations.py <baseline|prefetch>")
        print("       python profile_implementations.py compare <baseline_dir> <prefetch_dir>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "baseline":
        profile_dir = run_baseline_profile()
        print(f"\n📁 Baseline profiling completed. Results in: {profile_dir}")
        print("💡 Next: Run 'python profile_implementations.py prefetch' to profile prefetch implementation")

    elif command == "prefetch":
        profile_dir = run_prefetch_profile()
        print(f"\n📁 Prefetch profiling completed. Results in: {profile_dir}")
        print("💡 Next: Use Nsight Systems to compare the .nsys-rep files, or run configuration comparison")

    elif command == "compare":
        if len(sys.argv) != 4:
            print("Usage for compare: python profile_implementations.py compare <baseline_dir> <prefetch_dir>")
            sys.exit(1)
        baseline_dir = sys.argv[2]
        prefetch_dir = sys.argv[3]
        compare_configurations(baseline_dir, prefetch_dir)

    else:
        print("Error: Command must be 'baseline', 'prefetch', or 'compare'")
        sys.exit(1)

if __name__ == "__main__":
    main()