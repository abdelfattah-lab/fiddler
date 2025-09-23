#!/usr/bin/env python3
"""
Profile the single buffer Qwen implementation to analyze
fetch vs compute timing patterns.
"""

import sys
import os
from datetime import datetime

# Import the generic profiling function
from profile_nsight import profile_program

def create_single_buffer_test():
    """Create test script for single buffer profiling."""
    script_content = '''#!/usr/bin/env python3
"""Test script for single buffer profiling with detailed NVTX markers."""

import torch
import nvtx
from qwen_single_buffer import QwenSingleBuffer

def main():
    print("🚀 Single Buffer Profiling Test")

    with nvtx.annotate("Model Initialization"):
        model = QwenSingleBuffer()

    test_prompt = "The capital of France is"
    print(f"📝 Test prompt: '{test_prompt}'")

    with nvtx.annotate("Text Generation"):
        result = model.generate(test_prompt, max_new_tokens=8)

    print(f"✅ Generated: '{result['text']}'")
    print(f"⏱️ Time: {result['generation_time']:.2f}s")
    print(f"📊 Expert fetches: {result['expert_fetches']}")
    print(f"📊 Expert hits: {result['expert_hits']}")
    print(f"📊 Hit rate: {result['hit_rate']:.1%}")

    memory = model.get_memory_usage()
    if memory:
        print(f"💾 GPU Memory: {memory['allocated_mb']} MB")

if __name__ == "__main__":
    main()
'''

    with open("single_buffer_test.py", "w") as f:
        f.write(script_content)

    print("📝 Created single_buffer_test.py")

def profile_single_buffer():
    """Profile single buffer implementation."""
    print("🚀 Single Buffer Profiling")
    print("=" * 50)

    # Create test script
    create_single_buffer_test()

    # Create profile directory
    profile_dir = f"single_buffer_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(profile_dir, exist_ok=True)
    print(f"📁 Profile directory: {profile_dir}")

    # Profile the single buffer implementation
    command_args = ["bash", "-c", "source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen_profiling && python single_buffer_test.py"]
    scenario_name = "Single Buffer Qwen"
    output_name = "single_buffer_profile"

    print(f"🔍 Starting profiling...")
    print(f"📊 Focus: Expert fetch vs compute timing")

    success, profile_path = profile_program(
        command_args=command_args,
        scenario_name=scenario_name,
        output_name=output_name,
        profile_dir=profile_dir,
        duration=120,
        timeout=180
    )

    if success:
        print(f"\n✅ Single buffer profiling completed!")
        print(f"📊 Profile saved to: {profile_path}")

        print(f"\n🔍 ANALYSIS COMMANDS:")
        print(f"# Expert fetch operations:")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {profile_path}")
        print(f"nsys stats --report nvtx_sum {profile_path}")

        print(f"\n# Compute operations:")
        print(f"nsys stats --report cuda_gpu_kern_sum {profile_path}")

        print(f"\n# GUI analysis:")
        print(f"nsight-sys {profile_path}")

        print(f"\n📈 KEY METRICS TO ANALYZE:")
        print(f"1. Expert Fetch Time: Look for 'Expert Fetch' NVTX markers")
        print(f"2. Memory Transfer Patterns: CPU->GPU expert loading")
        print(f"3. Compute Efficiency: GPU kernel execution vs memory overhead")
        print(f"4. Single Buffer Utilization: Buffer reuse patterns")

    else:
        print(f"\n❌ Single buffer profiling failed")

    return success, profile_path if success else None

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("🚀 Single Buffer Profiling Script")
        print("=" * 40)
        print("Profile Qwen single buffer implementation to analyze")
        print("expert fetch vs compute timing patterns.")
        print()
        print("Usage:")
        print("  python profile_single_buffer.py")
        return

    success, profile_path = profile_single_buffer()

    if success:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Analyze expert fetch timing with NVTX markers")
        print(f"2. Compare single buffer vs original memory patterns")
        print(f"3. Measure fetch overhead vs compute time")
        print(f"4. Identify optimization opportunities")

if __name__ == "__main__":
    main()