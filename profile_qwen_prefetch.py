#!/usr/bin/env python3
"""
Profile FiddlerQwenWithPrefetch using Nsight Systems
"""

import sys
import os
import time
from datetime import datetime

# Add the current directory to Python path so we can import profile_nsight
sys.path.append(os.path.dirname(__file__))

from profile_nsight import profile_program

def main():
    print("🚀 Profiling FiddlerQwenWithPrefetch with Nsight Systems")
    print("=" * 60)

    # Create profile directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    profile_dir = f"qwen_prefetch_profile_{timestamp}"
    os.makedirs(profile_dir, exist_ok=True)

    # Remove any existing patterns file to ensure we start fresh
    patterns_file = "expert_usage_patterns_qwen.json"
    if os.path.exists(patterns_file):
        print(f"🧹 Removing existing {patterns_file} for clean profiling")
        os.remove(patterns_file)

    # First run to collect patterns (will be in collection mode)
    print("\n📊 STAGE 1: Pattern Collection Run")
    print("-" * 40)

    command_args = ["python", "quick_test.py", "FiddlerQwenWithPrefetch"]
    success1, profile_path1 = profile_program(
        command_args=command_args,
        scenario_name="QwenPrefetch Collection Mode",
        output_name="qwen_prefetch_collection",
        profile_dir=profile_dir,
        duration=45  # Shorter duration for quick test
    )

    if success1:
        print(f"✅ Collection mode profiling completed: {profile_path1}")
    else:
        print("❌ Collection mode profiling failed")
        return

    # Brief pause
    time.sleep(3)

    # Second run to use patterns (will be in prediction mode)
    print("\n📊 STAGE 2: Prediction Run (with learned patterns)")
    print("-" * 40)

    success2, profile_path2 = profile_program(
        command_args=command_args,
        scenario_name="QwenPrefetch Prediction Mode",
        output_name="qwen_prefetch_prediction",
        profile_dir=profile_dir,
        duration=45  # Shorter duration for quick test
    )

    if success2:
        print(f"✅ Prediction mode profiling completed: {profile_path2}")
    else:
        print("❌ Prediction mode profiling failed")

    # Summary
    print("\n" + "=" * 60)
    print("📋 QWEN PREFETCH PROFILING SUMMARY")
    print("=" * 60)

    if success1:
        print(f"✅ Collection Mode Profile: {profile_path1}")
    if success2:
        print(f"✅ Prediction Mode Profile: {profile_path2}")

    print(f"\n📁 All profiles saved in: {profile_dir}/")

    # Analysis commands
    print(f"\n🔍 ANALYSIS COMMANDS:")
    if success1:
        print(f"\n# Collection Mode Analysis:")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {profile_path1}")
        print(f"nsys stats --report cuda_api_sum {profile_path1}")
        print(f"nsight-sys {profile_path1}  # GUI")

    if success2:
        print(f"\n# Prediction Mode Analysis:")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {profile_path2}")
        print(f"nsys stats --report cuda_api_sum {profile_path2}")
        print(f"nsight-sys {profile_path2}  # GUI")

    if success1 and success2:
        print(f"\n🆚 COMPARISON ANALYSIS:")
        print(f"# Memory operations comparison:")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {profile_path1}")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {profile_path2}")

    print(f"\n🎯 Profile directory: {profile_dir}/")

if __name__ == "__main__":
    main()