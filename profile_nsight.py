#!/usr/bin/env python3
"""
Unified Nvidia Nsight Systems profiling script for Fiddler Mixtral implementations.
Supports both baseline vs prefetch comparison and hit rate comparison modes.
"""

import sys
import os
import subprocess
import time
import argparse
from datetime import datetime

def run_nsight_profile(command_args, scenario_name, output_name, profile_dir):
    """Run nsight profiling for a specific scenario."""

    # Nsight command with optimized settings for GPU memory analysis
    nsys_cmd = [
        "nsys", "profile",
        "--output", f"{profile_dir}/{output_name}.nsys-rep",
        "--force-overwrite", "true",
        "--trace", "cuda,cudnn,cublas,osrt,nvtx",
        "--cuda-memory-usage", "true",
        "--gpu-metrics-devices", "all",  # Updated from deprecated --gpu-metrics-device
        "--duration", "60",
        # Remove --sample cpu as it often fails and isn't critical
    ] + command_args

    print(f"🔍 Profiling {scenario_name}...")
    print(f"📁 Output: {profile_dir}/{output_name}.nsys-rep")
    print(f"🚀 Command: {' '.join(nsys_cmd)}")

    try:
        result = subprocess.run(nsys_cmd,
                              capture_output=True,
                              text=True,
                              timeout=180)  # 3 minute timeout

        if result.returncode == 0:
            print(f"✅ {scenario_name} profiling completed successfully")
            print(f"📊 Profile saved to: {profile_dir}/{output_name}.nsys-rep")
        else:
            print(f"❌ {scenario_name} profiling failed")
            print(f"stderr: {result.stderr}")
            if result.stdout:
                print(f"stdout: {result.stdout}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏱️ {scenario_name} profiling timed out")
        return False
    except Exception as e:
        print(f"💥 Error profiling {scenario_name}: {e}")
        return False

def generate_analysis_commands(profile_files):
    """Generate analysis commands for the given profile files."""
    print(f"\n🔍 ANALYSIS COMMANDS:")

    # Individual analysis
    for name, path in profile_files.items():
        print(f"\n# {name} analysis:")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {path}")
        print(f"nsys stats --report cuda_api_sum {path}")
        print(f"nsys stats --report cuda_gpu_kern_sum {path}")
        print(f"nsight-sys {path}  # GUI")

    # Comparison analysis if multiple profiles
    if len(profile_files) >= 2:
        print(f"\n🆚 COMPARISON ANALYSIS:")
        files = list(profile_files.values())
        names = list(profile_files.keys())

        print(f"# Memory operations comparison ({names[0]} vs {names[1]}):")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {files[0]}")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {files[1]}")

        print(f"\n# CUDA API comparison ({names[0]} vs {names[1]}):")
        print(f"nsys stats --report cuda_api_sum {files[0]}")
        print(f"nsys stats --report cuda_api_sum {files[1]}")

def baseline_vs_prefetch_mode():
    """Profile baseline vs prefetch implementations."""
    print("🚀 Baseline vs Prefetch Profiling for Fiddler Mixtral")
    print("=" * 65)

    # Create profile directory
    profile_dir = f"comparison_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(profile_dir, exist_ok=True)

    # Warm-up prefetch to generate patterns
    print("🔧 Warming up FiddlerMixtralWithPrefetch to generate expert patterns...")
    try:
        subprocess.run(["python", "quick_test.py", "FiddlerMixtralWithPrefetch"],
                      capture_output=True, timeout=60)
        print("✅ Warm-up completed")
    except:
        print("⚠️ Warm-up failed, continuing anyway...")

    implementations = [
        (["python", "quick_test.py", "FiddlerMixtral"], "Baseline (FiddlerMixtral)", "baseline"),
        (["python", "quick_test.py", "FiddlerMixtralWithPrefetch"], "Prefetch (FiddlerMixtralWithPrefetch)", "prefetch")
    ]

    results = {}

    for command_args, scenario_name, output_name in implementations:
        success = run_nsight_profile(command_args, scenario_name, output_name, profile_dir)

        if success:
            print(f"✅ {scenario_name} profiling completed")
            results[scenario_name] = f"{profile_dir}/{output_name}.nsys-rep"
        else:
            print(f"❌ {scenario_name} profiling failed")

        time.sleep(2)  # Brief pause between runs

    print("\n" + "=" * 65)
    print("📋 BASELINE vs PREFETCH PROFILING SUMMARY")
    print("=" * 65)

    for name, path in results.items():
        print(f"✅ {name}: {path}")

    if len(results) == 0:
        print("❌ All profiling failed")
        return

    generate_analysis_commands(results)

def hit_rate_comparison_mode():
    """Profile 0% hit rate vs high hit rate scenarios."""
    print("🚀 Hit Rate Comparison Profiling for FiddlerMixtralWithPrefetch")
    print("=" * 70)

    # Create profile directory
    profile_dir = f"hit_rate_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(profile_dir, exist_ok=True)

    results = {}

    # Stage 1: Profile with 0% hit rate (no expert_usage_patterns.json)
    print("📊 STAGE 1: Profiling with 0% hit rate (no patterns file)")
    print("-" * 50)

    if os.path.exists("expert_usage_patterns.json"):
        print("⚠️ expert_usage_patterns.json exists - deleting for clean 0% hit rate test")
        os.remove("expert_usage_patterns.json")

    command_args = ["python", "quick_test.py", "FiddlerMixtralWithPrefetch"]
    success_0 = run_nsight_profile(command_args, "0% Hit Rate", "hit_rate_0_percent", profile_dir)

    if success_0:
        print("✅ 0% hit rate profiling completed")
        results["0% Hit Rate"] = f"{profile_dir}/hit_rate_0_percent.nsys-rep"
    else:
        print("❌ Failed to profile 0% hit rate scenario")
        return

    time.sleep(3)

    # Stage 2: Generate patterns by running once to create expert_usage_patterns.json
    print("\n📊 STAGE 2: Generating expert patterns...")
    print("-" * 50)

    try:
        result = subprocess.run(
            ["python", "quick_test.py", "FiddlerMixtralWithPrefetch"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✅ Expert patterns generated")
        else:
            print("⚠️ Pattern generation had issues, continuing...")
    except:
        print("⚠️ Pattern generation failed, continuing...")

    time.sleep(3)

    # Stage 3: Profile with high hit rate (with expert_usage_patterns.json)
    print("\n📊 STAGE 3: Profiling with high hit rate (with patterns file)")
    print("-" * 50)

    if not os.path.exists("expert_usage_patterns.json"):
        print("⚠️ expert_usage_patterns.json not created - high hit rate test may not work")

    success_high = run_nsight_profile(command_args, "High Hit Rate", "hit_rate_high_percent", profile_dir)

    if success_high:
        print("✅ High hit rate profiling completed")
        results["High Hit Rate"] = f"{profile_dir}/hit_rate_high_percent.nsys-rep"
    else:
        print("❌ High hit rate profiling failed")

    print("\n" + "=" * 70)
    print("📋 HIT RATE COMPARISON SUMMARY")
    print("=" * 70)

    for name, path in results.items():
        print(f"✅ {name}: {path}")

    if len(results) == 0:
        print("❌ All profiling failed")
        return

    generate_analysis_commands(results)

def main():
    parser = argparse.ArgumentParser(description='Unified Nsight profiling for Fiddler Mixtral')
    parser.add_argument('mode', choices=['baseline-vs-prefetch', 'hit-rate-comparison'],
                       help='Profiling mode to run')

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("🚀 Unified Nsight Profiling Script for Fiddler Mixtral")
        print("=" * 55)
        print("Usage:")
        print("  python profile_nsight.py baseline-vs-prefetch")
        print("    - Profiles FiddlerMixtral vs FiddlerMixtralWithPrefetch")
        print("    - Compares baseline implementation with prefetch optimization")
        print()
        print("  python profile_nsight.py hit-rate-comparison")
        print("    - Profiles FiddlerMixtralWithPrefetch with 0% vs high hit rate")
        print("    - Compares prefetch performance with different hit rates")
        print()
        return

    args = parser.parse_args()

    if args.mode == 'baseline-vs-prefetch':
        baseline_vs_prefetch_mode()
    elif args.mode == 'hit-rate-comparison':
        hit_rate_comparison_mode()

if __name__ == "__main__":
    main()