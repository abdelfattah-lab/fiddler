#!/usr/bin/env python3
"""
Analyze the nsight sys profiles to identify speedup sources.

This script analyzes 4 profile configurations:
1. Fiddler-only at BS=8
2. Fiddler+Learned at BS=8
3. Fiddler-only at BS=16
4. Fiddler+Learned at BS=16
"""

import subprocess
import re
import json
from pathlib import Path


def run_nsys_stats(profile_path, report_type):
    """Run nsys stats on a profile and return the output."""
    cmd = ['nsys', 'stats', '--report', report_type, str(profile_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout
    except subprocess.TimeoutExpired:
        return f"ERROR: Timeout running nsys stats on {profile_path}"
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_cuda_api_summary(output):
    """Extract key CUDA API stats from nsys output."""
    stats = {}

    # Look for cudaMemcpy statistics
    memcpy_pattern = r'cudaMemcpy\S*\s+(\d+)\s+[\d.]+\s+([\d.]+)'
    for match in re.finditer(memcpy_pattern, output):
        call_name = match.group(0).split()[0]
        count = int(match.group(1))
        time_ms = float(match.group(2))
        if 'memcpy' not in stats:
            stats['memcpy'] = {'count': 0, 'time_ms': 0.0}
        stats['memcpy']['count'] += count
        stats['memcpy']['time_ms'] += time_ms

    # Look for cudaLaunchKernel
    kernel_pattern = r'cudaLaunchKernel\s+(\d+)\s+[\d.]+\s+([\d.]+)'
    match = re.search(kernel_pattern, output)
    if match:
        stats['kernel_launches'] = {
            'count': int(match.group(1)),
            'time_ms': float(match.group(2))
        }

    return stats


def analyze_profile(profile_name, profile_path):
    """Analyze a single profile."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {profile_name}")
    print(f"{'='*80}\n")

    analysis = {
        'profile_name': profile_name,
        'profile_path': str(profile_path)
    }

    # Get CUDA API summary
    print("Extracting CUDA API summary...")
    cuda_api_output = run_nsys_stats(profile_path, 'cuda_api_sum')

    # Get NVTX summary (for custom ranges)
    print("Extracting NVTX summary...")
    nvtx_output = run_nsys_stats(profile_path, 'nvtx_sum')

    # Extract statistics
    cuda_stats = extract_cuda_api_summary(cuda_api_output)

    analysis['cuda_api_stats'] = cuda_stats

    # Save raw outputs for manual inspection
    output_dir = Path('speedup_profiles/analysis')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f'{profile_name}_cuda_api.txt', 'w') as f:
        f.write(cuda_api_output)

    with open(output_dir / f'{profile_name}_nvtx.txt', 'w') as f:
        f.write(nvtx_output)

    print(f"\n✅ Analysis complete for {profile_name}")
    print(f"   Raw outputs saved to: speedup_profiles/analysis/")

    return analysis


def compare_configurations():
    """Compare the configurations and identify speedup sources."""

    profiles = {
        'fiddler_bs8': 'speedup_profiles/fiddler_bs8.nsys-rep',
        'fiddler_learned_bs8': 'speedup_profiles/fiddler_learned_bs8.nsys-rep',
        'fiddler_bs16': 'speedup_profiles/fiddler_bs16.nsys-rep',
        'fiddler_learned_bs16': 'speedup_profiles/fiddler_learned_bs16.nsys-rep',
    }

    results = {}

    for name, path in profiles.items():
        profile_path = Path(path)
        if not profile_path.exists():
            print(f"WARNING: Profile not found: {path}")
            continue

        results[name] = analyze_profile(name, profile_path)

    # Save combined results
    with open('speedup_profiles/analysis/combined_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print("Results saved to: speedup_profiles/analysis/")
    print("\nGenerated files:")
    print("  - combined_analysis.json")
    print("  - *_cuda_api.txt")
    print("  - *_nvtx.txt")

    return results


def main():
    """Main analysis function."""
    print("="*80)
    print("NSIGHT SYSTEMS PROFILE ANALYSIS")
    print("="*80)
    print("\nThis script analyzes the nsight sys profiles to identify speedup sources.")
    print("\nConfigurations:")
    print("  1. Fiddler-only at BS=8")
    print("  2. Fiddler+Learned at BS=8")
    print("  3. Fiddler-only at BS=16")
    print("  4. Fiddler+Learned at BS=16")
    print("="*80)

    results = compare_configurations()

    print("\nYou can now:")
    print("  1. Review the analysis files in speedup_profiles/analysis/")
    print("  2. Open the .nsys-rep files in Nsight Systems GUI for visual analysis")
    print("  3. Use the extracted statistics to write the speedup analysis report")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
