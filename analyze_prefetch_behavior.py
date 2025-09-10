#!/usr/bin/env python3
"""
Comprehensive hybrid prefetching analysis script.
Tests different prefetch percentages and plots speedup vs prefetching percentage.
"""

import argparse
import time
import csv
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Import the models
import sys
sys.path.append('src')

from fiddler.mixtral_with_buffers import MixtralWithBuffers
from fiddler.mixtral import FiddlerMixtral


def create_args(model_path: str = "mistralai/Mixtral-8x7B-v0.1"):
    """Create arguments object for model initialization"""
    class Args:
        def __init__(self):
            self.model = model_path
            self.beam_width = 1
            self.max_experts_gpu = 16  # For compatibility

    return Args()


def benchmark_prefetch_percentage(
    prefetch_percentage: int,
    prompt: str,
    n_tokens: int,
    runs: int = 3
) -> Dict:
    """
    Benchmark MixtralWithBuffers with specific prefetch percentage.
    
    Returns dictionary with performance metrics.
    """
    print(f"\n=== Testing {prefetch_percentage}% prefetching ===")
    
    times = []
    prefill_times = []
    decode_times = []
    hit_rates = []
    
    for run in range(runs):
        print(f"Run {run + 1}/{runs}...")
        
        # Create model with specific prefetch percentage
        args = create_args()
        model = MixtralWithBuffers(args, prefetch_percentage=prefetch_percentage)
        
        # Run benchmark
        start_time = time.time()
        prefill_time, decode_time, hit_rate = model.generate(
            text=prompt, 
            output_token=n_tokens
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        times.append(total_time)
        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
        hit_rates.append(hit_rate)
        
        # Print detailed stats for this run
        model.print_prefetch_stats()
        
        # Cleanup model
        model.shutdown_prefetch_threads()
        del model
    
    # Calculate statistics
    result = {
        'prefetch_percentage': prefetch_percentage,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_prefill_time': np.mean(prefill_times),
        'std_prefill_time': np.std(prefill_times),
        'mean_decode_time': np.mean(decode_times),
        'std_decode_time': np.std(decode_times),
        'mean_hit_rate': np.mean(hit_rates),
        'std_hit_rate': np.std(hit_rates),
        'runs': runs
    }
    
    print(f"Results for {prefetch_percentage}% prefetching:")
    print(f"  Total time: {result['mean_time']:.2f} ± {result['std_time']:.2f}s")
    print(f"  Prefill: {result['mean_prefill_time']:.2f} ± {result['std_prefill_time']:.2f}s") 
    print(f"  Decode: {result['mean_decode_time']:.2f} ± {result['std_decode_time']:.2f}s")
    print(f"  Hit rate: {result['mean_hit_rate']:.3f} ± {result['std_hit_rate']:.3f}")
    
    return result


def benchmark_baseline_fiddler(prompt: str, n_tokens: int, runs: int = 3) -> Dict:
    """Benchmark baseline FiddlerMixtral for comparison"""
    print(f"\n=== Testing baseline FiddlerMixtral ===")
    
    times = []
    prefill_times = []
    decode_times = []
    hit_rates = []
    
    for run in range(runs):
        print(f"Run {run + 1}/{runs}...")
        
        # Create baseline model
        args = create_args()
        model = FiddlerMixtral(args)
        
        # Run benchmark
        start_time = time.time()
        prefill_time, decode_time, hit_rate = model.generate(
            text=prompt,
            output_token=n_tokens
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        times.append(total_time)
        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
        hit_rates.append(hit_rate)
        
        # Cleanup model
        del model
    
    # Calculate statistics
    result = {
        'prefetch_percentage': 'baseline',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_prefill_time': np.mean(prefill_times),
        'std_prefill_time': np.std(prefill_times),
        'mean_decode_time': np.mean(decode_times),
        'std_decode_time': np.std(decode_times),
        'mean_hit_rate': np.mean(hit_rates),
        'std_hit_rate': np.std(hit_rates),
        'runs': runs
    }
    
    print(f"Results for baseline FiddlerMixtral:")
    print(f"  Total time: {result['mean_time']:.2f} ± {result['std_time']:.2f}s")
    print(f"  Prefill: {result['mean_prefill_time']:.2f} ± {result['std_prefill_time']:.2f}s")
    print(f"  Decode: {result['mean_decode_time']:.2f} ± {result['std_decode_time']:.2f}s") 
    print(f"  Hit rate: {result['mean_hit_rate']:.3f} ± {result['std_hit_rate']:.3f}")
    
    return result


def save_results_to_csv(results: List[Dict], filename: str):
    """Save results to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['prefetch_percentage', 'mean_time', 'std_time', 
                     'mean_prefill_time', 'std_prefill_time',
                     'mean_decode_time', 'std_decode_time',
                     'mean_hit_rate', 'std_hit_rate', 'runs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {filename}")


def plot_speedup_vs_prefetch_percentage(results: List[Dict], baseline_time: float, filename: str):
    """Create comprehensive plot showing speedup vs prefetch percentage"""
    # Prepare data (exclude baseline from percentage plot)
    percentages = []
    speedups = []
    speedup_errors = []
    times = []
    time_errors = []
    hit_rates = []
    hit_rate_errors = []
    
    for result in results:
        if result['prefetch_percentage'] != 'baseline':
            percentages.append(result['prefetch_percentage'])
            speedup = baseline_time / result['mean_time'] 
            speedups.append(speedup)
            
            # Error propagation for speedup
            speedup_error = speedup * (result['std_time'] / result['mean_time'])
            speedup_errors.append(speedup_error)
            
            times.append(result['mean_time'])
            time_errors.append(result['std_time'])
            hit_rates.append(result['mean_hit_rate'])
            hit_rate_errors.append(result['std_hit_rate'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hybrid Prefetching Performance Analysis', fontsize=16)
    
    # Plot 1: Speedup vs Prefetch Percentage
    ax1.errorbar(percentages, speedups, yerr=speedup_errors, 
                marker='o', linestyle='-', capsize=5, capthick=2)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Baseline (1.0x)')
    ax1.set_xlabel('Prefetch Percentage (%)')
    ax1.set_ylabel('Speedup (vs Baseline)')
    ax1.set_title('Speedup vs Prefetch Percentage')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight best performing percentage
    if speedups:
        best_idx = np.argmax(speedups)
        best_percentage = percentages[best_idx]
        best_speedup = speedups[best_idx]
        ax1.annotate(f'Best: {best_percentage}% ({best_speedup:.2f}x)', 
                    xy=(best_percentage, best_speedup),
                    xytext=(best_percentage + 10, best_speedup + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontweight='bold', color='red')
    
    # Plot 2: Execution Time vs Prefetch Percentage  
    ax2.errorbar(percentages, times, yerr=time_errors,
                marker='s', linestyle='-', color='orange', capsize=5, capthick=2)
    ax2.axhline(y=baseline_time, color='r', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_time:.1f}s)')
    ax2.set_xlabel('Prefetch Percentage (%)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Execution Time vs Prefetch Percentage')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Hit Rate vs Prefetch Percentage
    ax3.errorbar(percentages, hit_rates, yerr=hit_rate_errors,
                marker='^', linestyle='-', color='green', capsize=5, capthick=2)
    ax3.set_xlabel('Prefetch Percentage (%)')
    ax3.set_ylabel('Expert Hit Rate')
    ax3.set_title('Expert Hit Rate vs Prefetch Percentage')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # Plot 4: Performance Summary (Speedup vs Hit Rate)
    colors = [plt.cm.viridis(p/100) for p in percentages]
    scatter = ax4.scatter(hit_rates, speedups, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Expert Hit Rate')
    ax4.set_ylabel('Speedup (vs Baseline)')
    ax4.set_title('Speedup vs Hit Rate (color = prefetch %)')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for prefetch percentage
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Prefetch Percentage (%)')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    
    # Return best configuration info
    if speedups:
        best_config = {
            'percentage': best_percentage,
            'speedup': best_speedup,
            'time': times[best_idx],
            'hit_rate': hit_rates[best_idx]
        }
        return best_config
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze hybrid prefetching performance")
    parser.add_argument("--percentages", type=str, default="0,25,50,75,100",
                       help="Comma-separated prefetch percentages to test")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs per configuration")
    parser.add_argument("--n-tokens", type=int, default=10,
                       help="Number of output tokens to generate")
    parser.add_argument("--prompt", type=str, 
                       default="What are the main benefits of renewable energy?",
                       help="Input prompt for generation")
    parser.add_argument("--include-baseline", action="store_true",
                       help="Include baseline FiddlerMixtral comparison")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick test: 1 run, 5 tokens, percentages 0,50,100")
    parser.add_argument("--output-prefix", type=str, default="prefetch_analysis",
                       help="Output files prefix")
    
    args = parser.parse_args()
    
    # Quick mode override
    if args.quick:
        percentages = [0, 50, 100]
        runs = 1
        n_tokens = 5
        print("QUICK MODE: Testing percentages [0, 50, 100], 1 run, 5 tokens")
    else:
        percentages = [int(p.strip()) for p in args.percentages.split(",")]
        runs = args.runs
        n_tokens = args.n_tokens
    
    print(f"Testing prefetch percentages: {percentages}")
    print(f"Runs per configuration: {runs}")
    print(f"Output tokens: {n_tokens}")
    print(f"Prompt: {args.prompt}")
    
    # Collect results
    results = []
    
    # Test baseline if requested
    baseline_time = None
    if args.include_baseline:
        baseline_result = benchmark_baseline_fiddler(args.prompt, n_tokens, runs)
        results.append(baseline_result)
        baseline_time = baseline_result['mean_time']
    
    # Test each prefetch percentage
    for percentage in percentages:
        result = benchmark_prefetch_percentage(percentage, args.prompt, n_tokens, runs)
        results.append(result)
    
    # Use 0% prefetch as baseline if no baseline was tested
    if baseline_time is None:
        for result in results:
            if result['prefetch_percentage'] == 0:
                baseline_time = result['mean_time']
                break
    
    # Save results to CSV
    timestamp = int(time.time())
    csv_filename = f"{args.output_prefix}_{timestamp}.csv"
    save_results_to_csv(results, csv_filename)
    
    # Create plots
    plot_filename = f"{args.output_prefix}_{timestamp}.png"
    best_config = plot_speedup_vs_prefetch_percentage(results, baseline_time, plot_filename)
    
    # Print final summary
    print(f"\n=== FINAL ANALYSIS SUMMARY ===")
    print(f"Tested prefetch percentages: {percentages}")
    print(f"Baseline time: {baseline_time:.2f}s")
    
    if best_config:
        print(f"\nBest configuration:")
        print(f"  Prefetch percentage: {best_config['percentage']}%")
        print(f"  Speedup: {best_config['speedup']:.2f}x")
        print(f"  Time: {best_config['time']:.2f}s")
        print(f"  Hit rate: {best_config['hit_rate']:.3f}")
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_filename}")
    print(f"  Plot: {plot_filename}")


if __name__ == "__main__":
    main()