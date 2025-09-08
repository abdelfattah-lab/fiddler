#!/usr/bin/env python3
"""
MixtralWithBuffers vs FiddlerMixtral Comparison Script

Generates plot comparing:
- MixtralWithBuffers (new implementation with buffer sets and prefetching)
- FiddlerMixtral (baseline implementation)

Results are saved to CSV first, then plotted with insightful analyses.
"""

import sys
import os
import time
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler import FiddlerMixtral
from fiddler.mixtral_with_buffers import MixtralWithBuffers
import torch


def generate_diverse_inputs(batch_size: int) -> List[str]:
    """Generate diverse input prompts to force different expert usage"""
    base_prompts = [
        "The future of artificial intelligence is",
        "In mathematics, the fundamental theorem states", 
        "The ancient civilization of Egypt was known for",
        "Climate change affects global weather patterns by",
        "Machine learning algorithms work by",
        "The history of computer science began with",
        "Quantum physics describes the behavior of",
        "Economic theory suggests that markets",
        "The process of photosynthesis involves",
        "Neural networks are inspired by",
        "The discovery of DNA led to",
        "Space exploration has revealed that",
        "Programming languages evolved from",
        "The theory of relativity explains",
        "Biological evolution occurs through",
        "Modern computing architectures rely on",
    ]
    
    if batch_size <= len(base_prompts):
        return base_prompts[:batch_size]
    else:
        return [base_prompts[i % len(base_prompts)] for i in range(batch_size)]


def run_fiddler_baseline_timing(args, batch_size: int, max_experts_gpu: int) -> Dict:
    """Run baseline FiddlerMixtral model and return detailed timing info"""
    # Create baseline args
    baseline_args = type(args)()
    for attr in dir(args):
        if not attr.startswith('_'):
            setattr(baseline_args, attr, getattr(args, attr))
    
    # Set baseline configuration
    baseline_args.cpu_offload = 1
    baseline_args.beam_width = 1
    baseline_args.max_experts_gpu = max_experts_gpu
    
    print(f"  FiddlerMixtral: Using {max_experts_gpu} preloaded experts")
    
    model = FiddlerMixtral(baseline_args)
    
    # Generate diverse inputs for testing
    inputs = generate_diverse_inputs(batch_size)
    
    # Warmup run
    if batch_size == 1:
        model.generate(inputs[0], output_token=1)
    else:
        model.generate(inputs, output_token=1)
    torch.cuda.empty_cache()
    
    # Time execution with detailed metrics
    times = []
    memory_peaks = []
    
    for run in range(args.runs):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # FiddlerMixtral takes list for batch processing or single string
        if batch_size == 1:
            result = model.generate(inputs[0], output_token=args.n_token)
        else:
            result = model.generate(inputs, output_token=args.n_token)
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        times.append(end_time - start_time)
        memory_peaks.append(peak_memory)
        
        # Clear cache between runs
        torch.cuda.empty_cache()
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_memory': np.mean(memory_peaks),
        'std_memory': np.std(memory_peaks),
        'raw_times': times,
        'raw_memories': memory_peaks
    }


def run_buffers_timing(args, batch_size: int, num_buffer_sets: int) -> Dict:
    """Run MixtralWithBuffers model and return detailed timing info"""
    buffers_args = type(args)()
    for attr in dir(args):
        if not attr.startswith('_'):
            setattr(buffers_args, attr, getattr(args, attr))
    
    # Set buffers configuration
    buffers_args.beam_width = 1
    buffers_args.max_experts_gpu = 0  # MixtralWithBuffers manages its own experts
    
    print(f"  MixtralWithBuffers: Using {num_buffer_sets} buffer sets ({num_buffer_sets * 8} total experts)")
    
    model = MixtralWithBuffers(buffers_args, num_buffer_sets=num_buffer_sets)
    
    # Generate diverse inputs for testing
    inputs = generate_diverse_inputs(batch_size)
    
    # Warmup run - handle single vs batch the same way as FiddlerMixtral
    if batch_size == 1:
        model.generate(inputs[0], output_token=1)
    else:
        model.generate(inputs, output_token=1)
    torch.cuda.empty_cache()
    
    # Time execution with detailed metrics
    times = []
    memory_peaks = []
    
    for run in range(args.runs):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # MixtralWithBuffers can handle both single string and batch (list)
        if batch_size == 1:
            result = model.generate(inputs[0], output_token=args.n_token)
        else:
            result = model.generate(inputs, output_token=args.n_token)
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        times.append(end_time - start_time)
        memory_peaks.append(peak_memory)
        
        # Clear cache between runs
        torch.cuda.empty_cache()
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_memory': np.mean(memory_peaks),
        'std_memory': np.std(memory_peaks),
        'raw_times': times,
        'raw_memories': memory_peaks
    }


def run_model_comparison(args, buffer_set_counts: List[int], batch_sizes: List[int]) -> Dict:
    """Run fair comparison between MixtralWithBuffers and FiddlerMixtral"""
    print(f"Testing buffer set counts: {buffer_set_counts}")
    print(f"Testing batch sizes: {batch_sizes}")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n=== Testing batch size {batch_size} ===")
        batch_results = []
        
        for num_buffer_sets in buffer_set_counts:
            print(f"  Testing {num_buffer_sets} buffer sets...")
            
            try:
                # Fair comparison: both models use same amount of GPU memory
                max_experts_gpu = num_buffer_sets * 8
                
                baseline_results = run_fiddler_baseline_timing(args, batch_size, max_experts_gpu)
                buffers_results = run_buffers_timing(args, batch_size, num_buffer_sets)
                
                speedup = baseline_results['mean_time'] / buffers_results['mean_time'] if buffers_results['mean_time'] > 0 else 0.0
                memory_ratio = buffers_results['mean_memory'] / baseline_results['mean_memory'] if baseline_results['mean_memory'] > 0 else 1.0
                
                result_entry = {
                    'batch_size': batch_size,
                    'num_buffer_sets': num_buffer_sets,
                    'total_experts': max_experts_gpu,
                    
                    # Baseline (FiddlerMixtral) results
                    'baseline_time_mean': baseline_results['mean_time'],
                    'baseline_time_std': baseline_results['std_time'],
                    'baseline_memory_mean': baseline_results['mean_memory'],
                    'baseline_memory_std': baseline_results['std_memory'],
                    
                    # Buffers (MixtralWithBuffers) results  
                    'buffers_time_mean': buffers_results['mean_time'],
                    'buffers_time_std': buffers_results['std_time'],
                    'buffers_memory_mean': buffers_results['mean_memory'],
                    'buffers_memory_std': buffers_results['std_memory'],
                    
                    # Derived metrics
                    'speedup': speedup,
                    'memory_ratio': memory_ratio,
                    'time_improvement_percent': ((baseline_results['mean_time'] - buffers_results['mean_time']) / baseline_results['mean_time']) * 100,
                }
                
                batch_results.append(result_entry)
                
                print(f"    FiddlerMixtral:      {baseline_results['mean_time']:.3f}±{baseline_results['std_time']:.3f}s, {baseline_results['mean_memory']:.2f}±{baseline_results['std_memory']:.2f}GB")
                print(f"    MixtralWithBuffers:  {buffers_results['mean_time']:.3f}±{buffers_results['std_time']:.3f}s, {buffers_results['mean_memory']:.2f}±{buffers_results['std_memory']:.2f}GB")
                print(f"    Speedup: {speedup:.2f}x, Memory ratio: {memory_ratio:.2f}x, Time improvement: {result_entry['time_improvement_percent']:.1f}%")
                
            except Exception as e:
                print(f"    Error: {e}")
                result_entry = {
                    'batch_size': batch_size,
                    'num_buffer_sets': num_buffer_sets,
                    'total_experts': num_buffer_sets * 8,
                    'baseline_time_mean': float('inf'),
                    'baseline_time_std': 0,
                    'baseline_memory_mean': 0,
                    'baseline_memory_std': 0,
                    'buffers_time_mean': float('inf'),
                    'buffers_time_std': 0,
                    'buffers_memory_mean': 0,
                    'buffers_memory_std': 0,
                    'speedup': 0.0,
                    'memory_ratio': 1.0,
                    'time_improvement_percent': 0.0,
                }
                batch_results.append(result_entry)
        
        results[batch_size] = batch_results
    
    return results


def save_results_to_csv(results: Dict, args) -> str:
    """Save results to CSV file for further analysis"""
    timestamp = int(time.time())
    filename = f'mixtral_comparison_results_{timestamp}.csv'
    
    # Flatten results for CSV
    csv_data = []
    for batch_size, batch_results in results.items():
        for result in batch_results:
            csv_data.append(result)
    
    if not csv_data:
        print("No results to save")
        return filename
    
    # Write to CSV
    fieldnames = csv_data[0].keys()
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Results saved to CSV: {filename}")
    return filename


def create_comparison_plots(results: Dict, args, csv_filename: str) -> str:
    """Create comprehensive comparison visualization"""
    
    # Setup plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MixtralWithBuffers vs FiddlerMixtral Performance Comparison', fontsize=16, fontweight='bold')
    
    # Color palette for different batch sizes
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    # Plot 1: Speedup vs Number of Buffer Sets
    ax1.set_title('Speedup vs Number of Buffer Sets')
    ax1.set_xlabel('Number of Buffer Sets')
    ax1.set_ylabel('Speedup (FiddlerMixtral/MixtralWithBuffers)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even (1.0x)')
    
    for i, (batch_size, batch_data) in enumerate(results.items()):
        buffer_counts = [entry['num_buffer_sets'] for entry in batch_data]
        speedups = [entry['speedup'] for entry in batch_data if entry['speedup'] > 0]
        
        if speedups:
            ax1.plot(buffer_counts[:len(speedups)], speedups, 
                    marker='o', label=f'Batch size {batch_size}', 
                    color=colors[i], linewidth=2, markersize=6)
    
    ax1.legend()
    max_speedup = max(1.5, max([max([e['speedup'] for e in batch_data if e['speedup'] > 0], default=1.0) 
                               for batch_data in results.values() if batch_data], default=1.0))
    ax1.set_ylim(0, max_speedup * 1.1)
    
    # Plot 2: Execution Times Comparison
    ax2.set_title('Execution Times Comparison')
    ax2.set_xlabel('Number of Buffer Sets')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    for i, (batch_size, batch_data) in enumerate(results.items()):
        buffer_counts = [entry['num_buffer_sets'] for entry in batch_data]
        baseline_times = [entry['baseline_time_mean'] for entry in batch_data if entry['baseline_time_mean'] != float('inf')]
        buffers_times = [entry['buffers_time_mean'] for entry in batch_data if entry['buffers_time_mean'] != float('inf')]
        
        if baseline_times:
            ax2.plot(buffer_counts[:len(baseline_times)], baseline_times, 
                    marker='s', linestyle='--', label=f'FiddlerMixtral B{batch_size}', 
                    color=colors[i], alpha=0.7, linewidth=1)
        if buffers_times:
            ax2.plot(buffer_counts[:len(buffers_times)], buffers_times, 
                    marker='o', label=f'MixtralWithBuffers B{batch_size}', 
                    color=colors[i], linewidth=2)
    
    ax2.legend()
    
    # Plot 3: Memory Usage Comparison
    ax3.set_title('Memory Usage Comparison')
    ax3.set_xlabel('Number of Buffer Sets')
    ax3.set_ylabel('Peak Memory (GB)')
    ax3.grid(True, alpha=0.3)
    
    for i, (batch_size, batch_data) in enumerate(results.items()):
        buffer_counts = [entry['num_buffer_sets'] for entry in batch_data]
        baseline_memory = [entry['baseline_memory_mean'] for entry in batch_data if entry['baseline_memory_mean'] > 0]
        buffers_memory = [entry['buffers_memory_mean'] for entry in batch_data if entry['buffers_memory_mean'] > 0]
        
        if baseline_memory:
            ax3.plot(buffer_counts[:len(baseline_memory)], baseline_memory, 
                    marker='s', linestyle='--', label=f'FiddlerMixtral B{batch_size}', 
                    color=colors[i], alpha=0.7, linewidth=1)
        if buffers_memory:
            ax3.plot(buffer_counts[:len(buffers_memory)], buffers_memory, 
                    marker='o', label=f'MixtralWithBuffers B{batch_size}', 
                    color=colors[i], linewidth=2)
    
    ax3.legend()
    ax3.axhline(y=8, color='orange', linestyle=':', alpha=0.5, label='8GB GPU')
    ax3.axhline(y=16, color='blue', linestyle=':', alpha=0.5, label='16GB GPU')
    ax3.axhline(y=24, color='green', linestyle=':', alpha=0.5, label='24GB GPU')
    
    # Plot 4: Performance Summary and Analysis
    ax4.set_title('Performance Analysis Summary')
    ax4.axis('off')
    
    # Create detailed summary text
    summary_text = "Model Comparison Analysis\n\n"
    
    # Calculate overall statistics
    best_speedups = {}
    best_configs = {}
    for batch_size, batch_data in results.items():
        if batch_data:
            valid_results = [entry for entry in batch_data if entry['speedup'] > 0]
            if valid_results:
                best_result = max(valid_results, key=lambda x: x['speedup'])
                best_speedups[batch_size] = best_result['speedup']
                best_configs[batch_size] = best_result['num_buffer_sets']
    
    if best_speedups:
        avg_best_speedup = np.mean(list(best_speedups.values()))
        summary_text += f"Average Best Speedup: {avg_best_speedup:.2f}x\n"
        summary_text += f"Best Results by Batch Size:\n"
        for batch_size in sorted(best_speedups.keys()):
            speedup = best_speedups[batch_size]
            config = best_configs[batch_size]
            summary_text += f"  Batch {batch_size}: {speedup:.2f}x @ {config} buffer sets\n"
    
    summary_text += f"\nTest Configuration:\n"
    summary_text += f"Model: {args.model}\n"
    summary_text += f"Output tokens: {args.n_token}\n"
    summary_text += f"Runs per config: {args.runs}\n"
    summary_text += f"CSV file: {csv_filename}\n"
    summary_text += f"Comparison: Fair memory usage\n"
    
    # Add insights
    summary_text += f"\nKey Insights:\n"
    summary_text += f"• MixtralWithBuffers uses buffer sets with prefetching\n"
    summary_text += f"• FiddlerMixtral uses traditional expert caching\n"
    summary_text += f"• Both models use same GPU memory for fair comparison\n"
    summary_text += f"• Results saved to CSV for further analysis\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Save plot
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = f'mixtral_comparison_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_filename


def main():
    parser = argparse.ArgumentParser(description='MixtralWithBuffers vs FiddlerMixtral Performance Comparison')
    parser.add_argument('--model', default='mistralai/Mixtral-8x7B-v0.1', help='Model path')
    parser.add_argument('--n-token', type=int, default=10, help='Number of output tokens')
    parser.add_argument('--runs', type=int, default=3, help='Runs per configuration')
    parser.add_argument('--beam-width', type=int, default=1, help='Beam width for generation')
    
    # Test configurations
    parser.add_argument('--quick', action='store_true', help='Quick test: 2-3 buffer sets, batch 1-2')
    parser.add_argument('--medium', action='store_true', help='Medium test: 2-4 buffer sets, batch 1-4')
    parser.add_argument('--full', action='store_true', help='Full test: 2-6 buffer sets, batch 1-8')
    
    args = parser.parse_args()
    
    # Set test configurations
    if args.quick:
        buffer_set_counts = [2]
        batch_sizes = [256]
    elif args.medium:
        buffer_set_counts = [2, 4]
        batch_sizes = [1, 2, 4]
    elif args.full:
        buffer_set_counts = [2, 4, 6]
        batch_sizes = [1, 2, 4, 8]
    else:
        # Default configuration
        buffer_set_counts = [2, 3, 4]
        batch_sizes = [1, 2]
    
    print(f"MixtralWithBuffers vs FiddlerMixtral Comparison Starting...")
    print(f"Model: {args.model}")
    print(f"Buffer set counts: {buffer_set_counts}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output tokens: {args.n_token}")
    print(f"Runs per config: {args.runs}")
    print(f"Fair comparison: Both models use same amount of GPU memory")
    
    # Run comparison
    results = run_model_comparison(args, buffer_set_counts, batch_sizes)
    
    # Save to CSV first
    csv_filename = save_results_to_csv(results, args)
    
    # Then create plots
    plot_filename = create_comparison_plots(results, args, csv_filename)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to CSV: {csv_filename}")
    print(f"Plots saved to: {plot_filename}")
    
    return results, csv_filename, plot_filename


if __name__ == "__main__":
    # Disable tokenizer parallelism to avoid threading issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()