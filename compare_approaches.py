import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# Add src to path so we can import fiddler modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.mixtral import FiddlerMixtral
from fiddler.mixtral_prefetch_fast import FiddlerMixtralPrefetchFast


def generate_diverse_prompts(base_input, batch_size):
    """Generate diverse prompts to force different expert selection"""
    
    # Predefined diverse prompt categories that should trigger different experts
    diverse_templates = [
        # Mathematical/Technical
        "Calculate the derivative of x^2 + 3x",
        "Explain quantum computing principles",
        "What is machine learning?",
        "Describe neural network architectures",
        "How does blockchain technology work?",
        
        # Creative/Literary  
        "Write a short poem about",
        "Tell me a story about",
        "Create a dialogue between",
        "Describe the scene where",
        "Imagine a world where",
        
        # Factual/Informational
        "What are the main causes of",
        "Explain the history of", 
        "List the benefits of",
        "Compare and contrast",
        "What factors contribute to",
        
        # Conversational/Social
        "How do you feel about",
        "What's your opinion on",
        "Can you help me understand",
        "I'm curious about",
        "Please explain why",
        
        # Problem-solving
        "How would you solve",
        "What's the best way to",
        "Give me steps to", 
        "Help me figure out",
        "What approach would you take to",
        
        # Scientific/Academic
        "The scientific method involves",
        "Research indicates that",
        "According to studies,",
        "Evidence suggests that",
        "The data shows that",
        
        # Different languages/contexts
        "En español, explica",
        "From a philosophical perspective,",
        "In terms of economics,",
        "From a psychological standpoint,",
        "Considering environmental factors,",
        
        # Technical domains
        "In computer science,", 
        "From an engineering perspective,",
        "In medical terms,",
        "Regarding legal aspects,",
        "In the context of physics,"
    ]
    
    # If we have enough diverse templates, use them directly
    if batch_size <= len(diverse_templates):
        return diverse_templates[:batch_size]
    
    # Otherwise, create variations of the base input
    if batch_size <= 8:
        # For small batches, use very different prompt types
        variations = [
            f"Mathematically speaking, {base_input}",
            f"From a creative perspective, {base_input}", 
            f"Scientifically, {base_input}",
            f"Historically, {base_input}",
            f"Philosophically, {base_input}",
            f"In simple terms, {base_input}",
            f"Technically, {base_input}",
            f"Practically, {base_input}"
        ]
        return variations[:batch_size]
    
    # For larger batches, combine templates with base input
    result = []
    for i in range(batch_size):
        template = diverse_templates[i % len(diverse_templates)]
        if base_input.lower() in template.lower():
            result.append(template)
        else:
            result.append(f"{template} {base_input}")
    
    return result


def run_single_comparison(args, batch_size=1):
    """Run comparison for a specific batch size"""
    
    print(f"=== Testing Batch Size {batch_size} ===")

    # Results storage for this batch size
    baseline_times = {"prefill": [], "decode": []}
    prefetch_times = {"prefill": [], "decode": []}
    baseline_hit_rates = []

    # Create diverse inputs for batch processing to force different expert selection
    if batch_size > 1:
        # IMPROVED: Use diverse prompts to force different expert selection
        diverse_prompts = generate_diverse_prompts(args.input, batch_size)
        test_input = diverse_prompts
        print(f"    Using diverse prompts: {diverse_prompts}")
    else:
        test_input = args.input

    # Run baseline approach multiple times
    print(f"  Running baseline approach (batch_size={batch_size})...")
    for run in range(args.runs):
        print(f"    Run {run + 1}/{args.runs}")
        
        # Set cpu_offload to 1 for the current Fiddler approach
        args.cpu_offload = 1
        
        # Add max_experts parameter if specified
        if hasattr(args, 'max_experts_gpu') and args.max_experts_gpu:
            args.max_experts_gpu = args.max_experts_gpu
        
        model_baseline = FiddlerMixtral(args)
        
        prefill_time, decode_time, hit_rate = model_baseline.generate(
            test_input, output_token=args.n_token
        )
        
        baseline_times["prefill"].append(prefill_time)
        baseline_times["decode"].append(decode_time)
        baseline_hit_rates.append(hit_rate)
        
        # Clean up
        del model_baseline
        import torch
        torch.cuda.empty_cache()

    print(f"  Running FAST prefetch approach (batch_size={batch_size})...")
    for run in range(args.runs):
        print(f"    Run {run + 1}/{args.runs}")
        
        model_prefetch = FiddlerMixtralPrefetchFast(args)
        
        prefill_time, decode_time, _ = model_prefetch.generate(
            test_input, output_token=args.n_token
        )
        
        prefetch_times["prefill"].append(prefill_time)
        prefetch_times["decode"].append(decode_time)
        
        # Clean up
        del model_prefetch
        import torch
        torch.cuda.empty_cache()

    # Calculate statistics for this batch size
    baseline_total = np.array(baseline_times["prefill"]) + np.array(baseline_times["decode"])
    prefetch_total = np.array(prefetch_times["prefill"]) + np.array(prefetch_times["decode"])
    
    baseline_mean = np.mean(baseline_total)
    prefetch_mean = np.mean(prefetch_total)
    
    speedup = baseline_mean / prefetch_mean
    
    print(f"  Results for batch_size={batch_size}:")
    print(f"    Baseline: {baseline_mean:.4f}s, Prefetch: {prefetch_mean:.4f}s, Speedup: {speedup:.2f}x")
    print()

    return {
        'batch_size': batch_size,
        'baseline_total': baseline_total,
        'prefetch_total': prefetch_total,
        'baseline_mean': baseline_mean,
        'prefetch_mean': prefetch_mean,
        'speedup': speedup,
        'baseline_times': baseline_times,
        'prefetch_times': prefetch_times,
        'baseline_hit_rates': baseline_hit_rates
    }


def run_comparison(args):
    """Run comparison across different batch sizes"""
    
    print("=== Fiddler Approach Comparison with Variable Batch Sizes ===")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Tokens to generate: {args.n_token}")
    print(f"Runs per approach: {args.runs}")
    print(f"Batch sizes to test: {args.batch_sizes}")
    if hasattr(args, 'max_experts_gpu') and args.max_experts_gpu:
        print(f"Max experts on GPU (baseline): {args.max_experts_gpu}")
    print()

    # Run comparison for each batch size
    all_results = []
    for batch_size in args.batch_sizes:
        result = run_single_comparison(args, batch_size)
        all_results.append(result)

    # Print summary
    print("=== SUMMARY ACROSS BATCH SIZES ===")
    print("Batch Size | Baseline Time | Prefetch Time | Speedup")
    print("-" * 55)
    for result in all_results:
        print(f"{result['batch_size']:10d} | {result['baseline_mean']:11.4f}s | {result['prefetch_mean']:11.4f}s | {result['speedup']:7.2f}x")
    
    print()
    best_speedup = max(all_results, key=lambda x: x['speedup'])
    print(f"🏆 Best speedup: {best_speedup['speedup']:.2f}x at batch_size={best_speedup['batch_size']}")
    
    return all_results


def create_batch_size_visualization(all_results, args):
    """Create visualization showing performance across different batch sizes"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for plotting
    batch_sizes = [r['batch_size'] for r in all_results]
    baseline_means = [r['baseline_mean'] for r in all_results]
    prefetch_means = [r['prefetch_mean'] for r in all_results]
    speedups = [r['speedup'] for r in all_results]
    
    # Plot 1: Performance vs Batch Size (line plot)
    ax1.plot(batch_sizes, baseline_means, 'o-', color='blue', linewidth=2, markersize=8, label='Baseline (Current Fiddler)')
    ax1.plot(batch_sizes, prefetch_means, 's-', color='red', linewidth=2, markersize=8, label='FAST Prefetch (2 Buffers)')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Time (seconds)')
    ax1.set_title('Performance vs Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2) if max(batch_sizes) >= 4 else None
    
    # Plot 2: Speedup vs Batch Size
    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = ax2.bar(range(len(batch_sizes)), speedups, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Break-even line')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup (Baseline/Prefetch)')
    ax2.set_title('Speedup vs Batch Size')
    ax2.set_xticks(range(len(batch_sizes)))
    ax2.set_xticklabels([str(bs) for bs in batch_sizes])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add speedup values on top of bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Time breakdown by batch size
    prefill_baseline = [np.mean(r['baseline_times']['prefill']) for r in all_results]
    decode_baseline = [np.mean(r['baseline_times']['decode']) for r in all_results]
    prefill_prefetch = [np.mean(r['prefetch_times']['prefill']) for r in all_results]
    decode_prefetch = [np.mean(r['prefetch_times']['decode']) for r in all_results]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    ax3.bar(x - width/2, prefill_baseline, width, label='Baseline Prefill', color='lightblue', alpha=0.8)
    ax3.bar(x - width/2, decode_baseline, width, bottom=prefill_baseline, label='Baseline Decode', color='blue', alpha=0.8)
    ax3.bar(x + width/2, prefill_prefetch, width, label='Prefetch Prefill', color='lightcoral', alpha=0.8)
    ax3.bar(x + width/2, decode_prefetch, width, bottom=prefill_prefetch, label='Prefetch Decode', color='red', alpha=0.8)
    
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Time Breakdown by Batch Size')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(bs) for bs in batch_sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax4.axis('off')
    
    best_result = max(all_results, key=lambda x: x['speedup'])
    worst_result = min(all_results, key=lambda x: x['speedup'])
    
    max_experts_text = f"Max experts on GPU: {args.max_experts_gpu}" if hasattr(args, 'max_experts_gpu') and args.max_experts_gpu else "Max experts on GPU: Auto (based on available memory)"
    
    summary_text = f"""
BATCH SIZE PERFORMANCE ANALYSIS

Model: {args.model.split('/')[-1]}
Input: "{args.input[:30]}..." 
Tokens: {args.n_token}, Runs per test: {args.runs}
{max_experts_text}

TESTED BATCH SIZES: {batch_sizes}

🏆 BEST PERFORMANCE:
• Batch size: {best_result['batch_size']}
• Speedup: {best_result['speedup']:.2f}x
• Baseline: {best_result['baseline_mean']:.2f}s
• Prefetch: {best_result['prefetch_mean']:.2f}s

📉 WORST PERFORMANCE:
• Batch size: {worst_result['batch_size']}
• Speedup: {worst_result['speedup']:.2f}x  
• Baseline: {worst_result['baseline_mean']:.2f}s
• Prefetch: {worst_result['prefetch_mean']:.2f}s

OBSERVATIONS:
{'✅ Prefetch wins at larger batch sizes!' if best_result['speedup'] > 1 else '⚠️  Baseline faster across all tested batch sizes'}
{'📈 Performance improves with batch size' if speedups[-1] > speedups[0] else '📉 Performance degrades with batch size'}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    
    # Save the plot
    output_file = f'fiddler_batch_comparison_{int(time.time())}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Batch size analysis saved as: {output_file}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Fiddler baseline vs prefetch approach across batch sizes")
    
    parser.add_argument(
        "--model",
        type=str, 
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="University of Washington is",
        help="Input text for generation"
    )
    parser.add_argument(
        "--n-token", 
        type=int,
        default=10,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of runs per approach per batch size"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs='+',
        default=[1, 2, 4],
        help="Batch sizes to test (e.g., --batch-sizes 1 2 4 8)"
    )
    parser.add_argument(
        "--max-experts-gpu",
        type=int,
        default=None,
        help="Maximum number of experts to place on GPU for baseline (default: auto-detect based on memory)"
    )
    parser.add_argument("--beam-width", type=int, default=1, help="Beam search width")
    
    args = parser.parse_args()
    
    # Set environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run comparison across batch sizes
    all_results = run_comparison(args)
    
    # Create batch size visualization
    viz_file = create_batch_size_visualization(all_results, args)
    
    print(f"\n🎉 Batch size comparison complete! Check out {viz_file} for detailed results.")