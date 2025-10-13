#!/usr/bin/env python3
"""
Phase 5: Prediction Methods Benchmark
Compares learned predictor approach against baseline and Fiddler configurations
across multiple batch sizes.

Key objectives:
1. Show configurations where Fiddler+Learned-Prefetch > Fiddler alone
2. Use different sentences for each batch element
3. Test multiple batch sizes (1, 2, 4, 8, 16)
4. Multiple trials for statistical reliability
"""

import os
import sys
import time
import torch
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen import FiddlerQwen
from fiddler.qwen_with_learned_prefetch import FiddlerQwenWithLearnedPrefetch


class Args:
    """Arguments for Qwen model."""
    def __init__(self, use_fiddler_mode=False):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1
        self.use_fiddler_mode = use_fiddler_mode
        self.fiddler_batch_threshold = 32  # Use CPU for all tested batch sizes


# Diverse test sentences for batching
# These are different across domains to test generalization
DIVERSE_SENTENCES = [
    # Science & Technology
    "The theory of general relativity was developed by Albert Einstein in",
    "Artificial intelligence systems use machine learning algorithms to",
    "The human brain contains approximately 86 billion neurons that",
    "Quantum computers utilize the principles of superposition and",
    "CRISPR gene editing technology allows scientists to modify",
    "The speed of light in vacuum is approximately 299,792,458",
    "Neural networks are inspired by biological neurons and use",
    "The periodic table organizes chemical elements based on their",

    # History & Geography
    "The Roman Empire reached its greatest extent under Emperor",
    "The Great Wall of China was built over many centuries to",
    "The Renaissance began in Italy during the 14th century and",
    "Mount Everest is the highest mountain on Earth with a",
    "The Industrial Revolution started in Britain in the late",
    "Ancient Egypt was one of the world's first civilizations along",
    "The Amazon rainforest is the largest tropical rainforest covering",
    "World War II began in 1939 when Germany invaded",

    # Arts & Culture
    "Leonardo da Vinci painted the Mona Lisa during the",
    "Shakespeare wrote 37 plays and 154 sonnets during his",
    "The Taj Mahal was built in India by Emperor",
    "Jazz music originated in New Orleans at the turn",
    "The Impressionist movement in art began in France in",
    "The Olympic Games were first held in ancient Greece",
    "Classical music composers like Mozart and Beethoven created",
    "The Great Pyramid of Giza was built as a",

    # Nature & Animals
    "Blue whales are the largest animals ever known to",
    "The Amazon River flows through South America carrying more",
    "Photosynthesis is the process by which plants convert sunlight",
    "The African elephant is the largest land animal weighing",
    "Coral reefs are diverse underwater ecosystems formed by",
    "The Aurora Borealis occurs when solar particles interact with",
    "Monarch butterflies migrate thousands of miles each year from",
    "The Sahara Desert is the largest hot desert covering",

    # Everyday topics
    "The capital of France is Paris, which is located",
    "Coffee is one of the most popular beverages consumed",
    "The internet revolutionized communication by connecting millions of",
    "Basketball was invented by James Naismith in Springfield",
    "Pizza originated in Naples, Italy and has become",
    "The bicycle is a human-powered vehicle with two wheels",
    "Solar panels convert sunlight into electricity using photovoltaic",
    "The Eiffel Tower was completed in 1889 for the",
]


def get_diverse_batch(batch_size: int, seed: int = None) -> List[str]:
    """
    Get a batch of diverse sentences for testing.

    Args:
        batch_size: Number of sentences to return
        seed: Random seed for reproducibility

    Returns:
        List of diverse sentences
    """
    if seed is not None:
        np.random.seed(seed)

    # Shuffle and select batch_size sentences
    sentences = DIVERSE_SENTENCES.copy()
    np.random.shuffle(sentences)
    return sentences[:batch_size]


def run_single_batch_baseline(model, prompts: List[str], output_tokens: int = 20) -> Dict:
    """
    Run baseline or Fiddler (single GPU buffer) on a batch.
    For batch_size=1, use generate(). For batch>1, use manual batching.
    """
    batch_size = len(prompts)

    if batch_size == 1:
        # Use generate method
        prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
            prompts[0],
            output_token=output_tokens
        )
    else:
        # Manual batching
        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.model.device)
        attention_mask = inputs.attention_mask.to(model.model.device)

        # Reset stats
        model.expert_fetch_count = 0
        model.expert_hit_count = 0

        # Generate tokens
        generated = input_ids
        prefill_time = 0.0
        decode_time = 0.0

        for i in range(output_tokens):
            with torch.no_grad():
                is_decode = (i > 0)

                torch.cuda.synchronize()
                step_start = time.time()

                outputs = model.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    use_cache=False
                )

                torch.cuda.synchronize()
                step_time = time.time() - step_start

                if is_decode:
                    decode_time += step_time
                else:
                    prefill_time = step_time

                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=-1)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device)
                ], dim=-1)

        # No hit rates for baseline
        prefill_hit_rate = 0.0
        decode_hit_rate = 0.0

    return {
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'total_time': prefill_time + decode_time,
        'prefill_hit_rate': prefill_hit_rate,
        'decode_hit_rate': decode_hit_rate,
        'tokens_per_second': (output_tokens * batch_size) / decode_time if decode_time > 0 else 0
    }


def run_single_batch_learned(model, prompts: List[str], output_tokens: int = 20) -> Dict:
    """
    Run learned prefetch model on a batch.
    For batch_size=1, use generate(). For batch>1, use manual batching.
    """
    batch_size = len(prompts)

    if batch_size == 1:
        # Use generate method
        prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
            prompts[0],
            output_token=output_tokens
        )
    else:
        # Manual batching with hit rate tracking
        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.model.device)
        attention_mask = inputs.attention_mask.to(model.model.device)

        # Reset stats
        if hasattr(model, 'expert_fetch_count'):
            model.expert_fetch_count = 0
            model.expert_hit_count = 0
            model.cnt_expert_hit = 0
            model.cnt_expert_all = 0
            model.prefill_hit_count = 0
            model.prefill_total = 0
            model.decode_hit_count = 0
            model.decode_total = 0

        # Reset prefetch caches
        if hasattr(model, 'prefetch_cache_A'):
            model.prefetch_cache_A.clear()
            model.prefetch_cache_B.clear()
            model.expert_ready_events.clear()

        # Reset attention capture
        if hasattr(model, 'current_attention_output'):
            model.current_attention_output = None

        # Reset profiler token position
        if hasattr(model, 'profiler') and not model.collection_mode:
            model.profiler.current_token_pos = 0

        # Generate tokens
        generated = input_ids
        prefill_time = 0.0
        decode_time = 0.0

        for i in range(output_tokens):
            with torch.no_grad():
                is_decode = (i > 0)

                torch.cuda.synchronize()
                step_start = time.time()

                outputs = model.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    use_cache=False
                )

                torch.cuda.synchronize()
                step_time = time.time() - step_start

                if is_decode:
                    decode_time += step_time
                else:
                    prefill_time = step_time

                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=-1)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device)
                ], dim=-1)

        # Calculate hit rates
        if hasattr(model, 'prefill_total') and model.prefill_total > 0:
            prefill_hit_rate = model.prefill_hit_count / model.prefill_total
        else:
            prefill_hit_rate = 0.0

        if hasattr(model, 'decode_total') and model.decode_total > 0:
            decode_hit_rate = model.decode_hit_count / model.decode_total
        else:
            decode_hit_rate = 0.0

    return {
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'total_time': prefill_time + decode_time,
        'prefill_hit_rate': prefill_hit_rate,
        'decode_hit_rate': decode_hit_rate,
        'tokens_per_second': (output_tokens * batch_size) / decode_time if decode_time > 0 else 0
    }


def run_configuration(config_name: str, batch_size: int, num_trials: int = 3,
                     output_tokens: int = 20, use_fiddler_mode: bool = False, **model_kwargs) -> Dict:
    """
    Run a single configuration across multiple trials.

    Args:
        config_name: Name of configuration
        batch_size: Batch size to test
        num_trials: Number of trials to run
        output_tokens: Number of tokens to generate per prompt
        use_fiddler_mode: Whether to use Fiddler mode (for FiddlerQwen baseline)
        **model_kwargs: Model initialization arguments

    Returns:
        Dictionary with averaged results across trials
    """
    print(f"\n{'='*80}")
    print(f"CONFIG: {config_name} | Batch Size: {batch_size} | Trials: {num_trials}")
    print(f"{'='*80}")

    # Create args with Fiddler mode if needed
    args = Args(use_fiddler_mode=use_fiddler_mode)

    # Determine model class
    if 'Learned' in config_name:
        model_class = FiddlerQwenWithLearnedPrefetch
        run_batch_fn = run_single_batch_learned
    else:
        model_class = FiddlerQwen
        run_batch_fn = run_single_batch_baseline

    # Load model
    print(f"Loading {model_class.__name__}...")
    model = model_class(args, **model_kwargs)

    trial_results = []

    for trial in range(num_trials):
        print(f"\n  Trial {trial + 1}/{num_trials}:")

        # Get diverse batch (different seed for each trial)
        prompts = get_diverse_batch(batch_size, seed=trial)
        print(f"    Prompts: {[p[:30] + '...' for p in prompts[:3]]}")

        try:
            result = run_batch_fn(model, prompts, output_tokens)
            trial_results.append(result)

            print(f"    ✅ Total: {result['total_time']:.3f}s | "
                  f"Decode: {result['decode_time']:.3f}s | "
                  f"Tok/s: {result['tokens_per_second']:.1f} | "
                  f"Decode hit: {result['decode_hit_rate']*100:.1f}%")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Calculate statistics across trials
    if not trial_results:
        return None

    result_summary = {
        'config': config_name,
        'batch_size': batch_size,
        'num_trials': len(trial_results),
        'model_kwargs': model_kwargs
    }

    # Calculate mean and std for each metric
    for metric in ['prefill_time', 'decode_time', 'total_time',
                   'prefill_hit_rate', 'decode_hit_rate', 'tokens_per_second']:
        values = [r[metric] for r in trial_results]
        result_summary[f'{metric}_mean'] = np.mean(values)
        result_summary[f'{metric}_std'] = np.std(values)

    print(f"\n  📊 SUMMARY ({num_trials} trials):")
    print(f"    Total time: {result_summary['total_time_mean']:.3f}s ± {result_summary['total_time_std']:.3f}s")
    print(f"    Decode time: {result_summary['decode_time_mean']:.3f}s ± {result_summary['decode_time_std']:.3f}s")
    print(f"    Tokens/sec: {result_summary['tokens_per_second_mean']:.1f} ± {result_summary['tokens_per_second_std']:.1f}")
    print(f"    Decode hit: {result_summary['decode_hit_rate_mean']*100:.1f}% ± {result_summary['decode_hit_rate_std']*100:.1f}%")

    return result_summary


def plot_results(results: List[Dict], output_dir: str):
    """Generate comprehensive visualization of results."""

    # Organize by configuration
    configs = {}
    for r in results:
        config = r['config']
        if config not in configs:
            configs[config] = []
        configs[config].append(r)

    # Sort by batch size
    for config in configs:
        configs[config] = sorted(configs[config], key=lambda x: x['batch_size'])

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = {
        'Baseline': '#1f77b4',
        'Fiddler': '#2ca02c',
        'Learned-Prefetch': '#ff7f0e',
        'Fiddler+Learned-Prefetch': '#d62728'
    }

    # Plot 1: Total Time vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        times = [d['total_time_mean'] for d in data]
        errors = [d['total_time_std'] for d in data]
        ax1.errorbar(batch_sizes, times, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax1.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Time (s)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Time vs Batch Size', fontsize=12, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Tokens per Second
    ax2 = fig.add_subplot(gs[0, 1])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        tok_s = [d['tokens_per_second_mean'] for d in data]
        errors = [d['tokens_per_second_std'] for d in data]
        ax2.errorbar(batch_sizes, tok_s, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tokens/Second', fontsize=11, fontweight='bold')
    ax2.set_title('Token Throughput vs Batch Size', fontsize=12, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Speedup vs Baseline
    ax3 = fig.add_subplot(gs[0, 2])
    baseline_data = configs.get('Baseline', [])
    baseline_by_batch = {d['batch_size']: d['total_time_mean'] for d in baseline_data}

    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue
        batch_sizes = [d['batch_size'] for d in data]
        speedups = [baseline_by_batch.get(d['batch_size'], 1.0) / d['total_time_mean']
                   if d['batch_size'] in baseline_by_batch else 1.0 for d in data]
        ax3.plot(batch_sizes, speedups, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Speedup vs Baseline', fontsize=11, fontweight='bold')
    ax3.set_title('Speedup vs Baseline', fontsize=12, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Fiddler vs Fiddler+Learned Direct Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    fiddler_data = configs.get('Fiddler', [])
    learned_fiddler_data = configs.get('Fiddler+Learned-Prefetch', [])

    if fiddler_data and learned_fiddler_data:
        fiddler_bs = [d['batch_size'] for d in fiddler_data]
        fiddler_times = [d['total_time_mean'] for d in fiddler_data]
        fiddler_errs = [d['total_time_std'] for d in fiddler_data]

        lf_bs = [d['batch_size'] for d in learned_fiddler_data]
        lf_times = [d['total_time_mean'] for d in learned_fiddler_data]
        lf_errs = [d['total_time_std'] for d in learned_fiddler_data]

        ax4.errorbar(fiddler_bs, fiddler_times, yerr=fiddler_errs, fmt='o-',
                    label='Fiddler', color=colors['Fiddler'], linewidth=2, markersize=8, capsize=4)
        ax4.errorbar(lf_bs, lf_times, yerr=lf_errs, fmt='s-',
                    label='Fiddler+Learned', color=colors['Fiddler+Learned-Prefetch'],
                    linewidth=2, markersize=8, capsize=4)

    ax4.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Total Time (s)', fontsize=11, fontweight='bold')
    ax4.set_title('🎯 Fiddler vs Fiddler+Learned-Prefetch', fontsize=12, fontweight='bold')
    ax4.set_xscale('log', base=2)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Speedup of Fiddler+Learned vs Fiddler
    ax5 = fig.add_subplot(gs[1, 1])
    if fiddler_data and learned_fiddler_data:
        fiddler_by_batch = {d['batch_size']: d['total_time_mean'] for d in fiddler_data}
        lf_by_batch = {d['batch_size']: d['total_time_mean'] for d in learned_fiddler_data}

        common_bs = sorted(set(fiddler_by_batch.keys()) & set(lf_by_batch.keys()))
        speedups = [fiddler_by_batch[bs] / lf_by_batch[bs] for bs in common_bs]

        bars = ax5.bar(range(len(common_bs)), speedups,
                      color=['green' if s > 1.0 else 'red' for s in speedups],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
        ax5.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Speedup (Fiddler+Learned / Fiddler)', fontsize=11, fontweight='bold')
        ax5.set_title('🏆 Fiddler+Learned Speedup vs Fiddler', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(len(common_bs)))
        ax5.set_xticklabels(common_bs)
        ax5.grid(True, alpha=0.3, axis='y')

        # Annotate bars
        for i, (bs, val) in enumerate(zip(common_bs, speedups)):
            ax5.text(i, val + 0.02, f'{val:.3f}x', ha='center', va='bottom', fontweight='bold')

    # Plot 6: Decode Hit Rates
    ax6 = fig.add_subplot(gs[1, 2])
    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue
        batch_sizes = [d['batch_size'] for d in data]
        hit_rates = [d['decode_hit_rate_mean'] * 100 for d in data]
        errors = [d['decode_hit_rate_std'] * 100 for d in data]
        ax6.errorbar(batch_sizes, hit_rates, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax6.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Decode Hit Rate (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Decode Hit Rate vs Batch Size', fontsize=12, fontweight='bold')
    ax6.set_xscale('log', base=2)
    ax6.set_ylim([0, 105])
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # Plot 7: Prefill Time
    ax7 = fig.add_subplot(gs[2, 0])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        times = [d['prefill_time_mean'] for d in data]
        errors = [d['prefill_time_std'] for d in data]
        ax7.errorbar(batch_sizes, times, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax7.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Prefill Time (s)', fontsize=11, fontweight='bold')
    ax7.set_title('Prefill Phase Time', fontsize=12, fontweight='bold')
    ax7.set_xscale('log', base=2)
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # Plot 8: Decode Time
    ax8 = fig.add_subplot(gs[2, 1])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        times = [d['decode_time_mean'] for d in data]
        errors = [d['decode_time_std'] for d in data]
        ax8.errorbar(batch_sizes, times, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax8.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Decode Time (s)', fontsize=11, fontweight='bold')
    ax8.set_title('Decode Phase Time', fontsize=12, fontweight='bold')
    ax8.set_xscale('log', base=2)
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # Plot 9: Configuration Comparison at Peak Batch Size
    ax9 = fig.add_subplot(gs[2, 2])
    # Find the largest batch size that all configs have
    all_batch_sizes = set.intersection(*[set(d['batch_size'] for d in data)
                                         for data in configs.values() if data])
    if all_batch_sizes:
        max_common_bs = max(all_batch_sizes)
        config_names = []
        tok_s_values = []

        for config_name, data in configs.items():
            for d in data:
                if d['batch_size'] == max_common_bs:
                    config_names.append(config_name)
                    tok_s_values.append(d['tokens_per_second_mean'])
                    break

        bars = ax9.bar(range(len(config_names)), tok_s_values,
                      color=[colors.get(c, 'gray') for c in config_names],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax9.set_xticks(range(len(config_names)))
        ax9.set_xticklabels(config_names, rotation=15, ha='right')
        ax9.set_ylabel('Tokens/Second', fontsize=11, fontweight='bold')
        ax9.set_title(f'Throughput Comparison @ BS={max_common_bs}', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')

        # Annotate bars
        for i, val in enumerate(tok_s_values):
            ax9.text(i, val + max(tok_s_values)*0.02, f'{val:.1f}',
                    ha='center', va='bottom', fontweight='bold')

    # Overall title
    fig.suptitle('Phase 5: Learned Predictor Benchmark Results',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    plot_path = os.path.join(output_dir, 'phase5_benchmark_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive plot saved to: {plot_path}")
    plt.close()


def generate_analysis(results: List[Dict], output_dir: str):
    """Generate detailed analysis of results."""

    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    # Group by configuration
    configs = {}
    for r in results:
        config = r['config']
        if config not in configs:
            configs[config] = []
        configs[config].append(r)

    # Sort by batch size
    for config in configs:
        configs[config] = sorted(configs[config], key=lambda x: x['batch_size'])

    # Find where Fiddler+Learned beats Fiddler
    fiddler_data = configs.get('Fiddler', [])
    learned_fiddler_data = configs.get('Fiddler+Learned-Prefetch', [])

    analysis_report = []
    analysis_report.append("=" * 80)
    analysis_report.append("PHASE 5: KEY FINDINGS")
    analysis_report.append("=" * 80)
    analysis_report.append("")

    if fiddler_data and learned_fiddler_data:
        analysis_report.append("🎯 PRIMARY OBJECTIVE: Where does Fiddler+Learned-Prefetch beat Fiddler?")
        analysis_report.append("")
        analysis_report.append(f"{'Batch Size':<12} {'Fiddler (s)':<15} {'F+Learned (s)':<15} {'Speedup':<12} {'Winner':<20}")
        analysis_report.append("-" * 80)

        fiddler_by_batch = {d['batch_size']: d for d in fiddler_data}
        lf_by_batch = {d['batch_size']: d for d in learned_fiddler_data}

        wins = []
        for bs in sorted(set(fiddler_by_batch.keys()) & set(lf_by_batch.keys())):
            f_time = fiddler_by_batch[bs]['total_time_mean']
            f_std = fiddler_by_batch[bs]['total_time_std']
            lf_time = lf_by_batch[bs]['total_time_mean']
            lf_std = lf_by_batch[bs]['total_time_std']

            speedup = f_time / lf_time
            winner = "🏆 F+Learned" if lf_time < f_time else "Fiddler"

            if lf_time < f_time:
                wins.append(bs)

            analysis_report.append(
                f"{bs:<12} "
                f"{f_time:.3f}±{f_std:.3f}    "
                f"{lf_time:.3f}±{lf_std:.3f}    "
                f"{speedup:.3f}x      "
                f"{winner}"
            )

        analysis_report.append("")
        if wins:
            analysis_report.append(f"✅ Fiddler+Learned-Prefetch WINS at batch sizes: {wins}")
            analysis_report.append(f"   Peak speedup: {max([fiddler_by_batch[bs]['total_time_mean'] / lf_by_batch[bs]['total_time_mean'] for bs in wins]):.3f}x")
        else:
            analysis_report.append("⚠️  Fiddler+Learned-Prefetch does not beat Fiddler at any tested batch size")
            analysis_report.append("   Consider: higher batch sizes, different k values, or different aggregation strategies")

        analysis_report.append("")

    # Overall performance comparison
    analysis_report.append("📊 OVERALL PERFORMANCE SUMMARY")
    analysis_report.append("")

    for config_name, data in configs.items():
        analysis_report.append(f"{config_name}:")
        for d in data:
            bs = d['batch_size']
            time_mean = d['total_time_mean']
            time_std = d['total_time_std']
            tok_s = d['tokens_per_second_mean']
            hit_rate = d['decode_hit_rate_mean'] * 100

            analysis_report.append(
                f"  BS={bs:2d}: {time_mean:.3f}±{time_std:.3f}s | "
                f"{tok_s:.1f} tok/s | Hit: {hit_rate:.1f}%"
            )
        analysis_report.append("")

    # Save analysis
    analysis_text = "\n".join(analysis_report)
    analysis_path = os.path.join(output_dir, 'ANALYSIS.md')
    with open(analysis_path, 'w') as f:
        f.write(analysis_text)

    print(analysis_text)
    print(f"\n💾 Analysis saved to: {analysis_path}")

    return analysis_text


def main():
    """Main benchmark function."""
    print("\n" + "="*80)
    print("PHASE 5: PREDICTION METHODS BENCHMARK")
    print("="*80)
    print("\nObjectives:")
    print("  1. Find configurations where Fiddler+Learned-Prefetch > Fiddler alone")
    print("  2. Test with diverse sentences (different for each batch element)")
    print("  3. Multiple trials for statistical reliability")
    print("  4. Results suitable for research paper")
    print("="*80)

    # Configuration
    batch_sizes = [1, 2, 4, 8, 16]
    num_trials = 3
    output_tokens = 20

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"phase5_benchmark_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Configurations to test
    configurations = [
        {
            'name': 'Baseline',
            'use_fiddler_mode': False,
            'kwargs': {}
        },
        {
            'name': 'Fiddler',
            'use_fiddler_mode': True,  # Enable Fiddler CPU offloading via args
            'kwargs': {}
        },
        {
            'name': 'Learned-Prefetch',
            'use_fiddler_mode': False,
            'kwargs': {
                'num_experts_to_prefetch': 8,
                'enable_cpu_offload': False,
                'predictor_path': 'predictor_checkpoints/best_model.pt'
            }
        },
        {
            'name': 'Fiddler+Learned-Prefetch',
            'use_fiddler_mode': False,  # Learned prefetch has its own CPU offload
            'kwargs': {
                'num_experts_to_prefetch': 8,
                'enable_cpu_offload': True,
                'latency_cpu': 0.1,
                'latency_gpu': 10.0,
                'predictor_path': 'predictor_checkpoints/best_model.pt'
            }
        }
    ]

    print(f"\nConfigurations: {[c['name'] for c in configurations]}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Trials per configuration: {num_trials}")
    print(f"Output tokens per prompt: {output_tokens}")

    # Run all benchmarks
    all_results = []

    for config in configurations:
        for batch_size in batch_sizes:
            try:
                result = run_configuration(
                    config['name'],
                    batch_size,
                    num_trials=num_trials,
                    output_tokens=output_tokens,
                    use_fiddler_mode=config.get('use_fiddler_mode', False),
                    **config['kwargs']
                )

                if result:
                    all_results.append(result)

            except Exception as e:
                print(f"\n❌ Error running {config['name']} @ BS={batch_size}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save raw results
    json_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Raw results saved to: {json_path}")

    # Save CSV
    if all_results:
        csv_path = os.path.join(output_dir, 'benchmark_results.csv')
        with open(csv_path, 'w', newline='') as f:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"💾 CSV results saved to: {csv_path}")

    # Generate visualizations
    if all_results:
        plot_results(all_results, output_dir)

    # Generate analysis
    if all_results:
        generate_analysis(all_results, output_dir)

    print("\n" + "="*80)
    print("✅ PHASE 5 BENCHMARK COMPLETE")
    print("="*80)
    print(f"Results directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
