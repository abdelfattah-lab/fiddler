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
    Get a batch of diverse sentences for testing. Supports large batches by
    creating deterministic variations when unique base sentences are exhausted.

    Args:
        batch_size: Number of sentences to return
        seed: Random seed for reproducibility

    Returns:
        List of diverse sentences
    """
    rng = np.random.default_rng(seed)

    base_sentences = DIVERSE_SENTENCES.copy()
    rng.shuffle(base_sentences)

    if batch_size <= len(base_sentences):
        return base_sentences[:batch_size]

    # Extend with deterministic variations to cover large batches while keeping prompts distinct.
    sentences = base_sentences[:]
    variant_index = 0
    while len(sentences) < batch_size:
        template = DIVERSE_SENTENCES[variant_index % len(DIVERSE_SENTENCES)]
        variant_suffix = variant_index // len(DIVERSE_SENTENCES) + 1
        sentences.append(f"{template} (variation {variant_suffix})")
        variant_index += 1

    rng.shuffle(sentences)
    return sentences[:batch_size]


def generate_text_for_correctness(model, prompts: List[str], output_tokens: int) -> List[str]:
    """
    Generate text from model for correctness checking.
    Uses model.generate from HuggingFace to get actual token IDs.

    Args:
        model: The model instance
        prompts: List of input prompts
        output_tokens: Number of tokens to generate

    Returns:
        List of generated text (decoded from tokens)
    """
    # Tokenize inputs
    if len(prompts) == 1:
        inputs = model.tokenizer(prompts[0], return_tensors="pt")
    else:
        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    input_ids = inputs.input_ids.to(model.model.device)
    attention_mask = inputs.attention_mask.to(model.model.device) if inputs.attention_mask is not None else None

    # Generate tokens
    with torch.no_grad():
        generated_ids = model.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=output_tokens,
            do_sample=False,  # Deterministic generation for correctness check
            pad_token_id=model.tokenizer.eos_token_id,
            eos_token_id=None,  # Force exact token count
        )

    # Decode and return
    generated_texts = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts


def check_correctness(results_by_config: Dict[str, Dict], prompts: List[str]) -> Dict:
    """
    Check if different configurations produce similar outputs.

    Args:
        results_by_config: Dict mapping config name to result dict (with 'generated_text')
        prompts: The input prompts used

    Returns:
        Dictionary with correctness check results
    """
    if not results_by_config or len(results_by_config) < 2:
        return {'status': 'skipped', 'reason': 'Not enough configurations to compare'}

    # Use baseline as reference if available, otherwise use first config
    reference_name = 'Baseline' if 'Baseline' in results_by_config else list(results_by_config.keys())[0]
    reference_text = results_by_config[reference_name].get('generated_text', [])

    if not reference_text:
        return {'status': 'skipped', 'reason': 'No generated text available'}

    # Compare all other configs to reference
    comparisons = {}
    all_match = True

    for config_name, result in results_by_config.items():
        if config_name == reference_name:
            continue

        generated_text = result.get('generated_text', [])
        if not generated_text:
            comparisons[config_name] = {'status': 'no_text'}
            continue

        # Check if outputs match
        matches = [ref == gen for ref, gen in zip(reference_text, generated_text)]
        match_rate = sum(matches) / len(matches) if matches else 0.0

        comparisons[config_name] = {
            'status': 'match' if match_rate == 1.0 else 'mismatch',
            'match_rate': match_rate,
            'reference': reference_name
        }

        if match_rate < 1.0:
            all_match = False

    return {
        'status': 'pass' if all_match else 'fail',
        'reference': reference_name,
        'comparisons': comparisons
    }


def run_single_batch_baseline(model, prompts: List[str], output_tokens: int = 20,
                             return_generated_text: bool = False) -> Dict:
    """
    Run baseline or Fiddler (single GPU buffer) on a batch.
    Now uses the generate() method for all batch sizes (single string or list of strings).

    Args:
        model: The model to run
        prompts: List of input prompts
        output_tokens: Number of tokens to generate
        return_generated_text: If True, also generate and return text for correctness checking
    """
    batch_size = len(prompts)

    # Use generate method with proper input (single string or list of strings)
    if batch_size == 1:
        text_input = prompts[0]
    else:
        text_input = prompts

    prefill_time, decode_time_per_token, prefill_hit_rate, decode_hit_rate = model.generate(
        text_input,
        output_token=output_tokens
    )

    # Convert per-token decode latency back to total decode phase latency.
    decode_tokens = getattr(model, 'decode_token_count', output_tokens)
    decode_time_total = decode_time_per_token * decode_tokens
    total_tokens_generated = decode_tokens * batch_size

    tokens_per_second = (
        total_tokens_generated / decode_time_total
        if decode_time_total > 0 else 0
    )

    result = {
        'prefill_time': prefill_time,
        'decode_time': decode_time_total,
        'decode_time_per_token': decode_time_per_token,
        'decode_tokens': decode_tokens,
        'tokens_generated': total_tokens_generated,
        'total_time': prefill_time + decode_time_total,
        'prefill_hit_rate': prefill_hit_rate,
        'decode_hit_rate': decode_hit_rate,
        'tokens_per_second': tokens_per_second
    }

    # Optionally generate text for correctness checking
    if return_generated_text:
        result['generated_text'] = generate_text_for_correctness(model, prompts, output_tokens)

    return result


def run_single_batch_learned(model, prompts: List[str], output_tokens: int = 20,
                            return_generated_text: bool = False) -> Dict:
    """
    Run learned prefetch model on a batch.
    Now uses the generate() method for all batch sizes (single string or list of strings).

    Args:
        model: The model to run
        prompts: List of input prompts
        output_tokens: Number of tokens to generate
        return_generated_text: If True, also generate and return text for correctness checking
    """
    batch_size = len(prompts)

    # Use generate method with proper input (single string or list of strings)
    if batch_size == 1:
        text_input = prompts[0]
    else:
        text_input = prompts

    prefill_time, decode_time_per_token, prefill_hit_rate, decode_hit_rate = model.generate(
        text_input,
        output_token=output_tokens
    )

    decode_tokens = getattr(model, 'decode_token_count', output_tokens)
    decode_time_total = decode_time_per_token * decode_tokens
    total_tokens_generated = decode_tokens * batch_size

    tokens_per_second = (
        total_tokens_generated / decode_time_total
        if decode_time_total > 0 else 0
    )

    result = {
        'prefill_time': prefill_time,
        'decode_time': decode_time_total,
        'decode_time_per_token': decode_time_per_token,
        'decode_tokens': decode_tokens,
        'tokens_generated': total_tokens_generated,
        'total_time': prefill_time + decode_time_total,
        'prefill_hit_rate': prefill_hit_rate,
        'decode_hit_rate': decode_hit_rate,
        'tokens_per_second': tokens_per_second
    }

    # Optionally generate text for correctness checking
    if return_generated_text:
        result['generated_text'] = generate_text_for_correctness(model, prompts, output_tokens)

    return result


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

            print(
                "    ✅ Total: {total:.3f}s | Decode: {decode:.3f}s ({per_token:.3f}s/token) | "
                "Tok/s: {tps:.1f} | Decode hit: {hit:.1f}%".format(
                    total=result['total_time'],
                    decode=result['decode_time'],
                    per_token=result.get('decode_time_per_token', 0.0),
                    tps=result['tokens_per_second'],
                    hit=result['decode_hit_rate'] * 100,
                )
            )

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
    for metric in ['prefill_time', 'decode_time', 'decode_time_per_token', 'total_time',
                   'prefill_hit_rate', 'decode_hit_rate', 'tokens_per_second',
                   'decode_tokens', 'tokens_generated']:
        values = [r[metric] for r in trial_results]
        result_summary[f'{metric}_mean'] = np.mean(values)
        result_summary[f'{metric}_std'] = np.std(values)

    print(f"\n  📊 SUMMARY ({num_trials} trials):")
    print(f"    Total time: {result_summary['total_time_mean']:.3f}s ± {result_summary['total_time_std']:.3f}s")
    print(
        "    Decode time: {total:.3f}s ± {total_std:.3f}s ({per_token:.3f}s/token)".format(
            total=result_summary['decode_time_mean'],
            total_std=result_summary['decode_time_std'],
            per_token=result_summary['decode_time_per_token_mean'],
        )
    )
    print(f"    Tokens/sec: {result_summary['tokens_per_second_mean']:.1f} ± {result_summary['tokens_per_second_std']:.1f}")
    print(
        "    Decode tokens: {per_seq:.1f} per sequence | Total tokens: {total_toks:.1f}".format(
            per_seq=result_summary['decode_tokens_mean'],
            total_toks=result_summary['tokens_generated_mean'],
        )
    )
    print(f"    Decode hit: {result_summary['decode_hit_rate_mean']*100:.1f}% ± {result_summary['decode_hit_rate_std']*100:.1f}%")

    return result_summary


def plot_results(results: List[Dict], output_dir: str):
    """Generate comprehensive visualization of results with prefill/decode separation."""

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

    # Create figure with 4x3 grid for better organization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    colors = {
        'Baseline': '#1f77b4',
        'Fiddler': '#2ca02c',
        'Learned-Prefetch': '#ff7f0e',
        'Fiddler+Learned-Prefetch': '#d62728'
    }

    # ============================================================================
    # ROW 1: OVERALL METRICS
    # ============================================================================

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

    # Plot 3: Total Speedup vs Baseline
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
    ax3.set_title('Total Speedup vs Baseline', fontsize=12, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # ============================================================================
    # ROW 2: PREFILL PHASE METRICS
    # ============================================================================

    # Plot 4: Prefill Time
    ax4 = fig.add_subplot(gs[1, 0])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        times = [d['prefill_time_mean'] for d in data]
        errors = [d['prefill_time_std'] for d in data]
        ax4.errorbar(batch_sizes, times, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax4.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Prefill Time (s)', fontsize=11, fontweight='bold')
    ax4.set_title('PREFILL: Time vs Batch Size', fontsize=12, fontweight='bold',
                  color='#8B4513')
    ax4.set_xscale('log', base=2)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Prefill Prediction Accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue
        batch_sizes = [d['batch_size'] for d in data]
        hit_rates = [d['prefill_hit_rate_mean'] * 100 for d in data]
        errors = [d['prefill_hit_rate_std'] * 100 for d in data]
        ax5.errorbar(batch_sizes, hit_rates, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax5.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Prefill Prediction Accuracy (%)', fontsize=11, fontweight='bold')
    ax5.set_title('PREFILL: Prediction Accuracy vs Batch Size', fontsize=12, fontweight='bold',
                  color='#8B4513')
    ax5.set_xscale('log', base=2)
    ax5.set_ylim([0, 105])
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Prefill Speedup vs Baseline
    ax6 = fig.add_subplot(gs[1, 2])
    baseline_prefill_by_batch = {d['batch_size']: d['prefill_time_mean'] for d in baseline_data}

    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue
        batch_sizes = [d['batch_size'] for d in data]
        speedups = [baseline_prefill_by_batch.get(d['batch_size'], 1.0) / d['prefill_time_mean']
                   if d['batch_size'] in baseline_prefill_by_batch else 1.0 for d in data]
        ax6.plot(batch_sizes, speedups, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax6.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Speedup vs Baseline', fontsize=11, fontweight='bold')
    ax6.set_title('PREFILL: Speedup vs Baseline', fontsize=12, fontweight='bold',
                  color='#8B4513')
    ax6.set_xscale('log', base=2)
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # ============================================================================
    # ROW 3: DECODE PHASE METRICS
    # ============================================================================

    # Plot 7: Decode Time
    ax7 = fig.add_subplot(gs[2, 0])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        times = [d['decode_time_mean'] for d in data]
        errors = [d['decode_time_std'] for d in data]
        ax7.errorbar(batch_sizes, times, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax7.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Decode Time (s)', fontsize=11, fontweight='bold')
    ax7.set_title('DECODE: Time vs Batch Size', fontsize=12, fontweight='bold',
                  color='#006400')
    ax7.set_xscale('log', base=2)
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # Plot 8: Decode Prediction Accuracy
    ax8 = fig.add_subplot(gs[2, 1])
    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue
        batch_sizes = [d['batch_size'] for d in data]
        hit_rates = [d['decode_hit_rate_mean'] * 100 for d in data]
        errors = [d['decode_hit_rate_std'] * 100 for d in data]
        ax8.errorbar(batch_sizes, hit_rates, yerr=errors, fmt='o-', label=config_name,
                    color=colors.get(config_name, 'gray'), linewidth=2, markersize=8, capsize=4)
    ax8.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Decode Prediction Accuracy (%)', fontsize=11, fontweight='bold')
    ax8.set_title('DECODE: Prediction Accuracy vs Batch Size', fontsize=12, fontweight='bold',
                  color='#006400')
    ax8.set_xscale('log', base=2)
    ax8.set_ylim([0, 105])
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # Plot 9: Decode Speedup vs Baseline
    ax9 = fig.add_subplot(gs[2, 2])
    baseline_decode_by_batch = {d['batch_size']: d['decode_time_mean'] for d in baseline_data}

    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue
        batch_sizes = [d['batch_size'] for d in data]
        speedups = [baseline_decode_by_batch.get(d['batch_size'], 1.0) / d['decode_time_mean']
                   if d['batch_size'] in baseline_decode_by_batch else 1.0 for d in data]
        ax9.plot(batch_sizes, speedups, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax9.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Speedup vs Baseline', fontsize=11, fontweight='bold')
    ax9.set_title('DECODE: Speedup vs Baseline', fontsize=12, fontweight='bold',
                  color='#006400')
    ax9.set_xscale('log', base=2)
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    # ============================================================================
    # ROW 4: KEY COMPARISONS
    # ============================================================================

    # Plot 10: Fiddler vs Fiddler+Learned Direct Comparison
    ax10 = fig.add_subplot(gs[3, 0])
    fiddler_data = configs.get('Fiddler', [])
    learned_fiddler_data = configs.get('Fiddler+Learned-Prefetch', [])

    if fiddler_data and learned_fiddler_data:
        fiddler_bs = [d['batch_size'] for d in fiddler_data]
        fiddler_times = [d['total_time_mean'] for d in fiddler_data]
        fiddler_errs = [d['total_time_std'] for d in fiddler_data]

        lf_bs = [d['batch_size'] for d in learned_fiddler_data]
        lf_times = [d['total_time_mean'] for d in learned_fiddler_data]
        lf_errs = [d['total_time_std'] for d in learned_fiddler_data]

        ax10.errorbar(fiddler_bs, fiddler_times, yerr=fiddler_errs, fmt='o-',
                    label='Fiddler', color=colors['Fiddler'], linewidth=2, markersize=8, capsize=4)
        ax10.errorbar(lf_bs, lf_times, yerr=lf_errs, fmt='s-',
                    label='Fiddler+Learned', color=colors['Fiddler+Learned-Prefetch'],
                    linewidth=2, markersize=8, capsize=4)

    ax10.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax10.set_ylabel('Total Time (s)', fontsize=11, fontweight='bold')
    ax10.set_title('🎯 Fiddler vs Fiddler+Learned-Prefetch', fontsize=12, fontweight='bold')
    ax10.set_xscale('log', base=2)
    ax10.grid(True, alpha=0.3)
    ax10.legend()

    # Plot 11: Speedup of Fiddler+Learned vs Fiddler
    ax11 = fig.add_subplot(gs[3, 1])
    if fiddler_data and learned_fiddler_data:
        fiddler_by_batch = {d['batch_size']: d['total_time_mean'] for d in fiddler_data}
        lf_by_batch = {d['batch_size']: d['total_time_mean'] for d in learned_fiddler_data}

        common_bs = sorted(set(fiddler_by_batch.keys()) & set(lf_by_batch.keys()))
        speedups = [fiddler_by_batch[bs] / lf_by_batch[bs] for bs in common_bs]

        bars = ax11.bar(range(len(common_bs)), speedups,
                      color=['green' if s > 1.0 else 'red' for s in speedups],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax11.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
        ax11.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
        ax11.set_ylabel('Speedup (Fiddler+Learned / Fiddler)', fontsize=11, fontweight='bold')
        ax11.set_title('🏆 Fiddler+Learned Speedup vs Fiddler', fontsize=12, fontweight='bold')
        ax11.set_xticks(range(len(common_bs)))
        ax11.set_xticklabels(common_bs)
        ax11.grid(True, alpha=0.3, axis='y')

        # Annotate bars
        for i, (bs, val) in enumerate(zip(common_bs, speedups)):
            ax11.text(i, val + 0.02, f'{val:.3f}x', ha='center', va='bottom', fontweight='bold')

    # Plot 12: Configuration Comparison at Peak Batch Size
    ax12 = fig.add_subplot(gs[3, 2])
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

        bars = ax12.bar(range(len(config_names)), tok_s_values,
                      color=[colors.get(c, 'gray') for c in config_names],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax12.set_xticks(range(len(config_names)))
        ax12.set_xticklabels(config_names, rotation=15, ha='right')
        ax12.set_ylabel('Tokens/Second', fontsize=11, fontweight='bold')
        ax12.set_title(f'Throughput Comparison @ BS={max_common_bs}', fontsize=12, fontweight='bold')
        ax12.grid(True, alpha=0.3, axis='y')

        # Annotate bars
        for i, val in enumerate(tok_s_values):
            ax12.text(i, val + max(tok_s_values)*0.02, f'{val:.1f}',
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
    """Generate detailed analysis of results with prefill/decode separation."""

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
    baseline_data = configs.get('Baseline', [])

    analysis_report = []
    analysis_report.append("=" * 80)
    analysis_report.append("PHASE 5: KEY FINDINGS (PREFILL & DECODE SEPARATED)")
    analysis_report.append("=" * 80)
    analysis_report.append("")

    # ========================================================================
    # SECTION 0: Correctness Check Results
    # ========================================================================
    correctness_path = os.path.join(output_dir, 'correctness_check.json')
    if os.path.exists(correctness_path):
        with open(correctness_path, 'r') as f:
            correctness_data = json.load(f)

        analysis_report.append("✅ CORRECTNESS CHECK")
        analysis_report.append("=" * 80)
        analysis_report.append("")

        status = correctness_data.get('status', 'unknown')
        if status == 'pass':
            analysis_report.append("Status: ✅ PASSED - All configurations produce identical outputs")
        elif status == 'fail':
            analysis_report.append("Status: ⚠️  FAILED - Some configurations produce different outputs")
            comparisons = correctness_data.get('comparisons', {})
            if comparisons:
                analysis_report.append("")
                analysis_report.append("Mismatches detected:")
                for config_name, comparison in comparisons.items():
                    if comparison.get('status') == 'mismatch':
                        match_rate = comparison.get('match_rate', 0) * 100
                        reference = comparison.get('reference', 'unknown')
                        analysis_report.append(f"  - {config_name}: {match_rate:.1f}% match with {reference}")
        else:
            analysis_report.append(f"Status: {status.upper()}")

        analysis_report.append("")
        analysis_report.append("Reference: " + correctness_data.get('reference', 'N/A'))
        analysis_report.append("Configurations tested: " + ", ".join(correctness_data.get('configurations_tested', [])))
        analysis_report.append("")
        analysis_report.append("")

    # ========================================================================
    # SECTION 1: Overall Comparison
    # ========================================================================
    if fiddler_data and learned_fiddler_data:
        analysis_report.append("🎯 OVERALL: Fiddler+Learned-Prefetch vs Fiddler (Total Time)")
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

        analysis_report.append("")
        analysis_report.append("")

    # ========================================================================
    # SECTION 2: Prefill Phase Analysis
    # ========================================================================
    analysis_report.append("📊 PREFILL PHASE ANALYSIS")
    analysis_report.append("=" * 80)
    analysis_report.append("")

    if baseline_data:
        baseline_by_batch = {d['batch_size']: d for d in baseline_data}

        analysis_report.append(f"{'Config':<25} {'BS':<5} {'Time (s)':<22} {'Hit Rate':<12} {'Speedup vs Baseline':<20}")
        analysis_report.append("-" * 80)

        for config_name, data in configs.items():
            for d in data:
                bs = d['batch_size']
                prefill_time = d['prefill_time_mean']
                prefill_std = d['prefill_time_std']
                prefill_hr = d['prefill_hit_rate_mean'] * 100

                # Calculate speedup vs baseline
                if bs in baseline_by_batch and prefill_time > 0:
                    baseline_prefill = baseline_by_batch[bs]['prefill_time_mean']
                    speedup = baseline_prefill / prefill_time
                    speedup_str = f"{speedup:.3f}x"
                else:
                    speedup_str = "N/A"

                hr_str = f"{prefill_hr:.1f}%" if config_name != 'Baseline' else "N/A"

                analysis_report.append(
                    f"{config_name:<25} {bs:<5} {prefill_time:.3f}±{prefill_std:.3f}  "
                    f"{hr_str:<12} {speedup_str:<20}"
                )

        analysis_report.append("")
        analysis_report.append("")

    # ========================================================================
    # SECTION 3: Decode Phase Analysis
    # ========================================================================
    analysis_report.append("⚡ DECODE PHASE ANALYSIS")
    analysis_report.append("=" * 80)
    analysis_report.append("")

    if baseline_data:
        analysis_report.append(f"{'Config':<25} {'BS':<5} {'Time (s)':<27} {'Hit Rate':<12} {'Speedup vs Baseline':<20}")
        analysis_report.append("-" * 80)

        for config_name, data in configs.items():
            for d in data:
                bs = d['batch_size']
                decode_time = d['decode_time_mean']
                decode_std = d['decode_time_std']
                decode_per_token = d['decode_time_per_token_mean']
                decode_hr = d['decode_hit_rate_mean'] * 100

                # Calculate speedup vs baseline
                if bs in baseline_by_batch and decode_time > 0:
                    baseline_decode = baseline_by_batch[bs]['decode_time_mean']
                    speedup = baseline_decode / decode_time
                    speedup_str = f"{speedup:.3f}x"
                else:
                    speedup_str = "N/A"

                hr_str = f"{decode_hr:.1f}%" if config_name != 'Baseline' else "N/A"

                analysis_report.append(
                    f"{config_name:<25} {bs:<5} "
                    f"{decode_time:.3f}±{decode_std:.3f} ({decode_per_token:.3f}s/token)  "
                    f"{hr_str:<12} {speedup_str:<20}"
                )

        analysis_report.append("")
        analysis_report.append("")

    # ========================================================================
    # SECTION 4: Prediction Accuracy Analysis
    # ========================================================================
    analysis_report.append("🎯 PREDICTION ACCURACY ANALYSIS (PREFILL vs DECODE)")
    analysis_report.append("=" * 80)
    analysis_report.append("")
    analysis_report.append("Prediction Accuracy = (Experts prefetched that were used) / (Total experts prefetched)")
    analysis_report.append("Higher accuracy = Better predictions = Less wasted memory transfers")
    analysis_report.append("")

    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue

        analysis_report.append(f"{config_name}:")
        analysis_report.append(f"  {'BS':<5} {'Prefill Acc':<15} {'Decode Acc':<15} {'Notes':<30}")
        analysis_report.append("  " + "-" * 70)

        for d in data:
            bs = d['batch_size']
            prefill_hr = d['prefill_hit_rate_mean'] * 100
            decode_hr = d['decode_hit_rate_mean'] * 100

            # Add notes based on accuracy
            if decode_hr >= 80:
                note = "Excellent prediction"
            elif decode_hr >= 50:
                note = "Good prediction"
            elif decode_hr >= 30:
                note = "Moderate prediction"
            else:
                note = "Poor prediction"

            analysis_report.append(
                f"  {bs:<5} {prefill_hr:>6.1f}%        {decode_hr:>6.1f}%        {note:<30}"
            )

        analysis_report.append("")

    # ========================================================================
    # SECTION 5: Overall Summary Table
    # ========================================================================
    analysis_report.append("📈 COMPREHENSIVE SUMMARY (All Metrics)")
    analysis_report.append("=" * 80)
    analysis_report.append("")

    for config_name, data in configs.items():
        analysis_report.append(f"{config_name}:")
        for d in data:
            bs = d['batch_size']
            total = d['total_time_mean']
            total_std = d['total_time_std']
            prefill = d['prefill_time_mean']
            decode = d['decode_time_mean']
            decode_per_token = d['decode_time_per_token_mean']
            tok_s = d['tokens_per_second_mean']
            prefill_hr = d['prefill_hit_rate_mean'] * 100
            decode_hr = d['decode_hit_rate_mean'] * 100

            analysis_report.append(
                f"  BS={bs:2d}: Total={total:.3f}±{total_std:.3f}s | "
                f"Prefill={prefill:.3f}s | Decode={decode:.3f}s ({decode_per_token:.3f}s/token) | "
                f"{tok_s:.1f} tok/s"
            )
            if config_name != 'Baseline':
                analysis_report.append(
                    f"        Prefill Accuracy: {prefill_hr:.1f}% | Decode Accuracy: {decode_hr:.1f}%"
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


def run_correctness_check(configurations, output_dir):
    """
    Run quick correctness checks to ensure all configurations produce the same output.
    We validate both single-prompt (batch size 1) and multi-prompt batches to
    confirm that batched generation paths stay aligned across configurations.

    Args:
        configurations: List of configuration dicts
        output_dir: Directory to save correctness check results

    Returns:
        Dictionary with aggregated correctness outcomes across all scenarios
    """
    print("\n" + "="*80)
    print("CORRECTNESS CHECK")
    print("="*80)
    print("Verifying that all configurations produce identical outputs...")
    print("")

    # Scenario definitions keep prompts intentionally short so we can eyeball the outputs.
    scenarios = [
        {
            'name': 'single_prompt',
            'description': 'Single prompt (batch size = 1)',
            'prompts': ["The capital of France is"],
            'output_tokens': 10
        },
        {
            'name': 'batched_prompts',
            'description': 'Diverse prompts (batch size = 4)',
            'prompts': get_diverse_batch(4, seed=1234),
            'output_tokens': 10
        }
    ]

    scenario_results = {}
    scenario_statuses = []

    for scenario in scenarios:
        print("-" * 80)
        print(f"Scenario: {scenario['description']}")
        print(f"  Batch size: {len(scenario['prompts'])} | Output tokens: {scenario['output_tokens']}")
        results_by_config = {}

        for config in configurations:
            config_name = config['name']
            print(f"  Testing {config_name}...")

            model = None
            try:
                args = Args(use_fiddler_mode=config.get('use_fiddler_mode', False))

                if 'Learned' in config_name:
                    model_class = FiddlerQwenWithLearnedPrefetch
                else:
                    model_class = FiddlerQwen

                model = model_class(args, **config['kwargs'])

                generated_text = generate_text_for_correctness(
                    model,
                    scenario['prompts'],
                    scenario['output_tokens']
                )
                results_by_config[config_name] = {'generated_text': generated_text}

                preview = " | ".join(text[:60] for text in generated_text[:2])
                print(f"    Output preview: {preview}...")

            except Exception as e:
                print(f"    ❌ Error: {e}")
                results_by_config[config_name] = {'error': str(e)}
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()

        print("\n  Checking consistency across configurations...")
        correctness_result = check_correctness(results_by_config, scenario['prompts'])

        scenario_statuses.append(correctness_result['status'])
        scenario_results[scenario['name']] = {
            'status': correctness_result['status'],
            'reference': correctness_result.get('reference', 'N/A'),
            'configurations_tested': list(results_by_config.keys()),
            'test_prompts': scenario['prompts'],
            'test_output_tokens': scenario['output_tokens']
        }
        if 'comparisons' in correctness_result:
            scenario_results[scenario['name']]['comparisons'] = correctness_result['comparisons']

        print(f"\n  Status: {correctness_result['status'].upper()}")
        if correctness_result['status'] == 'pass':
            print("  ✅ All configurations produce identical outputs!")
        elif correctness_result['status'] == 'fail':
            print("  ⚠️  Warning: Some configurations produce different outputs")
            for config_name, comparison in correctness_result.get('comparisons', {}).items():
                if comparison.get('status') == 'mismatch':
                    match_rate = comparison.get('match_rate', 0)
                    print(f"     - {config_name}: {match_rate*100:.1f}% match with {comparison['reference']}")
        else:
            print(f"  ⏭️  {correctness_result.get('reason', 'Skipped')}")

    correctness_path = os.path.join(output_dir, 'correctness_check.json')
    if any(status == 'fail' for status in scenario_statuses):
        overall_status = 'fail'
    elif any(status == 'pass' for status in scenario_statuses):
        overall_status = 'pass'
    else:
        overall_status = scenario_statuses[0] if scenario_statuses else 'skipped'

    with open(correctness_path, 'w') as f:
        serializable_result = {
            'status': overall_status,
            'scenarios': scenario_results
        }
        json.dump(serializable_result, f, indent=2)

    print(f"\n  💾 Correctness check results saved to: {correctness_path}")
    print("="*80)

    return {
        'status': overall_status,
        'scenarios': scenario_results
    }


def main():
    """Main benchmark function."""
    print("\n" + "="*80)
    print("PHASE 5: PREDICTION METHODS BENCHMARK")
    print("="*80)
    print("\nObjectives:")
    print("  1. Find configurations where Fiddler+Learned-Prefetch > Fiddler alone")
    print("  2. Test with diverse sentences (different for each batch element)")
    print("  3. Multiple trials for statistical reliability")
    print("  4. Check correctness of outputs across configurations")
    print("  5. Track prediction accuracy for prefill and decode phases separately")
    print("="*80)

    # Configuration
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
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

    # Run correctness check first
    correctness_result = run_correctness_check(configurations, output_dir)

    # Check if correctness test passed
    if correctness_result['status'] == 'fail':
        print("\n⚠️  WARNING: Correctness check detected differences between configurations!")
        print("   Proceeding with benchmark, but results should be interpreted carefully.")
        print("   Check correctness_check.json for details.")

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
