#!/usr/bin/env python3
"""
Benchmark script for Fiddler + Learned Prefetch with detailed per-layer expert usage logging.

This script instruments the model to capture:
- Which experts were prefetched in each layer
- Which experts were actually used (based on gating)
- Which prefetched experts were not used (wasted)
- Which experts were fetched on demand (cache miss)
- Which experts were on CPU vs GPU (for Fiddler mode)
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, 'src')

import torch
from fiddler.qwen_with_learned_prefetch import FiddlerQwenWithLearnedPrefetch


def add_detailed_logging(model):
    """
    Instrument the model to capture detailed expert usage at each layer.

    This wraps the _moe_forward_with_management method to log:
    - Layer index and token position
    - Selected experts (from gating)
    - Prefetched experts
    - GPU resident experts
    - CPU experts (if Fiddler mode enabled)
    - Cache hits vs misses
    """

    # Storage for logs
    model.detailed_expert_logs = []

    # Save original method
    original_moe_forward = model._moe_forward_with_management

    def logged_moe_forward(hidden_states, layer_idx):
        """Wrapped MoE forward that logs expert usage before calling original."""

        # Extract information before forward pass
        moe_layer = model.model.model.layers[layer_idx].mlp
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Get the gating decisions
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = moe_layer.gate(hidden_states_flat)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_top, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        # Get unique experts selected
        used_experts = selected_experts.flatten().unique().cpu().tolist()

        # Get current token position
        token_pos = model.profiler.current_token_pos

        # Get prefetched experts for this layer
        prefetched_experts = []
        if not model.collection_mode:
            predicted = model._predict_experts_for_layer(layer_idx, token_pos)
            prefetched_experts = predicted if predicted else []

        # Determine which experts are GPU-resident
        gpu_resident = []
        if layer_idx in model.gpu_resident_layers:
            # All experts are GPU-resident for these layers
            gpu_resident = list(range(model.n_expert))

        # Determine cache hits and misses
        cache_hits = []
        cache_misses = []
        for expert_idx in used_experts:
            if model._is_expert_prefetched(layer_idx, expert_idx) or expert_idx in gpu_resident:
                cache_hits.append(expert_idx)
            else:
                cache_misses.append(expert_idx)

        # Determine CPU vs GPU experts (for Fiddler mode)
        cpu_experts = []
        gpu_experts = []

        if model.enable_cpu_offload and layer_idx not in model.gpu_resident_layers:
            # Calculate which experts would go to CPU vs GPU
            # This is done by the partitioning algorithm in the original forward
            # For logging purposes, we'll approximate it

            # Count tokens per expert
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=moe_layer.num_experts).permute(2, 1, 0)
            expert_token_counts = {}

            for expert_idx in used_experts:
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                expert_token_counts[expert_idx] = len(top_x)

            # Calculate costs
            cost_per_expert = model._calculate_expert_costs(expert_token_counts, layer_idx)
            cpu_list, gpu_list = model._partition_experts_greedy(cost_per_expert)

            cpu_experts = cpu_list
            gpu_experts = gpu_list
        else:
            # All experts go to GPU in non-Fiddler mode or GPU-resident layers
            gpu_experts = used_experts

        # Calculate derived metrics
        wasted_prefetch = list(set(prefetched_experts) - set(used_experts))
        properly_prefetched = list(set(prefetched_experts) & set(used_experts))

        # Determine phase
        is_prefill = sequence_length > 1
        phase = "prefill" if is_prefill else "decode"

        # Create log entry
        log_entry = {
            'layer_idx': layer_idx,
            'token_pos': token_pos,
            'phase': phase,
            'used_experts': sorted(used_experts),
            'prefetched_experts': sorted(prefetched_experts),
            'properly_prefetched': sorted(properly_prefetched),
            'wasted_prefetch': sorted(wasted_prefetch),
            'cache_hits': sorted(cache_hits),
            'cache_misses': sorted(cache_misses),
            'gpu_resident_experts': sorted(gpu_resident),
            'cpu_experts': sorted(cpu_experts),
            'gpu_experts': sorted(gpu_experts),
            'expert_token_counts': {str(k): v for k, v in expert_token_counts.items()} if 'expert_token_counts' in locals() else {},
            'stats': {
                'total_used': len(used_experts),
                'total_prefetched': len(prefetched_experts),
                'properly_prefetched': len(properly_prefetched),
                'wasted': len(wasted_prefetch),
                'hits': len(cache_hits),
                'misses': len(cache_misses),
                'on_cpu': len(cpu_experts),
                'on_gpu': len(gpu_experts)
            }
        }

        # Store log
        model.detailed_expert_logs.append(log_entry)

        # Call original forward
        return original_moe_forward(hidden_states, layer_idx)

    # Replace the method
    model._moe_forward_with_management = logged_moe_forward

    return model


def print_detailed_logs(logs, compact=False):
    """Print detailed expert usage logs."""

    print("\n" + "="*100)
    print("DETAILED EXPERT USAGE LOGS")
    print("="*100)

    for log in logs:
        if compact and log['stats']['total_used'] == 0:
            continue  # Skip empty logs in compact mode

        print(f"\n{'─'*100}")
        print(f"[Layer {log['layer_idx']:2d}] Token Position: {log['token_pos']:3d} | Phase: {log['phase'].upper():7s}")
        print(f"{'─'*100}")

        print(f"  📊 Used Experts ({len(log['used_experts'])}):")
        print(f"     {log['used_experts']}")

        if log['prefetched_experts']:
            print(f"\n  🔮 Prefetched ({len(log['prefetched_experts'])}):")
            print(f"     {log['prefetched_experts']}")

        if log['properly_prefetched']:
            print(f"  ✓ Properly Prefetched ({len(log['properly_prefetched'])}):")
            print(f"     {log['properly_prefetched']}")

        if log['wasted_prefetch']:
            print(f"  ✗ Wasted Prefetch ({len(log['wasted_prefetch'])}):")
            print(f"     {log['wasted_prefetch']}")

        print(f"\n  💾 Cache Performance:")
        print(f"     Hits   ({len(log['cache_hits']):2d}): {log['cache_hits']}")
        print(f"     Misses ({len(log['cache_misses']):2d}): {log['cache_misses']}")

        if log['cpu_experts']:
            print(f"\n  🖥️  Device Distribution:")
            print(f"     CPU ({len(log['cpu_experts']):2d}): {log['cpu_experts']}")
            print(f"     GPU ({len(log['gpu_experts']):2d}): {log['gpu_experts']}")

        if log['expert_token_counts']:
            print(f"\n  🎯 Token Distribution:")
            for exp_id, count in sorted(log['expert_token_counts'].items(), key=lambda x: int(x[0])):
                print(f"     Expert {exp_id:2s}: {count:3d} tokens")

        # Summary stats
        prefetch_efficiency = (log['stats']['properly_prefetched'] / log['stats']['total_prefetched'] * 100) \
            if log['stats']['total_prefetched'] > 0 else 0
        hit_rate = (log['stats']['hits'] / log['stats']['total_used'] * 100) \
            if log['stats']['total_used'] > 0 else 0

        print(f"\n  📈 Summary:")
        print(f"     Prefetch Efficiency: {prefetch_efficiency:.1f}% ({log['stats']['properly_prefetched']}/{log['stats']['total_prefetched']})")
        print(f"     Cache Hit Rate:      {hit_rate:.1f}% ({log['stats']['hits']}/{log['stats']['total_used']})")


def generate_summary_stats(logs):
    """Generate summary statistics from logs."""

    if not logs:
        return {}

    # Separate by phase
    prefill_logs = [l for l in logs if l['phase'] == 'prefill']
    decode_logs = [l for l in logs if l['phase'] == 'decode']

    def calc_stats(log_list):
        if not log_list:
            return {}

        total_used = sum(l['stats']['total_used'] for l in log_list)
        total_prefetched = sum(l['stats']['total_prefetched'] for l in log_list)
        total_properly_prefetched = sum(l['stats']['properly_prefetched'] for l in log_list)
        total_wasted = sum(l['stats']['wasted'] for l in log_list)
        total_hits = sum(l['stats']['hits'] for l in log_list)
        total_misses = sum(l['stats']['misses'] for l in log_list)
        total_cpu = sum(l['stats']['on_cpu'] for l in log_list)
        total_gpu = sum(l['stats']['on_gpu'] for l in log_list)

        return {
            'num_layers': len(log_list),
            'total_experts_used': total_used,
            'total_prefetched': total_prefetched,
            'properly_prefetched': total_properly_prefetched,
            'wasted_prefetch': total_wasted,
            'prefetch_efficiency': (total_properly_prefetched / total_prefetched * 100) if total_prefetched > 0 else 0,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0,
            'cpu_experts': total_cpu,
            'gpu_experts': total_gpu,
            'avg_experts_per_layer': total_used / len(log_list)
        }

    return {
        'overall': calc_stats(logs),
        'prefill': calc_stats(prefill_logs),
        'decode': calc_stats(decode_logs)
    }


def print_summary(summary):
    """Print summary statistics."""

    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    for phase_name, stats in summary.items():
        if not stats:
            continue

        print(f"\n{phase_name.upper()}:")
        print(f"  Layers Processed:      {stats['num_layers']}")
        print(f"  Total Experts Used:    {stats['total_experts_used']}")
        print(f"  Avg Experts/Layer:     {stats['avg_experts_per_layer']:.1f}")

        if stats['total_prefetched'] > 0:
            print(f"\n  Prefetch Performance:")
            print(f"    Total Prefetched:     {stats['total_prefetched']}")
            print(f"    Properly Prefetched:  {stats['properly_prefetched']}")
            print(f"    Wasted:               {stats['wasted_prefetch']}")
            print(f"    Efficiency:           {stats['prefetch_efficiency']:.1f}%")

        print(f"\n  Cache Performance:")
        print(f"    Hits:                 {stats['total_hits']}")
        print(f"    Misses:               {stats['total_misses']}")
        print(f"    Hit Rate:             {stats['hit_rate']:.1f}%")

        if stats['cpu_experts'] > 0 or stats['gpu_experts'] > 0:
            print(f"\n  Device Distribution:")
            print(f"    CPU Experts:          {stats['cpu_experts']}")
            print(f"    GPU Experts:          {stats['gpu_experts']}")
            total_dev = stats['cpu_experts'] + stats['gpu_experts']
            if total_dev > 0:
                print(f"    CPU Percentage:       {stats['cpu_experts'] / total_dev * 100:.1f}%")


def main():
    """Run benchmark with detailed logging."""

    print("="*100)
    print("FIDDLER + LEARNED PREFETCH: DETAILED EXPERT USAGE LOGGING")
    print("="*100)

    # Configuration
    class Args:
        def __init__(self):
            self.model = "Qwen/Qwen1.5-MoE-A2.7B"
            self.cpu_offload = 0
            self.max_experts_gpu = 0
            self.beam_width = 1

    args = Args()

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Machine learning is a branch of",
    ]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"detailed_expert_logs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📁 Output Directory: {output_dir}/")
    print(f"\nConfiguration:")
    print(f"  Model: Fiddler + Learned Prefetch")
    print(f"  CPU Offload: Enabled")
    print(f"  Experts to Prefetch: 8")
    print(f"  Predictor: predictor_checkpoints/best_model.pt")
    print(f"  Output Tokens: 20 per prompt")
    print(f"  Test Prompts: {len(test_prompts)}")

    # Load model
    print("\n" + "─"*100)
    print("Loading model...")
    print("─"*100)

    model = FiddlerQwenWithLearnedPrefetch(
        args,
        num_experts_to_prefetch=8,
        enable_cpu_offload=True,
        predictor_path='predictor_checkpoints/best_model.pt'
    )

    # Add detailed logging instrumentation
    print("\n" + "─"*100)
    print("Adding detailed logging instrumentation...")
    print("─"*100)
    add_detailed_logging(model)

    # Run benchmark
    print("\n" + "─"*100)
    print("Running benchmark...")
    print("─"*100)

    all_results = []

    for i, prompt in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] Processing: '{prompt}'")
        print("─"*60)

        # Clear logs for this prompt
        model.detailed_expert_logs = []

        # Generate
        start_time = time.time()
        prefill_time, decode_time, prefill_hr, decode_hr = model.generate(
            text=prompt,
            output_token=20
        )
        total_time = time.time() - start_time

        # Get stats
        stats = model.get_prefetch_stats()

        # Store results
        result = {
            'prompt_idx': i,
            'prompt': prompt,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'prefill_hit_rate': prefill_hr * 100,
            'decode_hit_rate': decode_hr * 100,
            'stats': stats,
            'detailed_logs': model.detailed_expert_logs.copy()
        }

        all_results.append(result)

        print(f"\n✓ Completed:")
        print(f"  Prefill:  {prefill_time:.3f}s (Hit Rate: {prefill_hr*100:.1f}%)")
        print(f"  Decode:   {decode_time:.3f}s/token (Hit Rate: {decode_hr*100:.1f}%)")
        print(f"  Total:    {total_time:.3f}s")
        print(f"  Logged:   {len(model.detailed_expert_logs)} layer executions")

    # Save results
    results_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Print detailed logs for each prompt
    for i, result in enumerate(all_results):
        print(f"\n{'═'*100}")
        print(f"PROMPT {i+1}: \"{result['prompt']}\"")
        print(f"{'═'*100}")

        print_detailed_logs(result['detailed_logs'], compact=False)

        # Generate and print summary for this prompt
        summary = generate_summary_stats(result['detailed_logs'])
        print_summary(summary)

        # Save per-prompt detailed log
        prompt_log_file = os.path.join(output_dir, f'prompt_{i+1}_detailed_log.json')
        with open(prompt_log_file, 'w') as f:
            json.dump({
                'prompt': result['prompt'],
                'timing': {
                    'prefill_time': result['prefill_time'],
                    'decode_time': result['decode_time'],
                    'total_time': result['total_time']
                },
                'hit_rates': {
                    'prefill': result['prefill_hit_rate'],
                    'decode': result['decode_hit_rate']
                },
                'detailed_logs': result['detailed_logs'],
                'summary': summary
            }, f, indent=2)

        print(f"\n✓ Detailed log saved to: {prompt_log_file}")

    # Generate final report
    report_file = os.path.join(output_dir, 'README.md')
    with open(report_file, 'w') as f:
        f.write("# Detailed Expert Usage Logging Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write("- **Model**: Fiddler + Learned Prefetch\n")
        f.write("- **CPU Offload**: Enabled\n")
        f.write("- **Experts to Prefetch**: 8 (learned predictor)\n")
        f.write("- **Predictor**: predictor_checkpoints/best_model.pt\n")
        f.write("- **Output Tokens**: 20 per prompt\n\n")

        f.write("## Files in This Directory\n\n")
        f.write("- `benchmark_results.json` - Complete results for all prompts\n")
        for i in range(len(all_results)):
            f.write(f"- `prompt_{i+1}_detailed_log.json` - Detailed layer-by-layer logs for prompt {i+1}\n")
        f.write("- `README.md` - This file\n\n")

        f.write("## Summary\n\n")

        for i, result in enumerate(all_results):
            f.write(f"### Prompt {i+1}: \"{result['prompt']}\"\n\n")
            f.write(f"- **Prefill Time**: {result['prefill_time']:.3f}s (Hit Rate: {result['prefill_hit_rate']:.1f}%)\n")
            f.write(f"- **Decode Time**: {result['decode_time']:.3f}s/token (Hit Rate: {result['decode_hit_rate']:.1f}%)\n")
            f.write(f"- **Total Time**: {result['total_time']:.3f}s\n")
            f.write(f"- **Layers Logged**: {len(result['detailed_logs'])}\n\n")

            summary = generate_summary_stats(result['detailed_logs'])
            if 'decode' in summary and summary['decode']:
                f.write(f"**Decode Phase Statistics**:\n")
                f.write(f"- Prefetch Efficiency: {summary['decode']['prefetch_efficiency']:.1f}%\n")
                f.write(f"- Cache Hit Rate: {summary['decode']['hit_rate']:.1f}%\n")
                f.write(f"- Properly Prefetched: {summary['decode']['properly_prefetched']}/{summary['decode']['total_prefetched']}\n")
                f.write(f"- Cache Hits: {summary['decode']['total_hits']}/{summary['decode']['total_experts_used']}\n\n")

        f.write("## Log Format\n\n")
        f.write("Each layer execution is logged with:\n")
        f.write("- **Used Experts**: Experts selected by gating function\n")
        f.write("- **Prefetched Experts**: Experts that were prefetched by the predictor\n")
        f.write("- **Properly Prefetched**: Prefetched experts that were actually used\n")
        f.write("- **Wasted Prefetch**: Prefetched experts that were NOT used\n")
        f.write("- **Cache Hits**: Used experts that were readily available (prefetched or GPU-resident)\n")
        f.write("- **Cache Misses**: Used experts that had to be loaded on-demand\n")
        f.write("- **CPU/GPU Distribution**: Which experts ran on CPU vs GPU (Fiddler mode)\n")
        f.write("- **Token Distribution**: Number of tokens processed by each expert\n\n")

    print(f"\n✓ Report saved to: {report_file}")

    print("\n" + "="*100)
    print("BENCHMARK COMPLETE")
    print("="*100)
    print(f"\n📁 All logs saved to: {output_dir}/")
    print(f"\nFiles generated:")
    print(f"  - {results_file}")
    print(f"  - {report_file}")
    for i in range(len(all_results)):
        print(f"  - {os.path.join(output_dir, f'prompt_{i+1}_detailed_log.json')}")
    print("\n" + "="*100)

    return output_dir


if __name__ == '__main__':
    output_dir = main()
    print(f"\n✅ All detailed logs are in: {output_dir}/")
