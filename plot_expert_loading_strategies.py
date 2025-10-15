#!/usr/bin/env python3
"""Benchmark and plot expert-loading strategies for Qwen MoE."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure src modules are importable when the script is executed from repository root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from fiddler.qwen import FiddlerQwen  # noqa: E402  (import after sys.path tweak)
from fiddler.qwen_oracle import FiddlerQwenOracle  # noqa: E402
from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch  # noqa: E402


class Args:
    """Minimal argument container expected by the model classes."""

    def __init__(self, model_name: str):
        self.model = model_name
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1


DEFAULT_PROMPTS = [
    "The theory of general relativity was developed by Albert Einstein in",
    "Artificial intelligence systems use machine learning algorithms to",
    "Quantum computers utilize the principles of superposition and",
    "CRISPR gene editing technology allows scientists to modify",
    "The human brain contains approximately 86 billion neurons that",
    "The speed of light in vacuum is approximately 299,792,458",
    "Neural networks are inspired by biological neurons and use",
    "The periodic table organizes chemical elements based on their",
    "The Roman Empire reached its greatest extent under Emperor",
    "The Great Wall of China was built over many centuries to",
    "The Renaissance began in Italy during the 14th century and",
    "Mount Everest is the highest mountain on Earth with a",
    "The Industrial Revolution started in Britain in the late",
    "Ancient Egypt was one of the world's first civilizations along",
    "The Amazon rainforest is the largest tropical rainforest covering",
    "World War II began in 1939 when Germany invaded",
    "Leonardo da Vinci painted the Mona Lisa during the",
    "Shakespeare wrote 37 plays and 154 sonnets during his",
    "Jazz music originated in New Orleans at the turn",
    "The Taj Mahal was built in India by Emperor",
    "The Impressionist movement in art began in France in",
    "The Olympic Games were first held in ancient Greece",
    "Classical music composers like Mozart and Beethoven created",
    "The Great Pyramid of Giza was built as a",
    "Blue whales are the largest animals ever known to",
    "Photosynthesis is the process by which plants convert sunlight",
    "The African elephant is the largest land animal weighing",
    "Coral reefs are diverse underwater ecosystems formed by",
    "The Aurora Borealis occurs when solar particles interact with",
    "Monarch butterflies migrate thousands of miles each year from",
    "The Sahara Desert is the largest hot desert covering",
    "Coffee is one of the most popular beverages consumed",
    "Basketball was invented by James Naismith in Springfield",
    "Pizza originated in Naples, Italy and has become",
    "The bicycle is a human-powered vehicle with two wheels",
    "Solar panels convert sunlight into electricity using photovoltaic",
    "The Eiffel Tower was completed in 1889 for the",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen expert-loading strategies.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B",
                        help="Model identifier to benchmark.")
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 16, 32, 64],
                        help="List of output token counts to benchmark (per prompt).")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per configuration and token count.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for generation prompts.")
    parser.add_argument("--prefetch-counts", type=int, nargs="+", default=[2, 4, 8, 16],
                        help="Prefetch expert counts to evaluate.")
    parser.add_argument("--prompt", type=str,
                        default="The capital of France is",
                        help="Base prompt used when batch size is 1.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for prompt sampling.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to store raw results and plots (auto-generated if omitted).")
    parser.add_argument("--pattern-file", type=str, default="expert_usage_patterns_qwen.json",
                        help="Path to the pattern cache used by pattern-based prefetching.")
    parser.add_argument("--reuse-patterns", action="store_true",
                        help="Reuse existing pattern cache instead of recollecting it.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation (still saves CSV/JSON results).")
    return parser.parse_args()


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this benchmark.")


def prepare_output_dir(requested_dir: Optional[str]) -> str:
    if requested_dir:
        output_dir = requested_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"expert_loading_benchmark_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_prompt_schedule(tokens: List[int], trials: int, batch_size: int,
                          base_prompt: str, seed: int) -> Dict[int, List[List[str]]]:
    schedule: Dict[int, List[List[str]]] = {}
    rng = np.random.default_rng(seed)
    for count in sorted(tokens):
        schedule[count] = []
        for _ in range(trials):
            if batch_size == 1:
                schedule[count].append([base_prompt])
            else:
                replace = batch_size > len(DEFAULT_PROMPTS)
                selected = rng.choice(DEFAULT_PROMPTS, size=batch_size, replace=replace)
                schedule[count].append(selected.tolist())
    return schedule


def run_generation(model, prompts: List[str], output_tokens: int) -> Dict[str, float]:
    text_input = prompts[0] if len(prompts) == 1 else prompts
    prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
        text_input,
        output_token=output_tokens
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return {
        "prefill_time": float(prefill_time),
        "decode_time": float(decode_time),
        "prefill_hit_rate": float(prefill_hit_rate),
        "decode_hit_rate": float(decode_hit_rate),
    }


def aggregate_trials(trial_metrics: List[Dict[str, float]], tokens: int, batch_size: int) -> Dict[str, float]:
    prefill = np.array([m["prefill_time"] for m in trial_metrics], dtype=np.float64)
    decode = np.array([m["decode_time"] for m in trial_metrics], dtype=np.float64)
    total = prefill + decode

    throughput = (batch_size * tokens) / total
    decode_throughput = np.divide(batch_size * tokens, decode, out=np.zeros_like(decode), where=decode > 0)

    prefill_hit = np.array([m["prefill_hit_rate"] for m in trial_metrics], dtype=np.float64)
    decode_hit = np.array([m["decode_hit_rate"] for m in trial_metrics], dtype=np.float64)

    def mean_and_std(values: np.ndarray) -> Tuple[float, float]:
        return float(values.mean()), float(values.std(ddof=0))

    summary = {}
    summary["prefill_time_mean"], summary["prefill_time_std"] = mean_and_std(prefill)
    summary["decode_time_mean"], summary["decode_time_std"] = mean_and_std(decode)
    summary["total_time_mean"], summary["total_time_std"] = mean_and_std(total)
    summary["tokens_per_second_mean"], summary["tokens_per_second_std"] = mean_and_std(throughput)
    summary["decode_tokens_per_second_mean"], summary["decode_tokens_per_second_std"] = mean_and_std(decode_throughput)
    summary["prefill_hit_rate_mean"], summary["prefill_hit_rate_std"] = mean_and_std(prefill_hit)
    summary["decode_hit_rate_mean"], summary["decode_hit_rate_std"] = mean_and_std(decode_hit)
    return summary


def create_prefetch_model(model_name: str, num_experts: int, tokens: List[int],
                          prompt_schedule: Dict[int, List[List[str]]]) -> FiddlerQwenWithPrefetch:
    args = Args(model_name)
    model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=num_experts)
    max_tokens = max(tokens)
    collection_prompts = prompt_schedule[max_tokens][0]
    text_input = collection_prompts[0] if len(collection_prompts) == 1 else collection_prompts

    if model.collection_mode:
        print(f"📊 Collecting expert usage patterns (num_experts={num_experts})...")
        model.generate(text_input, output_token=max_tokens)
        del model
        torch.cuda.empty_cache()
        model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=num_experts)

    return model


def benchmark_configuration(label: str, style_key: str, model, tokens: List[int],
                             prompt_schedule: Dict[int, List[List[str]]],
                             batch_size: int, trials: int) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []

    warmup_tokens = max(tokens)
    warmup_prompts = prompt_schedule[warmup_tokens][0]
    run_generation(model, warmup_prompts, warmup_tokens)

    for token_count in sorted(tokens):
        metrics: List[Dict[str, float]] = []
        for trial_idx in range(trials):
            prompts = prompt_schedule[token_count][trial_idx]
            metrics.append(run_generation(model, prompts, token_count))
        aggregated = aggregate_trials(metrics, token_count, batch_size)
        aggregated.update({
            "config": label,
            "style_key": style_key,
            "tokens": token_count,
            "batch_size": batch_size,
            "trials": trials,
        })
        results.append(aggregated)

    del model
    torch.cuda.empty_cache()
    return results


def plot_results(results: List[Dict[str, float]], tokens: List[int], output_dir: str,
                 style_map: Dict[str, Dict[str, str]]) -> str:
    by_config: Dict[str, List[Dict[str, float]]] = {}
    for entry in results:
        by_config.setdefault(entry["config"], []).append(entry)

    plt.figure(figsize=(10, 6))
    for config_name, entries in by_config.items():
        entries = sorted(entries, key=lambda x: x["tokens"])
        times = [e["total_time_mean"] for e in entries]
        x_vals = [e["tokens"] for e in entries]
        style = style_map.get(entries[0]["style_key"], {})
        plt.plot(
            x_vals,
            times,
            label=config_name,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
        )

    plt.xlabel("Output Tokens")
    plt.ylabel("Total Time (s)")
    plt.title("Qwen Expert Loading Strategies")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = os.path.join(output_dir, "expert_loading_strategies.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


def write_csv(results: List[Dict[str, float]], output_dir: str) -> str:
    fieldnames = [
        "config",
        "style_key",
        "tokens",
        "batch_size",
        "trials",
        "prefill_time_mean",
        "prefill_time_std",
        "decode_time_mean",
        "decode_time_std",
        "total_time_mean",
        "total_time_std",
        "tokens_per_second_mean",
        "tokens_per_second_std",
        "decode_tokens_per_second_mean",
        "decode_tokens_per_second_std",
        "prefill_hit_rate_mean",
        "prefill_hit_rate_std",
        "decode_hit_rate_mean",
        "decode_hit_rate_std",
    ]

    csv_path = os.path.join(output_dir, "expert_loading_results.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            writer.writerow({key: entry.get(key, "") for key in fieldnames})
    return csv_path


def write_json(results: List[Dict[str, float]], output_dir: str) -> str:
    json_path = os.path.join(output_dir, "expert_loading_results.json")
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2)
    return json_path


def main() -> int:
    args = parse_args()
    ensure_cuda()

    tokens = sorted(set(args.tokens))
    output_dir = prepare_output_dir(args.output_dir)
    prompt_schedule = build_prompt_schedule(tokens, args.trials, args.batch_size, args.prompt, args.seed)

    if not args.reuse_patterns and os.path.exists(args.pattern_file):
        print(f"🗑️  Removing existing pattern cache: {args.pattern_file}")
        os.remove(args.pattern_file)

    results: List[Dict[str, float]] = []

    style_map = {
        "baseline": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "oracle": {"color": "#2ca02c", "linestyle": "-", "marker": "s"},
    }

    print("\n=== Baseline (On-Demand) ===")
    baseline_model = FiddlerQwen(Args(args.model))
    results.extend(benchmark_configuration(
        "Baseline (On-Demand)",
        "baseline",
        baseline_model,
        tokens,
        prompt_schedule,
        args.batch_size,
        args.trials,
    ))

    print("\n=== Oracle (All Experts on GPU) ===")
    oracle_model = FiddlerQwenOracle(Args(args.model))
    results.extend(benchmark_configuration(
        "Oracle (All Experts)",
        "oracle",
        oracle_model,
        tokens,
        prompt_schedule,
        args.batch_size,
        args.trials,
    ))

    for idx, count in enumerate(sorted(set(args.prefetch_counts))):
        style_map[f"prefetch-{count}"] = {
            "color": plt.cm.tab10((idx + 2) % 10),
            "linestyle": "--",
            "marker": "o",
        }
        print(f"\n=== Prefetch (Pattern) — {count} experts ===")
        prefetch_model = create_prefetch_model(args.model, count, tokens, prompt_schedule)
        results.extend(benchmark_configuration(
            f"Prefetch-{count} (Pattern)",
            f"prefetch-{count}",
            prefetch_model,
            tokens,
            prompt_schedule,
            args.batch_size,
            args.trials,
        ))

    csv_path = write_csv(results, output_dir)
    json_path = write_json(results, output_dir)

    if not args.no_plot:
        plot_path = plot_results(results, tokens, output_dir, style_map)
        print(f"📊 Plot saved to: {plot_path}")
    else:
        plot_path = "(skipped)"

    print(f"📄 CSV saved to: {csv_path}")
    print(f"🗂️  JSON saved to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
