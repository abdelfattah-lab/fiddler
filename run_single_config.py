#!/usr/bin/env python3
"""Run a single benchmark configuration and save results."""
import argparse
import json
import sys
import numpy as np
import torch

sys.path.insert(0, "src")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["baseline", "oracle", "prefetch"])
    parser.add_argument("--prefetch-count", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    class ModelArgs:
        def __init__(self):
            self.model = "Qwen/Qwen1.5-MoE-A2.7B"
            self.cpu_offload = 0
            self.max_experts_gpu = 0
            self.beam_width = 1

    # Load appropriate model
    if args.config == "baseline":
        from fiddler.qwen import FiddlerQwen
        model = FiddlerQwen(ModelArgs())
        config_name = "Baseline (On-Demand)"
    elif args.config == "oracle":
        from fiddler.qwen_oracle import FiddlerQwenOracle
        model = FiddlerQwenOracle(ModelArgs())
        config_name = "Oracle (All Experts)"
    else:  # prefetch
        from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch
        model = FiddlerQwenWithPrefetch(ModelArgs(), num_experts_to_prefetch=args.prefetch_count)

        # Collect patterns if needed
        if model.collection_mode:
            print(f"📊 Collecting expert usage patterns for {args.prefetch_count} experts...")
            model.generate(args.prompt, output_token=args.tokens)
            del model
            torch.cuda.empty_cache()
            model = FiddlerQwenWithPrefetch(ModelArgs(), num_experts_to_prefetch=args.prefetch_count)

        config_name = f"Prefetch-{args.prefetch_count} (Pattern)"

    # Run trials
    metrics = []
    for trial in range(args.trials):
        prefill, decode, ph, dh = model.generate(args.prompt, output_token=args.tokens)
        torch.cuda.synchronize()
        metrics.append({
            "prefill_time": float(prefill),
            "decode_time": float(decode),
            "prefill_hit_rate": float(ph),
            "decode_hit_rate": float(dh),
        })

    # Aggregate
    prefill = np.array([m["prefill_time"] for m in metrics])
    decode = np.array([m["decode_time"] for m in metrics])
    total = prefill + decode
    prefill_hit = np.array([m["prefill_hit_rate"] for m in metrics])
    decode_hit = np.array([m["decode_hit_rate"] for m in metrics])

    result = {
        "config": config_name,
        "tokens": args.tokens,
        "batch_size": 1,
        "trials": args.trials,
        "prefill_time_mean": float(prefill.mean()),
        "prefill_time_std": float(prefill.std(ddof=0)),
        "decode_time_mean": float(decode.mean()),
        "decode_time_std": float(decode.std(ddof=0)),
        "total_time_mean": float(total.mean()),
        "total_time_std": float(total.std(ddof=0)),
        "prefill_hit_rate_mean": float(prefill_hit.mean()),
        "prefill_hit_rate_std": float(prefill_hit.std(ddof=0)),
        "decode_hit_rate_mean": float(decode_hit.mean()),
        "decode_hit_rate_std": float(decode_hit.std(ddof=0)),
    }

    with open(args.output, "w") as f:
        json.dump([result], f, indent=2)

    print(f"✅ {config_name}: {total.mean():.3f}s (prefill: {prefill.mean():.3f}s, decode: {decode.mean():.3f}s)")
    if args.config == "prefetch":
        print(f"   Hit rate: {decode_hit.mean():.1%}")

if __name__ == "__main__":
    main()
