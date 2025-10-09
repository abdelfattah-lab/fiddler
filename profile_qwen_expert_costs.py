#!/usr/bin/env python3
"""
Profile Qwen Expert Costs for Fiddler's Cost Model

This script measures:
1. latency_cpu: Cost per token for CPU execution (slope of linear fit)
2. latency_gpu: Constant cost for GPU transfer + execution

These parameters are used by Fiddler's cost model to decide which experts
to run on CPU vs GPU to minimize critical path latency.
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json


def setup_model():
    """Load Qwen model for profiling."""
    print("Loading Qwen1.5-MoE-A2.7B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B",
        torch_dtype=torch.bfloat16,
        device_map=None  # Manual device placement
    )

    # Move model to GPU initially
    model = model.to('cuda')

    # Find MoE layers
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):  # Has router gate
            moe_layers.append(i)

    print(f"Found {len(moe_layers)} MoE layers: {moe_layers}")
    return model, moe_layers


def profile_cpu_execution(model, moe_layers, num_trials=5):
    """
    Profile CPU execution cost as a function of token count.
    Returns latency_cpu (ms per token).
    """
    print("\n" + "="*80)
    print("PROFILING CPU EXECUTION COST")
    print("="*80)

    # Token counts to test
    token_counts = [1, 10, 50, 100, 200, 500, 1000]

    # Use middle MoE layer for profiling
    layer_idx = moe_layers[len(moe_layers)//2]
    expert_idx = 0  # Use first expert

    print(f"Using layer {layer_idx}, expert {expert_idx}")
    print(f"Token counts: {token_counts}")

    moe_layer = model.model.layers[layer_idx].mlp
    expert = moe_layer.experts[expert_idx]

    # Move expert to CPU
    expert = expert.cpu()

    # Get expert dimensions
    hidden_dim = expert.gate_proj.weight.shape[1]

    results = []

    for num_tokens in token_counts:
        # Create random input on CPU
        input_tensor = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device='cpu')

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = expert(input_tensor)

        # Measure
        timings = []
        for trial in range(num_trials):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            with torch.no_grad():
                output = expert(input_tensor)

            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            timings.append(elapsed_ms)

        avg_time = np.mean(timings)
        std_time = np.std(timings)

        results.append({
            'num_tokens': num_tokens,
            'time_ms': avg_time,
            'time_std': std_time,
            'ms_per_token': avg_time / num_tokens
        })

        print(f"  {num_tokens:4d} tokens: {avg_time:7.3f} ± {std_time:5.3f} ms ({avg_time/num_tokens:6.3f} ms/token)")

    # Fit linear model: time = latency_cpu * num_tokens
    x = np.array([r['num_tokens'] for r in results])
    y = np.array([r['time_ms'] for r in results])

    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)

    print(f"\n📊 Linear fit: time = {slope:.4f} * num_tokens + {intercept:.4f}")
    print(f"   latency_cpu = {slope:.4f} ms/token")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Measured', s=100, alpha=0.7)
    plt.plot(x, slope * x + intercept, 'r--', label=f'Fit: {slope:.4f}*x + {intercept:.2f}')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Execution Time (ms)')
    plt.title('CPU Expert Execution Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cpu_execution_profile.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved plot to cpu_execution_profile.png")

    return slope, results


def profile_gpu_transfer_and_execution(model, moe_layers, num_trials=10):
    """
    Profile GPU transfer + execution cost.
    Returns latency_gpu (ms constant cost).
    """
    print("\n" + "="*80)
    print("PROFILING GPU TRANSFER + EXECUTION COST")
    print("="*80)

    # Token counts to test (GPU cost should be roughly constant, dominated by transfer)
    token_counts = [1, 10, 50, 100, 200, 500, 1000]

    # Use middle MoE layer for profiling
    layer_idx = moe_layers[len(moe_layers)//2]
    expert_idx = 0

    print(f"Using layer {layer_idx}, expert {expert_idx}")
    print(f"Token counts: {token_counts}")

    moe_layer = model.model.layers[layer_idx].mlp
    cpu_expert = moe_layer.experts[expert_idx]

    # Move expert to CPU (pinned memory for faster transfer)
    cpu_expert = cpu_expert.cpu()

    # Pin memory
    for param in cpu_expert.parameters():
        if param.device.type == 'cpu':
            pinned_param = torch.empty_like(param, pin_memory=True)
            pinned_param.copy_(param)
            param.data = pinned_param

    # Create GPU buffer
    gpu_expert = type(cpu_expert)(cpu_expert.gate_proj.weight.shape[1], cpu_expert.up_proj.weight.shape[0])
    gpu_expert = gpu_expert.to('cuda', dtype=torch.bfloat16)

    # Get expert dimensions
    hidden_dim = cpu_expert.gate_proj.weight.shape[1]

    results = []

    for num_tokens in token_counts:
        # Create input on GPU
        input_tensor = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device='cuda')

        # Warmup
        for _ in range(3):
            # Load expert from CPU to GPU
            state_dict = cpu_expert.state_dict()
            gpu_expert.load_state_dict(state_dict)
            torch.cuda.synchronize()

            # Execute on GPU
            with torch.no_grad():
                _ = gpu_expert(input_tensor)
            torch.cuda.synchronize()

        # Measure transfer + execution
        timings = []
        for trial in range(num_trials):
            torch.cuda.synchronize()
            start = time.perf_counter()

            # Transfer expert from CPU to GPU
            state_dict = cpu_expert.state_dict()
            for name, param in gpu_expert.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name])

            # Execute on GPU
            with torch.no_grad():
                output = gpu_expert(input_tensor)

            torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            timings.append(elapsed_ms)

        avg_time = np.mean(timings)
        std_time = np.std(timings)

        results.append({
            'num_tokens': num_tokens,
            'time_ms': avg_time,
            'time_std': std_time
        })

        print(f"  {num_tokens:4d} tokens: {avg_time:7.3f} ± {std_time:5.3f} ms")

    # GPU cost should be roughly constant (dominated by transfer overhead)
    # Use mean of all measurements
    latency_gpu = np.mean([r['time_ms'] for r in results])
    latency_gpu_std = np.std([r['time_ms'] for r in results])

    print(f"\n📊 GPU transfer + execution cost:")
    print(f"   latency_gpu = {latency_gpu:.4f} ± {latency_gpu_std:.4f} ms (constant)")

    # Plot
    x = np.array([r['num_tokens'] for r in results])
    y = np.array([r['time_ms'] for r in results])

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Measured', s=100, alpha=0.7)
    plt.axhline(y=latency_gpu, color='r', linestyle='--', label=f'Mean: {latency_gpu:.2f} ms')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Transfer + Execution Time (ms)')
    plt.title('GPU Expert Transfer + Execution Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gpu_transfer_profile.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved plot to gpu_transfer_profile.png")

    return latency_gpu, results


def profile_multiple_layers(model, moe_layers, num_trials=5):
    """
    Profile multiple layers to check if costs are consistent.
    """
    print("\n" + "="*80)
    print("PROFILING MULTIPLE LAYERS")
    print("="*80)

    # Test first, middle, and last MoE layers
    test_layers = [
        moe_layers[0],
        moe_layers[len(moe_layers)//2],
        moe_layers[-1]
    ]

    cpu_latencies = []
    gpu_latencies = []

    for layer_idx in test_layers:
        print(f"\n--- Layer {layer_idx} ---")

        moe_layer = model.model.layers[layer_idx].mlp
        expert = moe_layer.experts[0]  # Test first expert

        # CPU profiling (100 tokens)
        expert_cpu = expert.cpu()
        hidden_dim = expert_cpu.gate_proj.weight.shape[1]
        input_cpu = torch.randn(100, hidden_dim, dtype=torch.bfloat16, device='cpu')

        timings_cpu = []
        for _ in range(num_trials):
            start = time.perf_counter()
            with torch.no_grad():
                _ = expert_cpu(input_cpu)
            end = time.perf_counter()
            timings_cpu.append((end - start) * 1000)

        cpu_latency = np.mean(timings_cpu) / 100  # ms per token
        cpu_latencies.append(cpu_latency)
        print(f"  CPU: {cpu_latency:.4f} ms/token")

        # GPU profiling
        expert_gpu = expert.to('cuda', dtype=torch.bfloat16)
        input_gpu = torch.randn(100, hidden_dim, dtype=torch.bfloat16, device='cuda')

        # Pin CPU memory
        for param in expert_cpu.parameters():
            if param.device.type == 'cpu':
                pinned_param = torch.empty_like(param, pin_memory=True)
                pinned_param.copy_(param)
                param.data = pinned_param

        timings_gpu = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start = time.perf_counter()

            # Transfer + execute
            state_dict = expert_cpu.state_dict()
            for name, param in expert_gpu.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name])

            with torch.no_grad():
                _ = expert_gpu(input_gpu)

            torch.cuda.synchronize()
            end = time.perf_counter()
            timings_gpu.append((end - start) * 1000)

        gpu_latency = np.mean(timings_gpu)
        gpu_latencies.append(gpu_latency)
        print(f"  GPU: {gpu_latency:.4f} ms (constant)")

    print(f"\n📊 Across layers:")
    print(f"   CPU latency: {np.mean(cpu_latencies):.4f} ± {np.std(cpu_latencies):.4f} ms/token")
    print(f"   GPU latency: {np.mean(gpu_latencies):.4f} ± {np.std(gpu_latencies):.4f} ms (constant)")

    return cpu_latencies, gpu_latencies


def main():
    parser = argparse.ArgumentParser(description='Profile Qwen expert costs for Fiddler')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials per measurement')
    args = parser.parse_args()

    print("="*80)
    print("QWEN EXPERT COST PROFILING FOR FIDDLER")
    print("="*80)

    # Setup model
    model, moe_layers = setup_model()

    # Profile CPU execution
    latency_cpu, cpu_results = profile_cpu_execution(model, moe_layers, num_trials=args.trials)

    # Profile GPU transfer + execution
    latency_gpu, gpu_results = profile_gpu_transfer_and_execution(model, moe_layers, num_trials=args.trials)

    # Profile multiple layers for consistency check
    cpu_latencies, gpu_latencies = profile_multiple_layers(model, moe_layers, num_trials=args.trials)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    avg_cpu = np.mean(cpu_latencies)
    avg_gpu = np.mean(gpu_latencies)

    print(f"\n📊 Recommended Cost Model Parameters:")
    print(f"   latency_cpu = {latency_cpu:.4f} ms/token (from linear fit)")
    print(f"   latency_gpu = {latency_gpu:.4f} ms (constant transfer overhead)")
    print(f"\n📊 Consistency check across layers:")
    print(f"   CPU: {avg_cpu:.4f} ± {np.std(cpu_latencies):.4f} ms/token")
    print(f"   GPU: {avg_gpu:.4f} ± {np.std(gpu_latencies):.4f} ms")

    # Comparison with Mixtral values
    print(f"\n📊 Comparison with Mixtral (from paper):")
    print(f"   Mixtral latency_cpu: 7 ms/token")
    print(f"   Mixtral latency_gpu: 70 ms")
    print(f"   Qwen latency_cpu: {latency_cpu:.4f} ms/token ({latency_cpu/7:.2f}x Mixtral)")
    print(f"   Qwen latency_gpu: {latency_gpu:.4f} ms ({latency_gpu/70:.2f}x Mixtral)")

    # Cost model decision boundary
    print(f"\n📊 Cost Model Decision Boundary:")
    print(f"   For an expert to be worth running on CPU:")
    print(f"   num_tokens * {latency_cpu:.4f} < {latency_gpu:.4f}")
    print(f"   num_tokens < {latency_gpu/latency_cpu:.1f}")
    print(f"   → Run on CPU if expert handles < {int(latency_gpu/latency_cpu)} tokens")

    # Save results
    results = {
        'latency_cpu': float(latency_cpu),
        'latency_gpu': float(latency_gpu),
        'cpu_latencies_by_layer': [float(x) for x in cpu_latencies],
        'gpu_latencies_by_layer': [float(x) for x in gpu_latencies],
        'cpu_detailed_results': cpu_results,
        'gpu_detailed_results': gpu_results,
        'recommendation': {
            'latency_cpu': float(latency_cpu),
            'latency_gpu': float(latency_gpu),
            'decision_boundary_tokens': float(latency_gpu / latency_cpu)
        }
    }

    with open('qwen_expert_cost_profile.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to qwen_expert_cost_profile.json")
    print(f"💾 Plots saved to cpu_execution_profile.png and gpu_transfer_profile.png")

    print("\n✅ Profiling complete!")


if __name__ == '__main__':
    main()
