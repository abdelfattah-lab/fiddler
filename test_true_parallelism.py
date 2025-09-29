#!/usr/bin/env python3
"""
Create a CUDA program that definitely achieves computation + memory transfer parallelism
for comparison with our prefetch implementation
"""

import torch
import time
import os

# Import NVTX for profiling markers
try:
    import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

def nvtx_range_push(name):
    if NVTX_AVAILABLE:
        try:
            nvtx.range_push(name)
        except AttributeError:
            # nvtx might use different API
            pass

def nvtx_range_pop():
    if NVTX_AVAILABLE:
        try:
            nvtx.range_pop()
        except AttributeError:
            pass

def create_parallel_workload():
    """Create a workload that definitely runs computation and memory transfer in parallel."""

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda:0")

    # Create separate streams
    compute_stream = torch.cuda.default_stream()
    memory_stream = torch.cuda.Stream()

    print("🚀 Creating controlled parallel workload...")

    # Create tensors for heavy computation (simulate expert inference)
    batch_size = 128
    hidden_size = 4096
    expert_size = 16384

    # Input data on GPU
    input_data = torch.randn(batch_size, hidden_size, device=device)

    # Multiple expert weights on GPU for computation
    expert_weights = []
    for i in range(8):  # 8 experts for sustained computation
        weight = torch.randn(hidden_size, expert_size, device=device)
        expert_weights.append(weight)

    # Large CPU tensors for memory transfer (simulate expert prefetching)
    cpu_experts = []
    gpu_buffers = []
    for i in range(4):  # 4 experts to prefetch
        cpu_expert = torch.randn(hidden_size, expert_size)  # Large expert on CPU
        gpu_buffer = torch.empty(hidden_size, expert_size, device=device)  # GPU buffer
        cpu_experts.append(cpu_expert)
        gpu_buffers.append(gpu_buffer)

    # SEQUENTIAL VERSION (for comparison)
    print("📊 Running sequential version...")
    torch.cuda.synchronize()
    start_time = time.time()

    nvtx_range_push("Sequential_Computation")
    # Heavy computation
    results = []
    for i, weight in enumerate(expert_weights):
        result = torch.mm(input_data, weight)
        results.append(result)
    nvtx_range_pop()

    nvtx_range_push("Sequential_Memory_Transfer")
    # Memory transfers
    for i in range(4):
        gpu_buffers[i].copy_(cpu_experts[i])
    nvtx_range_pop()

    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.4f}s")

    # PARALLEL VERSION
    print("⚡ Running parallel version...")
    torch.cuda.synchronize()
    start_time = time.time()

    # Start memory transfers on separate stream FIRST
    memory_events = []
    with torch.cuda.stream(memory_stream):
        nvtx_range_push("Parallel_Memory_Transfers")
        for i in range(4):
            nvtx_range_push(f"Transfer_Expert_{i}")
            gpu_buffers[i].copy_(cpu_experts[i], non_blocking=True)
            nvtx_range_pop()

            # Create event to track this transfer
            event = torch.cuda.Event()
            event.record(memory_stream)
            memory_events.append(event)
        nvtx_range_pop()

    # Do heavy computation on main stream (should overlap with memory transfers)
    nvtx_range_push("Parallel_Computation")
    results_parallel = []
    for i, weight in enumerate(expert_weights):
        nvtx_range_push(f"Expert_Compute_{i}")
        result = torch.mm(input_data, weight)
        results_parallel.append(result)
        nvtx_range_pop()
    nvtx_range_pop()

    # Wait for memory transfers to complete
    nvtx_range_push("Wait_For_Memory")
    for event in memory_events:
        event.wait()
    nvtx_range_pop()

    torch.cuda.synchronize()
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.4f}s")

    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1.3:
        print("✅ TRUE PARALLELISM ACHIEVED!")
        return True
    else:
        print("❌ No significant parallelism")
        return False

def create_overlapping_workload():
    """Create sustained overlapping computation and memory operations."""

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda:0")

    print("\n🔄 Creating sustained overlapping workload...")

    # Create streams
    compute_stream = torch.cuda.default_stream()
    memory_stream = torch.cuda.Stream()

    # Large workload parameters
    batch_size = 64
    hidden_size = 2048
    expert_size = 8192
    num_iterations = 12

    # GPU tensors for computation
    input_data = torch.randn(batch_size, hidden_size, device=device)
    expert_weight = torch.randn(hidden_size, expert_size, device=device)

    # CPU tensors for transfers
    cpu_data = [torch.randn(hidden_size, expert_size) for _ in range(num_iterations)]
    gpu_buffers = [torch.empty(hidden_size, expert_size, device=device) for _ in range(num_iterations)]

    torch.cuda.synchronize()
    start_time = time.time()

    nvtx_range_push("Sustained_Parallel_Workload")

    # Start first memory transfer
    with torch.cuda.stream(memory_stream):
        nvtx_range_push("Memory_Pipeline")
        for i in range(num_iterations):
            nvtx_range_push(f"Transfer_{i}")
            gpu_buffers[i].copy_(cpu_data[i], non_blocking=True)
            nvtx_range_pop()
        nvtx_range_pop()

    # Do sustained computation that should overlap with transfers
    nvtx_range_push("Compute_Pipeline")
    computation_results = []
    for i in range(num_iterations):
        nvtx_range_push(f"Compute_{i}")
        # Heavy computation
        result = torch.mm(input_data, expert_weight)
        result = torch.mm(result, expert_weight.T)  # More work
        computation_results.append(result)
        nvtx_range_pop()
    nvtx_range_pop()

    nvtx_range_pop()

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    print(f"Sustained parallel time: {total_time:.4f}s")

    return True

if __name__ == "__main__":
    print("Creating controlled parallel CUDA workload for Nsight analysis...")
    create_parallel_workload()
    create_overlapping_workload()
    print("✅ Workload complete - ready for Nsight profiling")