#!/usr/bin/env python3
"""
Test if we can achieve true parallelism with a simplified workload
that mimics our expert loading pattern
"""

import torch
import time
# Import NVTX for profiling markers
try:
    import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

def nvtx_range_push(name):
    if NVTX_AVAILABLE:
        nvtx.range_push(name)

def nvtx_range_pop():
    if NVTX_AVAILABLE:
        nvtx.range_pop()

def test_parallel_computation_and_memory():
    """Test computation + memory transfer in parallel."""

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda:0")

    # Create streams
    compute_stream = torch.cuda.default_stream()
    memory_stream = torch.cuda.Stream()

    # Create tensors for computation (simulating expert execution)
    size = 2048
    expert_input = torch.randn(64, size, device=device)
    expert_weight = torch.randn(size, size, device=device)

    # Create tensors for memory transfer (simulating expert prefetch)
    cpu_expert = torch.randn(size, size)  # Simulate expert on CPU
    gpu_buffer = torch.empty(size, size, device=device)  # Simulate prefetch buffer

    print("Testing computation + memory transfer in parallel...")

    # Sequential version (baseline)
    torch.cuda.synchronize()
    start_time = time.time()

    # Computation
    result1 = torch.mm(expert_input, expert_weight)

    # Memory transfer
    gpu_buffer.copy_(cpu_expert, non_blocking=False)

    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.4f}s")

    # Parallel version
    torch.cuda.synchronize()
    start_time = time.time()

    # Start memory transfer on separate stream
    with torch.cuda.stream(memory_stream):
        gpu_buffer.copy_(cpu_expert, non_blocking=True)

    # Do computation on main stream (should overlap with memory transfer)
    result2 = torch.mm(expert_input, expert_weight)

    # Wait for both to complete
    torch.cuda.synchronize()
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.4f}s")

    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1.2:
        print("✅ Achieved parallelism!")
        return True
    else:
        print("❌ No significant parallelism")
        return False

def test_multiple_memory_transfers():
    """Test multiple overlapping memory transfers."""

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda:0")

    # Create multiple streams for multiple transfers
    streams = [torch.cuda.Stream() for _ in range(4)]

    # Create multiple CPU tensors (simulating multiple experts)
    size = 1024
    cpu_tensors = [torch.randn(size, size) for _ in range(4)]
    gpu_buffers = [torch.empty(size, size, device=device) for _ in range(4)]

    print("\nTesting multiple overlapping memory transfers...")

    # Sequential transfers
    torch.cuda.synchronize()
    start_time = time.time()

    for i in range(4):
        gpu_buffers[i].copy_(cpu_tensors[i])

    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    print(f"Sequential transfers: {sequential_time:.4f}s")

    # Parallel transfers
    torch.cuda.synchronize()
    start_time = time.time()

    # Start all transfers in parallel
    for i in range(4):
        with torch.cuda.stream(streams[i]):
            gpu_buffers[i].copy_(cpu_tensors[i], non_blocking=True)

    # Wait for all to complete
    torch.cuda.synchronize()
    parallel_time = time.time() - start_time
    print(f"Parallel transfers: {parallel_time:.4f}s")

    speedup = sequential_time / parallel_time
    print(f"Transfer speedup: {speedup:.2f}x")

    if speedup > 1.5:
        print("✅ Multiple transfers can overlap!")
        return True
    else:
        print("❌ Transfers are serialized by hardware")
        return False

if __name__ == "__main__":
    test_parallel_computation_and_memory()
    test_multiple_memory_transfers()