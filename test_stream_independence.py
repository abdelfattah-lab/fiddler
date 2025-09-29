#!/usr/bin/env python3
"""
Test if CUDA streams actually work independently in our setup
"""

import torch
import time

def test_stream_independence():
    """Test if operations on different streams actually run in parallel."""

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda:0")

    # Create two streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Create large tensors for significant computation time
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = torch.randn(size, size, device=device)
    d = torch.randn(size, size, device=device)

    print("Testing CUDA stream independence...")

    # Test 1: Sequential execution on main stream
    torch.cuda.synchronize()
    start_time = time.time()

    result1 = torch.mm(a, b)
    result2 = torch.mm(c, d)

    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    print(f"Sequential execution time: {sequential_time:.4f}s")

    # Test 2: Parallel execution on different streams
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.cuda.stream(stream1):
        result3 = torch.mm(a, b)

    with torch.cuda.stream(stream2):
        result4 = torch.mm(c, d)

    # Wait for both streams to complete
    torch.cuda.synchronize()
    parallel_time = time.time() - start_time
    print(f"Parallel execution time: {parallel_time:.4f}s")

    # Check if parallelism worked
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1.3:  # At least 30% speedup
        print("✅ Streams are working independently!")
        return True
    else:
        print("❌ Streams are NOT working independently!")
        return False

def test_async_copy():
    """Test if async memory copies work."""

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda:0")
    stream = torch.cuda.Stream()

    # Create CPU tensor
    cpu_tensor = torch.randn(1024, 1024)
    gpu_tensor = torch.empty_like(cpu_tensor, device=device)

    print("\nTesting async memory copy...")

    # Test async copy
    with torch.cuda.stream(stream):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)

        # Record event
        event = torch.cuda.Event()
        event.record(stream)

    # Check if event is ready immediately (it shouldn't be for large copy)
    if event.query():
        print("❌ Copy completed immediately - not actually async!")
        return False
    else:
        print("✅ Copy is asynchronous!")
        event.wait()  # Wait for completion
        return True

if __name__ == "__main__":
    test_stream_independence()
    test_async_copy()