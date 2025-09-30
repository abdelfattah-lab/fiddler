#!/usr/bin/env python3
"""
Fixed version of parallel memory compute demonstration
Demonstrates overlapping memory transfers with compute operations using CUDA streams
Using PyTorch instead of Numba
"""
import math
import time
import torch

def heavy_elementwise_operation(inp, iters):
    """
    PyTorch-based compute-heavy operation that's heavy enough to allow overlap
    Uses element-wise operations to simulate compute-intensive work
    """
    x = inp
    for _ in range(iters):
        x = x * 1.0000003 + 0.000001
    return x

def main():
    # Parameters
    total_elems = 160 * 1024 * 1024        # 40M elements (~160MB for float32)
    chunk_elems = 10 * 1024 * 1024        # chunk size (10M elements) -> ~40MB
    dtype = torch.float32
    iters_per_elem = 100                  # increases compute per element

    # Derived
    n_chunks = math.ceil(total_elems / chunk_elems)
    chunk_elems = int(chunk_elems)
    total_elems = int(total_elems)

    print(f"Total elements: {total_elems:,}, chunk size: {chunk_elems:,}, n_chunks: {n_chunks}")

    # Create large pinned host tensors
    host_src = torch.randn(total_elems, dtype=dtype).pin_memory()
    host_out = torch.zeros(total_elems, dtype=dtype).pin_memory()

    # Create device buffers for double buffering
    d_buf_a = torch.empty(chunk_elems, dtype=dtype, device='cuda')
    d_buf_b = torch.empty(chunk_elems, dtype=dtype, device='cuda')

    # Create separate streams for memory transfer and compute
    transfer_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    print("Starting streaming loop with overlapped memory transfer and compute...")
    torch.cuda.synchronize()
    start_time = time.time()

    # Process chunks with overlap
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * chunk_elems
        this_chunk_size = min(chunk_elems, total_elems - offset)

        # Select buffer (alternate between A and B)
        d_buf = d_buf_a if chunk_idx % 2 == 0 else d_buf_b

        # Get host slices for this chunk
        h_in_slice = host_src[offset:offset + this_chunk_size]
        h_out_slice = host_out[offset:offset + this_chunk_size]

        # Async H2D transfer on transfer stream
        with torch.cuda.stream(transfer_stream):
            if this_chunk_size == chunk_elems:
                d_buf.copy_(h_in_slice, non_blocking=True)
                d_chunk = d_buf
            else:
                # Last chunk - use smaller slice
                d_chunk = torch.empty(this_chunk_size, dtype=dtype, device='cuda')
                d_chunk.copy_(h_in_slice, non_blocking=True)

        # Record event on transfer stream
        transfer_event = torch.cuda.Event()
        transfer_event.record(transfer_stream)

        # Wait for transfer to complete before starting compute
        compute_stream.wait_event(transfer_event)

        # Compute on compute stream
        with torch.cuda.stream(compute_stream):
            result = heavy_elementwise_operation(d_chunk, iters_per_elem)

            # Copy result back to host
            h_out_slice.copy_(result, non_blocking=True)

    # Wait for all operations to complete
    compute_stream.synchronize()
    transfer_stream.synchronize()

    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"Total elapsed time (ms): {elapsed_ms:.2f}")

    # Correctness check on multiple samples
    print("\nCorrectness verification:")
    for i in [123, 1234, 12345, 123456 % total_elems]:
        x = float(host_src[i].item())
        s = x
        for _ in range(iters_per_elem):
            s = s * 1.0000003 + 0.000001
        got = float(host_out[i].item())
        diff = abs(s - got)
        print(f"Sample {i:6d}: CPU={s:.6f}, GPU={got:.6f}, diff={diff:.6f} {'✓' if diff < 1e-5 else '✗'}")

if __name__ == "__main__":
    main()
