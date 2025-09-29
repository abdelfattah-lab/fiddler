#!/usr/bin/env python3
"""
Fixed version of parallel memory compute demonstration
Demonstrates overlapping memory transfers with compute operations using CUDA streams
"""
import math
import time
import numpy as np
from numba import cuda, float32

# Kernel: simple but compute-heavy per element
@cuda.jit
def heavy_elementwise_kernel(inp, out, iters):
    i = cuda.grid(1)
    n = inp.size
    if i < n:
        x = inp[i]
        # do some repeated work to simulate compute (avoids being memory-bound)
        s = x
        for _ in range(iters):
            s = s * 1.0000003 + 0.000001  # cheap FP ops
        out[i] = s

def main():
    # Parameters
    total_elems = 160 * 1024 * 1024        # 40M elements (~160MB for float32)
    chunk_elems = 10 * 1024 * 1024        # chunk size (10M elements) -> ~40MB
    dtype = np.float32
    iters_per_elem = 100                  # increases compute per element

    # Derived
    n_chunks = math.ceil(total_elems / chunk_elems)
    chunk_elems = int(chunk_elems)
    total_elems = int(total_elems)

    print(f"Total elements: {total_elems:,}, chunk size: {chunk_elems:,}, n_chunks: {n_chunks}")

    # Create large pinned host arrays
    host_src = cuda.pinned_array(total_elems, dtype=dtype)
    host_out = cuda.pinned_array(total_elems, dtype=dtype)

    # Fill with some values
    host_src[:] = np.random.rand(total_elems).astype(dtype)

    # Create device buffers for double buffering
    d_buf_a_in = cuda.device_array(chunk_elems, dtype=dtype)
    d_buf_a_out = cuda.device_array(chunk_elems, dtype=dtype)
    d_buf_b_in = cuda.device_array(chunk_elems, dtype=dtype)
    d_buf_b_out = cuda.device_array(chunk_elems, dtype=dtype)

    # Create separate streams for memory transfer and compute
    transfer_stream = cuda.stream()
    compute_stream = cuda.stream()

    # Events for synchronization
    start_event = cuda.event()
    end_event = cuda.event()

    print("Starting streaming loop with overlapped memory transfer and compute...")
    start_event.record()

    # Process chunks with overlap
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * chunk_elems
        this_chunk_size = min(chunk_elems, total_elems - offset)

        # Select buffers (alternate between A and B)
        if chunk_idx % 2 == 0:
            d_in_buf = d_buf_a_in
            d_out_buf = d_buf_a_out
        else:
            d_in_buf = d_buf_b_in
            d_out_buf = d_buf_b_out

        # Get host slices for this chunk
        h_in_slice = host_src[offset:offset + this_chunk_size]
        h_out_slice = host_out[offset:offset + this_chunk_size]

        # Async H2D transfer on transfer stream
        if this_chunk_size == chunk_elems:
            # Full chunk - use pre-allocated buffer
            d_in_buf.copy_to_device(h_in_slice, stream=transfer_stream)

            # Wait for transfer to complete before starting compute
            transfer_event = cuda.event()
            transfer_event.record(stream=transfer_stream)
            transfer_event.synchronize()

            # Launch kernel on compute stream
            threads_per_block = 256
            blocks = (this_chunk_size + threads_per_block - 1) // threads_per_block
            heavy_elementwise_kernel[blocks, threads_per_block, compute_stream](
                d_in_buf[:this_chunk_size], d_out_buf[:this_chunk_size], iters_per_elem
            )

            # Copy result back to host on compute stream
            d_out_buf[:this_chunk_size].copy_to_host(h_out_slice, stream=compute_stream)
        else:
            # Last chunk might be smaller - allocate exact size
            d_in_small = cuda.device_array(this_chunk_size, dtype=dtype)
            d_out_small = cuda.device_array(this_chunk_size, dtype=dtype)

            d_in_small.copy_to_device(h_in_slice, stream=transfer_stream)

            transfer_event = cuda.event()
            transfer_event.record(stream=transfer_stream)
            transfer_event.synchronize()

            threads_per_block = 256
            blocks = (this_chunk_size + threads_per_block - 1) // threads_per_block
            heavy_elementwise_kernel[blocks, threads_per_block, compute_stream](
                d_in_small, d_out_small, iters_per_elem
            )

            d_out_small.copy_to_host(h_out_slice, stream=compute_stream)

    # Wait for all operations to complete
    compute_stream.synchronize()
    transfer_stream.synchronize()

    end_event.record()
    end_event.synchronize()
    elapsed_ms = cuda.event_elapsed_time(start_event, end_event)
    print(f"Total elapsed time (ms): {elapsed_ms:.2f}")

    # Correctness check on multiple samples
    print("\nCorrectness verification:")
    for i in [123, 1234, 12345, 123456 % total_elems]:
        x = float(host_src[i])
        s = x
        for _ in range(iters_per_elem):
            s = s * 1.0000003 + 0.000001
        got = host_out[i]
        diff = abs(s - got)
        print(f"Sample {i:6d}: CPU={s:.6f}, GPU={got:.6f}, diff={diff:.6f} {'✓' if diff < 1e-5 else '✗'}")

if __name__ == "__main__":
    main()