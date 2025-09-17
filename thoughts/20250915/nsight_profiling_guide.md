# Nvidia Nsight Systems Profiling Guide for Fiddler

## Overview

This guide covers how to properly use Nvidia Nsight Systems to profile CUDA applications, specifically for analyzing performance bottlenecks in the Fiddler Mixtral implementation.

## Prerequisites

- Nvidia Nsight Systems installed (`nsys` command available)
- CUDA-enabled application to profile
- Sufficient disk space for profile files (.nsys-rep files can be large)

## Basic Profiling Command

```bash
nsys profile \
  --output profile_output.nsys-rep \
  --force-overwrite true \
  --trace cuda,cudnn,cublas,osrt,nvtx \
  --cuda-memory-usage true \
  --gpu-metrics-device all \
  --duration 60 \
  --sample cpu \
  your_application_command
```

### Command Breakdown

- `--output`: Specify output file name (.nsys-rep extension)
- `--force-overwrite`: Overwrite existing files without prompt
- `--trace`: Which APIs to trace (cuda=CUDA Runtime/Driver, cudnn=cuDNN, cublas=cuBLAS, osrt=OS Runtime, nvtx=NVTX markers)
- `--cuda-memory-usage`: Track CUDA memory operations
- `--gpu-metrics-device all`: Collect GPU metrics for all devices
- `--duration 60`: Maximum profiling duration in seconds
- `--sample cpu`: Enable CPU sampling (note: may not work on all systems)

## Common Issues and Solutions

### Issue 1: Deprecated GPU Metrics Warning
**Warning**: `'--gpu-metrics-device' is deprecated. Use '--gpu-metrics-devices' instead.`

**Solution**: Replace `--gpu-metrics-device all` with `--gpu-metrics-devices all`

### Issue 2: CPU Sampling Not Supported
**Warning**: `CPU IP/backtrace sampling not supported, disabling.`

**Solution**: This is normal on many systems. The warning can be ignored, or remove `--sample cpu` from the command.

### Issue 3: CPU Context Switch Tracing Not Supported
**Warning**: `CPU context switch tracing not supported, disabling.`

**Solution**: This is normal and can be ignored. Profiling will continue without CPU context switches.

### Issue 4: Large Profile Files
**Problem**: Profile files become very large (>1GB)

**Solutions**:
- Use `--duration` to limit profiling time
- Focus on specific APIs with selective `--trace` options
- Use `--capture-range cudaProfilerApi` for application-controlled profiling

## Analysis Commands

After profiling, analyze results using these commands:

### Memory Operations Analysis
```bash
nsys stats --report cuda_gpu_mem_time_sum profile.nsys-rep
nsys stats --report cuda_gpu_mem_size_sum profile.nsys-rep
```

### CUDA API Analysis
```bash
nsys stats --report cuda_api_sum profile.nsys-rep
```

### GPU Kernel Analysis
```bash
nsys stats --report cuda_gpu_kern_sum profile.nsys-rep
```

### Available Reports
List all available reports:
```bash
nsys stats --help-reports
```

Common useful reports:
- `cuda_api_sum` - CUDA API call summary
- `cuda_gpu_kern_sum` - GPU kernel execution summary
- `cuda_gpu_mem_time_sum` - Memory operations by time
- `cuda_gpu_mem_size_sum` - Memory operations by size
- `cuda_api_trace` - Detailed API call trace
- `nvtx_sum` - NVTX range summary (if using NVTX markers)

## GUI Analysis

For visual analysis, use Nsight Systems GUI:
```bash
nsight-sys profile.nsys-rep
```

The GUI provides:
- Timeline view of GPU activity
- Memory transfer visualization
- Kernel execution details
- CPU activity correlation

## Best Practices for Fiddler Profiling

### 1. Clean Environment
- Ensure GPU memory is cleared between runs
- Use consistent input for reproducible results
- Clear any existing pattern files when testing different scenarios

### 2. Focus on Bottlenecks
For memory-bound applications like Fiddler:
- Pay attention to `cuda_gpu_mem_time_sum` reports
- Look for large Host-to-Device transfer times
- Check number of memory transfer calls vs. total time

### 3. Comparative Analysis
When comparing implementations:
```bash
# Profile baseline
nsys profile --output baseline.nsys-rep [options] baseline_command

# Profile optimized version
nsys profile --output optimized.nsys-rep [options] optimized_command

# Compare memory operations
nsys stats --report cuda_gpu_mem_time_sum baseline.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum optimized.nsys-rep

# Compare API calls
nsys stats --report cuda_api_sum baseline.nsys-rep
nsys stats --report cuda_api_sum optimized.nsys-rep
```

### 4. Look for Key Metrics

**Memory Transfer Bottlenecks:**
- Total time in `[CUDA memcpy Host-to-Device]`
- Number of transfer calls vs. average transfer size
- Look for many small transfers (latency bound) vs. few large transfers (bandwidth bound)

**GPU Utilization:**
- Total kernel execution time vs. total application time
- If kernel time << memory transfer time, you have a memory bottleneck

**API Overhead:**
- Time spent in `cudaMemcpyAsync` vs. actual transfer time
- Excessive `cudaMalloc`/`cudaFree` calls indicate poor memory management

## Troubleshooting Profile Analysis

### Problem: Reports Show "No data"
**Cause**: Application finished before meaningful data collection
**Solution**: Increase application runtime or use shorter profiling duration

### Problem: Memory Transfer Times Don't Make Sense
**Cause**: Multiple overlapping transfers or async operations
**Solution**: Look at timeline in GUI to understand actual execution order

### Problem: Inconsistent Results Between Runs
**Cause**: GPU state, thermal throttling, or system load variations
**Solution**:
- Run multiple profiles and average results
- Ensure consistent system state
- Check for background GPU processes

## Example Analysis Workflow

1. **Profile both implementations**:
   ```bash
   nsys profile --output baseline.nsys-rep [options] baseline_command
   nsys profile --output optimized.nsys-rep [options] optimized_command
   ```

2. **Quick memory comparison**:
   ```bash
   nsys stats --report cuda_gpu_mem_time_sum baseline.nsys-rep
   nsys stats --report cuda_gpu_mem_time_sum optimized.nsys-rep
   ```

3. **Check API overhead**:
   ```bash
   nsys stats --report cuda_api_sum baseline.nsys-rep
   nsys stats --report cuda_api_sum optimized.nsys-rep
   ```

4. **Verify compute time**:
   ```bash
   nsys stats --report cuda_gpu_kern_sum baseline.nsys-rep
   nsys stats --report cuda_gpu_kern_sum optimized.nsys-rep
   ```

5. **Visual timeline analysis**:
   ```bash
   nsight-sys baseline.nsys-rep
   nsight-sys optimized.nsys-rep
   ```

This workflow helps identify whether optimizations are actually working and where the real bottlenecks lie.