#!/usr/bin/env python3
"""
FiddlerQwenWithPrefetch - Prefetch implementation extending working baseline
Adds prefetch capabilities on top of the working FiddlerQwen
"""

import torch
import torch.nn.functional as F
import json
import os
import time
import copy
from .qwen import FiddlerQwen

# Import NVTX for profiling markers
try:
    import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False


class ExpertUsageProfiler:
    """Records and provides expert usage patterns for prediction."""

    def __init__(self, n_layers, n_experts):
        # Use direct O(1) lookup: token_position -> layer_id -> [expert_id1, expert_id2, ...]
        self.expert_patterns = {}  # token_pos -> {layer_id: [expert1, expert2, expert3, expert4]}
        self.current_token_pos = 0
        self.collection_mode = True

    def record_layer_experts(self, layer_id, selected_experts):
        """Record the exact top-k experts for this layer at current token position."""
        # Convert to string keys for JSON compatibility
        token_key = str(self.current_token_pos)
        layer_key = str(layer_id)

        if token_key not in self.expert_patterns:
            self.expert_patterns[token_key] = {}

        # Take top-k experts (Qwen uses top_k=4)
        top_k_experts = selected_experts.flatten().tolist()
        self.expert_patterns[token_key][layer_key] = top_k_experts

    def advance_token_position(self):
        """Move to next token position (called after each complete forward pass)."""
        self.current_token_pos += 1

    def get_experts_for_layer(self, layer_id, token_pos):
        """O(1) lookup: return exact experts needed for this layer at this token position."""
        # Convert to string keys for JSON compatibility
        token_key = str(token_pos)
        layer_key = str(layer_id)

        if token_key in self.expert_patterns and layer_key in self.expert_patterns[token_key]:
            return self.expert_patterns[token_key][layer_key]
        return None  # No recorded pattern available


class PrefetchMetrics:
    """Tracks prefetch hit rates and performance metrics.

    Hit rate definition:
    - Numerator: Number of tokens assigned to GPU experts that were readily available (prefetched or GPU-resident)
    - Denominator: Total number of tokens assigned to GPU experts

    This gives a cache effectiveness measure weighted by the amount of work done.
    """

    def __init__(self):
        self.total_tokens = 0  # Total tokens assigned to GPU experts
        self.prefetch_hit_tokens = 0  # Tokens assigned to readily-available GPU experts
        self.prefetch_miss_tokens = 0  # Tokens assigned to on-demand-loaded GPU experts

        # Detailed tracking by phase (token-weighted)
        self.prefill_hit_tokens = 0
        self.prefill_miss_tokens = 0
        self.decode_hit_tokens = 0
        self.decode_miss_tokens = 0

        # Track current phase
        self.current_phase = "prefill"

    def set_phase(self, phase):
        """Set current phase (prefill or decode)."""
        self.current_phase = phase

    def record_expert_access(self, layer_id, expert_id, was_prefetched, token_count):
        """Record whether an expert access was a hit or miss, weighted by token count.

        Args:
            layer_id: Layer index
            expert_id: Expert index
            was_prefetched: True if expert was readily available (prefetched or GPU-resident), False if loaded on-demand
            token_count: Number of tokens assigned to this expert (for weighting)
        """
        self.total_tokens += token_count

        if was_prefetched:
            self.prefetch_hit_tokens += token_count
            if self.current_phase == "prefill":
                self.prefill_hit_tokens += token_count
            else:
                self.decode_hit_tokens += token_count
        else:
            self.prefetch_miss_tokens += token_count
            if self.current_phase == "prefill":
                self.prefill_miss_tokens += token_count
            else:
                self.decode_miss_tokens += token_count

    def get_hit_rate(self):
        """Get overall prefetch hit rate (token-weighted)."""
        if self.total_tokens == 0:
            return 0.0
        return self.prefetch_hit_tokens / self.total_tokens

    def get_prefill_hit_rate(self):
        """Get prefill-phase prefetch hit rate (token-weighted)."""
        total_prefill = self.prefill_hit_tokens + self.prefill_miss_tokens
        if total_prefill == 0:
            return 0.0
        return self.prefill_hit_tokens / total_prefill

    def get_decode_hit_rate(self):
        """Get decode-phase prefetch hit rate (token-weighted)."""
        total_decode = self.decode_hit_tokens + self.decode_miss_tokens
        if total_decode == 0:
            return 0.0
        return self.decode_hit_tokens / total_decode


class FiddlerQwenWithPrefetch(FiddlerQwen):
    """Qwen implementation with prefetch capabilities and optional Fiddler CPU offloading."""

    def __init__(self, args, num_experts_to_prefetch=1, enable_cpu_offload=False,
                 latency_cpu=None, latency_gpu=None, n_gpu_resident_experts=0):
        # Initialize base class (includes Fiddler mode support)
        super().__init__(args)

        # Configure number of experts to prefetch per layer (0-16)
        self.num_experts_to_prefetch = max(0, min(16, num_experts_to_prefetch))

        # Fiddler CPU offloading parameters
        self.enable_cpu_offload = enable_cpu_offload

        # Cost model parameters (will be set from profiling or defaults)
        self.latency_cpu = latency_cpu  # ms per token on CPU
        self.latency_gpu = latency_gpu  # ms constant on GPU (transfer overhead)

        # Number of experts to keep permanently on GPU (popular experts)
        self.n_gpu_resident_experts = n_gpu_resident_experts

        # Track expert location: expert_loc[layer_idx][expert_idx] = 0 (CPU) or 1 (GPU)
        # This is dynamically determined per forward pass for non-resident experts
        self.expert_loc = {}  # Will be populated during forward pass

        # Track CPU execution statistics
        self.cpu_expert_count = 0
        self.gpu_expert_count = 0
        self.cpu_execution_time = 0.0
        self.gpu_execution_time = 0.0

        # Fiddler mode is inherited from base class
        # self.use_fiddler_mode and self.fiddler_batch_threshold are already set

        # Initialize prefetch components
        self.profiler = ExpertUsageProfiler(len(self.moe_layers), self.n_expert)
        self.metrics = PrefetchMetrics()

        # Load or initialize expert usage patterns
        self.expert_patterns = self.load_expert_patterns()
        self.collection_mode = (self.expert_patterns is None)

        if not self.collection_mode:
            # Load patterns into profiler
            self.profiler.expert_patterns = self.expert_patterns
            self.profiler.collection_mode = False

        # Dual buffer system - Buffer A for even layers, Buffer B for odd layers
        self.prefetch_cache_A = {}  # (layer_idx, expert_idx) -> prefetch buffer (even layers)
        self.prefetch_cache_B = {}  # (layer_idx, expert_idx) -> prefetch buffer (odd layers)
        self.prefetch_buffer_A = {}  # Buffer A: 4 expert slots for even layers
        self.prefetch_buffer_B = {}  # Buffer B: 4 expert slots for odd layers

        # Create separate CUDA streams for asynchronous prefetching
        self.prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.expert_ready_events = {}  # Track when each expert is ready: (layer, expert) -> event

        # Pin CPU memory for all MoE experts to enable async transfers
        print("📌 Pinning CPU memory for MoE experts...")
        self._pin_expert_memory()

        # Keep first 2 MoE layers (0-1) permanently on GPU
        # Since we predict layer+2, layers 0-1 are never prefetched
        self.gpu_resident_layers = set()
        if len(self.moe_layers) >= 2:
            # Move experts from first 2 MoE layers to GPU
            for i in range(2):
                layer_idx = self.moe_layers[i]
                self.gpu_resident_layers.add(layer_idx)
                moe_layer = self.model.model.layers[layer_idx].mlp
                for expert_idx, expert in enumerate(moe_layer.experts):
                    expert.to(self.device, dtype=self.dtype)
            print(f"🔒 Layers {self.moe_layers[0]}-{self.moe_layers[1]} experts permanently on GPU")

        # Initialize dual buffer system with pinned memory
        if len(self.moe_layers) > 0 and not self.collection_mode and self.num_experts_to_prefetch > 0:
            print(f"🔧 Initializing dual buffer system for {len(self.moe_layers)} MoE layers...")
            sample_layer = self.model.model.layers[self.moe_layers[0]]
            sample_expert = sample_layer.mlp.experts[0]

            # Create Buffer A (N expert slots for even layers)
            self.prefetch_buffer_A = {}
            for i in range(self.num_experts_to_prefetch):
                self.prefetch_buffer_A[i] = copy.deepcopy(sample_expert).to(self.device, dtype=self.dtype)

            # Create Buffer B (N expert slots for odd layers)
            self.prefetch_buffer_B = {}
            for i in range(self.num_experts_to_prefetch):
                self.prefetch_buffer_B[i] = copy.deepcopy(sample_expert).to(self.device, dtype=self.dtype)

            print(f"✅ Dual buffer system initialized with {self.num_experts_to_prefetch} expert slots per buffer")
            print(f"✅ Buffer A for even layers, Buffer B for odd layers")
            print(f"✅ GPU buffers ready to receive async transfers from pinned CPU memory")
            if self.prefetch_stream:
                print(f"✅ Async prefetch stream initialized")

        print(f"🔮 Prefetch mode: {'Collection' if self.collection_mode else 'Prediction'}")
        if not self.collection_mode:
            print(f"🎯 Prefetching {self.num_experts_to_prefetch} expert(s) per layer")

    def _pin_expert_memory(self):
        """Pin CPU memory for all MoE experts to enable async transfers without blocking."""
        if not torch.cuda.is_available():
            return

        pinned_count = 0
        for layer_idx in self.moe_layers[2:]:  # Skip first 2 layers (they'll be on GPU)
            moe_layer = self.model.model.layers[layer_idx].mlp
            for expert_idx, expert in enumerate(moe_layer.experts):
                # Pin memory for each parameter in the expert
                for param in expert.parameters():
                    if param.device.type == 'cpu':
                        # Create pinned memory tensor and copy data
                        pinned_param = torch.empty_like(param, pin_memory=True)
                        pinned_param.copy_(param)
                        # Replace the parameter data with pinned version
                        param.data = pinned_param
                        pinned_count += 1

        print(f"✅ Pinned {pinned_count} expert parameters in CPU memory")

    def load_expert_patterns(self):
        """Load expert usage patterns from file."""
        pattern_file = "expert_usage_patterns_qwen.json"
        if os.path.exists(pattern_file):
            with open(pattern_file, 'r') as f:
                return json.load(f)
        return None

    def save_expert_patterns(self):
        """Save expert usage patterns to file."""
        if self.collection_mode:
            with open("expert_usage_patterns_qwen.json", 'w') as f:
                json.dump(self.profiler.expert_patterns, f, indent=2)

    def _predict_experts_for_layer(self, layer_idx, token_pos):
        """Predict which experts will be needed for this layer at given token position."""
        if self.collection_mode:
            return []

        predicted_experts = self.profiler.get_experts_for_layer(layer_idx, token_pos)
        return predicted_experts if predicted_experts else []

    def _moe_forward_with_management(self, hidden_states, layer_idx):
        """Enhanced MoE forward with prefetch prediction and CPU-to-GPU loading (matching baseline logic)."""
        # Start timing for this layer
        layer_start_time = time.time()

        moe_layer = self.model.model.layers[layer_idx].mlp

        # Get dimensions
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Determine if we're in prefill (sequence_length > 1) or decode (sequence_length == 1) phase
        is_prefill = sequence_length > 1

        # Set the phase in metrics tracker
        self.metrics.set_phase("prefill" if is_prefill else "decode")

        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router computation (same as baseline)
        router_logits = moe_layer.gate(hidden_states_flat)

        # Routing weights computation (same as baseline)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        if moe_layer.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Convert routing weights to the correct dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Record expert usage patterns for future prefetching
        if self.collection_mode:
            self.profiler.record_layer_experts(layer_idx, selected_experts)


        # Initialize output tensor with correct dtype (same as baseline)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # One-hot encode selected experts (same as baseline)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=moe_layer.num_experts).permute(2, 1, 0)

        # Find active experts (same as baseline)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # CPU OFFLOADING: Partition experts if enabled
        if self.enable_cpu_offload and layer_idx not in self.gpu_resident_layers:
            # Step 1: Calculate token counts for each active expert
            expert_token_counts = {}
            expert_data = {}  # Store expert data for later processing

            for i, expert_idx_tensor in enumerate(expert_hit):
                expert_idx = expert_idx_tensor.item()
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                if len(top_x) == 0:
                    continue

                expert_token_counts[expert_idx] = len(top_x)
                expert_data[expert_idx] = {
                    'idx': idx,
                    'top_x': top_x,
                    'input': hidden_states_flat[None, top_x].reshape(-1, hidden_dim),
                    'weights': routing_weights[top_x, idx, None]
                }

            # Step 2: Calculate costs and partition experts
            cost_per_expert = self._calculate_expert_costs(expert_token_counts, layer_idx)
            cpu_expert_list, gpu_expert_list = self._partition_experts_greedy(cost_per_expert)

            if NVTX_AVAILABLE and len(cpu_expert_list) > 0:
                nvtx_cpu_range = nvtx.start_range(f"CPU_EXPERTS: Layer{layer_idx} ({len(cpu_expert_list)} experts)")

            # Step 3: Process CPU experts
            for expert_idx in cpu_expert_list:
                data = expert_data[expert_idx]
                top_x = data['top_x']

                # Execute on CPU
                cpu_start = time.time()
                weighted_output = self._run_expert_on_cpu(
                    layer_idx, expert_idx, data['input'], data['weights']
                )
                cpu_elapsed = time.time() - cpu_start

                self.cpu_expert_count += 1
                self.cpu_execution_time += cpu_elapsed
                # Note: CPU experts are NOT counted in cnt_expert_all since hit rate only tracks GPU experts

                # Accumulate results
                weighted_output = weighted_output.to(final_hidden_states.dtype)
                final_hidden_states.index_add_(0, top_x, weighted_output)

            if NVTX_AVAILABLE and len(cpu_expert_list) > 0:
                nvtx.end_range(nvtx_cpu_range)

            if NVTX_AVAILABLE and len(gpu_expert_list) > 0:
                nvtx_gpu_range = nvtx.start_range(f"GPU_EXPERTS: Layer{layer_idx} ({len(gpu_expert_list)} experts)")

            # Step 4: Process GPU experts (with prefetch support)
            for expert_idx in gpu_expert_list:
                data = expert_data[expert_idx]
                top_x = data['top_x']
                current_state = data['input']
                expert_routing_weights = data['weights']

                # Check if expert was prefetched
                if self._is_expert_prefetched(layer_idx, expert_idx):
                    cache_key = (layer_idx, expert_idx)
                    if cache_key in self.expert_ready_events:
                        self.expert_ready_events[cache_key].synchronize()

                    expert_buffer = self._get_expert_for_execution_with_prefetch(
                        layer_idx, expert_idx, was_prefetched=True
                    )
                    self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=True, token_count=len(top_x))
                    self.cnt_expert_hit += len(top_x)
                else:
                    expert_buffer = self._get_expert_for_execution(layer_idx, expert_idx)
                    self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=False, token_count=len(top_x))

                self.gpu_expert_count += 1
                self.cnt_expert_all += len(top_x)

                # Execute on GPU
                if current_state.device != expert_buffer.gate_proj.weight.device:
                    current_state = current_state.to(expert_buffer.gate_proj.weight.device)

                expert_output = expert_buffer(current_state)
                current_hidden_states = expert_output * expert_routing_weights.to(expert_output.device)

                # Accumulate results
                if current_hidden_states.device != final_hidden_states.device:
                    current_hidden_states = current_hidden_states.to(final_hidden_states.device)
                current_hidden_states = current_hidden_states.to(final_hidden_states.dtype)
                final_hidden_states.index_add_(0, top_x, current_hidden_states)

            if NVTX_AVAILABLE and len(gpu_expert_list) > 0:
                nvtx.end_range(nvtx_gpu_range)

        else:
            # STANDARD PATH: No CPU offloading, use existing prefetch logic
            for i, expert_idx_tensor in enumerate(expert_hit):
                expert_idx = expert_idx_tensor.item()

                # Find tokens assigned to this expert (same as baseline)
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                if len(top_x) == 0:
                    continue

                # Get input for this expert (same as baseline)
                current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)

                # Get routing weights for this expert (same as baseline)
                expert_routing_weights = routing_weights[top_x, idx, None]

                # Check if this is a GPU-resident layer (0-1)
                if layer_idx in self.gpu_resident_layers:
                    # Use expert directly from GPU (no loading needed)
                    if NVTX_AVAILABLE:
                        range_id = nvtx.start_range(f"GPU_RESIDENT: Layer{layer_idx}_Expert{expert_idx}")
                    expert_buffer = moe_layer.experts[expert_idx]
                    if NVTX_AVAILABLE:
                        nvtx.end_range(range_id)
                    self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=True, token_count=len(top_x))
                    self.cnt_expert_hit += len(top_x)
                # Check if expert was prefetched, use prefetched version if available
                elif self._is_expert_prefetched(layer_idx, expert_idx):
                    # CRITICAL: Wait for async transfer to complete before using the expert
                    # This ensures correctness while still allowing parallelism (transfer happens
                    # in parallel with previous layer's compute, we only sync when actually needed)
                    cache_key = (layer_idx, expert_idx)
                    if cache_key in self.expert_ready_events:
                        # Synchronize with the prefetch stream to ensure transfer is complete
                        self.expert_ready_events[cache_key].synchronize()

                    # Use prefetched expert
                    if NVTX_AVAILABLE:
                        range_id = nvtx.start_range(f"PREFETCH_HIT: Layer{layer_idx}_Expert{expert_idx}")
                    expert_buffer = self._get_expert_for_execution_with_prefetch(layer_idx, expert_idx, was_prefetched=True)
                    if NVTX_AVAILABLE:
                        nvtx.end_range(range_id)
                    self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=True, token_count=len(top_x))
                    self.cnt_expert_hit += len(top_x)
                else:
                    # Load expert on-demand using the base class method
                    if NVTX_AVAILABLE:
                        range_id = nvtx.start_range(f"EXPERT_LOAD_ON_DEMAND: Layer{layer_idx}_Expert{expert_idx}")
                    expert_buffer = self._get_expert_for_execution(layer_idx, expert_idx)
                    if NVTX_AVAILABLE:
                        nvtx.end_range(range_id)
                    self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=False, token_count=len(top_x))

                self.cnt_expert_all += len(top_x)

                # Execute expert computation on GPU (same as baseline)
                if current_state.device != expert_buffer.gate_proj.weight.device:
                    current_state = current_state.to(expert_buffer.gate_proj.weight.device)

                expert_output = expert_buffer(current_state)

                # Apply routing weights (same as baseline)
                current_hidden_states = expert_output * expert_routing_weights.to(expert_output.device)

                # Move back to original device if needed and accumulate (same as baseline)
                if current_hidden_states.device != final_hidden_states.device:
                    current_hidden_states = current_hidden_states.to(final_hidden_states.device)

                # Ensure dtype consistency before accumulation (same as baseline)
                current_hidden_states = current_hidden_states.to(final_hidden_states.dtype)

                final_hidden_states.index_add_(0, top_x, current_hidden_states)

        # Add shared expert (same as baseline)
        shared_expert_output = moe_layer.shared_expert(hidden_states_flat)
        shared_expert_gate = F.sigmoid(moe_layer.shared_expert_gate(hidden_states_flat))
        shared_expert_output = shared_expert_gate * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        # TRIGGER PREFETCH AFTER finishing with current layer's experts
        # Now that we're done with layer N, start prefetching for layer N+2
        # This gives maximum time for async transfer to complete
        if not self.collection_mode and layer_idx + 2 in self.moe_layers:
            if NVTX_AVAILABLE:
                range_id = nvtx.start_range(f"PREFETCH_TRIGGER_AFTER_LAYER: Layer{layer_idx+2}")
            self._trigger_prefetch_for_layer(layer_idx + 2, self.profiler.current_token_pos)
            if NVTX_AVAILABLE:
                nvtx.end_range(range_id)

        # Advance token position after the last MoE layer
        if layer_idx == self.moe_layers[-1]:
            self.profiler.advance_token_position()

        # Return in the same format as original forward (output, router_logits)
        router_logits = router_logits.view(batch_size, sequence_length, -1)

        # Track timing
        layer_time = time.time() - layer_start_time
        if is_prefill:
            self.prefill_time += layer_time
        else:
            self.decode_time += layer_time
            if not self.moe_layers or layer_idx == self.moe_layers[-1]:
                self.decode_token_count += 1

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits


    def _is_expert_prefetched(self, layer_idx, expert_idx):
        """Check if the expert is already loaded in prefetch buffers."""
        cache_key = (layer_idx, expert_idx)
        # Check appropriate buffer based on layer parity
        if self._get_layer_index_in_moe_list(layer_idx) % 2 == 0:
            return cache_key in self.prefetch_cache_A
        else:
            return cache_key in self.prefetch_cache_B

    def _get_layer_index_in_moe_list(self, layer_idx):
        """Get the index of this layer in the MoE layers list (for parity calculation)."""
        return self.moe_layers.index(layer_idx) if layer_idx in self.moe_layers else 0

    def _get_expert_for_execution_with_prefetch(self, layer_idx, expert_idx, was_prefetched):
        """Get expert for execution, using appropriate prefetch buffer if available."""
        cache_key = (layer_idx, expert_idx)

        if was_prefetched:
            # Use appropriate buffer based on layer parity
            if self._get_layer_index_in_moe_list(layer_idx) % 2 == 0:
                if cache_key in self.prefetch_cache_A:
                    return self.prefetch_cache_A[cache_key]
            else:
                if cache_key in self.prefetch_cache_B:
                    return self.prefetch_cache_B[cache_key]

        # Load expert on-demand using the base class method
        return self._get_expert_for_execution(layer_idx, expert_idx)

    def _trigger_prefetch_for_layer(self, target_layer_idx, token_pos):
        """Trigger asynchronous prefetch for experts needed at target layer."""
        if target_layer_idx >= len(self.model.model.layers) or target_layer_idx not in self.moe_layers:
            return

        # Get predicted experts for the target layer
        predicted_experts = self._predict_experts_for_layer(target_layer_idx, token_pos)

        if not predicted_experts:
            return

        # Determine which buffer to use based on layer parity
        layer_moe_index = self._get_layer_index_in_moe_list(target_layer_idx)
        is_even_layer = layer_moe_index % 2 == 0

        # Clear old prefetch cache for this layer from appropriate buffer
        if is_even_layer:
            old_keys = [k for k in self.prefetch_cache_A.keys() if k[0] == target_layer_idx]
            for key in old_keys:
                del self.prefetch_cache_A[key]
            current_cache = self.prefetch_cache_A
            current_buffer = self.prefetch_buffer_A
            buffer_name = "Buffer A (even)"
        else:
            old_keys = [k for k in self.prefetch_cache_B.keys() if k[0] == target_layer_idx]
            for key in old_keys:
                del self.prefetch_cache_B[key]
            current_cache = self.prefetch_cache_B
            current_buffer = self.prefetch_buffer_B
            buffer_name = "Buffer B (odd)"

        # Use async stream for prefetching if available
        if self.prefetch_stream is not None:
            self._load_experts_async(target_layer_idx, predicted_experts, current_cache, current_buffer, buffer_name)
        else:
            # Fallback to synchronous loading
            self._load_experts_sync(target_layer_idx, predicted_experts, current_cache, current_buffer, buffer_name)

    def _load_experts_async(self, target_layer_idx, predicted_experts, current_cache, current_buffer, buffer_name):
        """Load N experts asynchronously based on num_experts_to_prefetch configuration."""
        if not predicted_experts:
            return

        # Load up to num_experts_to_prefetch experts
        num_to_load = min(self.num_experts_to_prefetch, len(predicted_experts), len(current_buffer))

        for slot_idx in range(num_to_load):
            expert_idx = predicted_experts[slot_idx]
            if expert_idx is None or not isinstance(expert_idx, int) or expert_idx >= self.n_expert:
                continue

            if slot_idx not in current_buffer:
                continue

            # PREPARE CPU OPERATIONS OUTSIDE STREAM CONTEXT
            cpu_expert = self.model.model.layers[target_layer_idx].mlp.experts[expert_idx]

            state_dict = cpu_expert.state_dict()

            # Convert dtype on CPU WHILE PRESERVING PINNED MEMORY
            converted_state_dict = {}
            for key, tensor in state_dict.items():
                # CRITICAL BUG FIX: state_dict() returns COPIES not references
                # Even if original param is pinned, state_dict copy might not be!
                # Solution: ALWAYS ensure pinned memory for async transfer
                if not tensor.is_pinned():
                    # Tensor is not pinned, we MUST pin it for async transfer
                    pinned_tensor = torch.empty_like(tensor, pin_memory=True)
                    pinned_tensor.copy_(tensor)
                    tensor = pinned_tensor

                if tensor.dtype != self.dtype:
                    # Convert dtype while preserving pinned status
                    converted = torch.empty_like(tensor, dtype=self.dtype, pin_memory=True)
                    converted.copy_(tensor)
                    converted_state_dict[key] = converted
                else:
                    converted_state_dict[key] = tensor

            # Setup cache and buffer references (CPU operations)
            cache_key = (target_layer_idx, expert_idx)
            prefetch_buffer = current_buffer[slot_idx]

            # Load CPU expert weights into GPU prefetch buffer with correct dtype
            if NVTX_AVAILABLE:
                range_id = nvtx.start_range(f"ASYNC_EXPERT_LOAD: Layer{target_layer_idx}_Expert{expert_idx}_{buffer_name}_Slot{slot_idx}")

            # Use stream context ONLY for the actual GPU memory transfers
            # CRITICAL: torch.no_grad() prevents autograd from adding hidden synchronization
            with torch.no_grad():
                with torch.cuda.stream(self.prefetch_stream):
                    # Expert transfer - using direct H2D copy from pinned CPU memory
                    for name, param in prefetch_buffer.named_parameters():
                        if name in converted_state_dict:
                            src_tensor = converted_state_dict[name]
                            # Direct copy from pinned CPU to GPU parameter
                            param.data.copy_(src_tensor, non_blocking=True)

            # CPU operations outside stream context to avoid hidden synchronization
            current_cache[cache_key] = prefetch_buffer

            # Record event on prefetch stream (for optional tracking, not used for forced waiting)
            expert_event = torch.cuda.Event()
            expert_event.record(self.prefetch_stream)
            self.expert_ready_events[cache_key] = expert_event

            if NVTX_AVAILABLE:
                nvtx.end_range(range_id)

    def _load_experts_sync(self, target_layer_idx, predicted_experts, current_cache, current_buffer, buffer_name):
        """Fallback synchronous expert loading."""
        num_to_load = min(self.num_experts_to_prefetch, len(predicted_experts))
        for i, expert_idx in enumerate(predicted_experts[:num_to_load]):
            if expert_idx is not None and isinstance(expert_idx, int) and 0 <= expert_idx < self.n_expert:
                cache_key = (target_layer_idx, expert_idx)

                # Check if we have the buffer slot
                if i not in current_buffer:
                    continue

                # Load CPU expert weights into GPU prefetch buffer with correct dtype
                if NVTX_AVAILABLE:
                    range_id = nvtx.start_range(f"EXPERT_PREFETCH_LOAD: Layer{target_layer_idx}_Expert{expert_idx}_{buffer_name}_Slot{i}")

                cpu_expert = self.model.model.layers[target_layer_idx].mlp.experts[expert_idx]
                prefetch_buffer = current_buffer[i]

                # Ensure all weights are loaded with the correct dtype
                state_dict = cpu_expert.state_dict()
                for key, tensor in state_dict.items():
                    if tensor.dtype != self.dtype:
                        state_dict[key] = tensor.to(self.dtype)

                prefetch_buffer.load_state_dict(state_dict)

                # Cache the prefetched expert in appropriate buffer
                current_cache[cache_key] = prefetch_buffer

                if NVTX_AVAILABLE:
                    nvtx.end_range(range_id)

    def _calculate_expert_costs(self, expert_token_counts, layer_idx):
        """
        Calculate CPU and GPU costs for each expert based on token counts.

        Args:
            expert_token_counts: dict mapping expert_idx -> num_tokens assigned to that expert
            layer_idx: current layer index

        Returns:
            cost_per_expert: dict mapping expert_idx -> (cpu_cost, gpu_cost) in ms
        """
        # Default cost model parameters if not specified
        latency_cpu = self.latency_cpu if self.latency_cpu is not None else 0.05  # ms per token on CPU
        latency_gpu = self.latency_gpu if self.latency_gpu is not None else 5.0   # ms expert transfer cost

        cost_per_expert = {}

        for expert_idx, num_tokens in expert_token_counts.items():
            if num_tokens <= 0:
                # Skip experts without work so greedy partitioning ignores them.
                continue

            # CPU cost grows linearly with the number of tokens assigned to the expert.
            cpu_cost = float(num_tokens) * latency_cpu

            # GPU cost mirrors the Mixtral implementation:
            # * Default case: expert must be transferred, so cost is the transfer latency.
            # * Resident/prefetched experts already live on the GPU, so cost is effectively 0.
            if layer_idx in self.gpu_resident_layers or self._is_expert_prefetched(layer_idx, expert_idx):
                gpu_cost = 0.0
            else:
                gpu_cost = latency_gpu

            cost_per_expert[expert_idx] = (cpu_cost, gpu_cost)

        return cost_per_expert

    def _partition_experts_greedy(self, cost_per_expert):
        """
        Greedy algorithm to partition experts between CPU and GPU.

        The goal is to minimize max(cpu_cost, gpu_cost) - the critical path latency.

        Args:
            cost_per_expert: dict mapping expert_idx -> (cpu_cost, gpu_cost)

        Returns:
            cpu_experts: list of expert indices to run on CPU
            gpu_experts: list of expert indices to run on GPU
        """
        if not cost_per_expert:
            return [], []

        # Calculate benefit of moving expert from GPU to CPU
        # Positive benefit means moving to CPU is beneficial
        benefits = []
        for expert_idx, (cpu_cost, gpu_cost) in cost_per_expert.items():
            # Benefit = GPU cost saved - CPU cost added
            benefit = gpu_cost - cpu_cost
            benefits.append((benefit, expert_idx, cpu_cost, gpu_cost))

        # Sort by benefit (descending - highest benefit first)
        benefits.sort(reverse=True)

        # Initialize: all experts on GPU
        cpu_experts = []
        gpu_experts = list(cost_per_expert.keys())

        # Calculate initial costs
        cpu_cost_total = 0.0
        gpu_cost_total = sum(cost_per_expert[idx][1] for idx in gpu_experts)

        current_bottleneck = max(cpu_cost_total, gpu_cost_total)

        # Greedily move experts to CPU if it reduces the bottleneck
        for benefit, expert_idx, cpu_cost, gpu_cost in benefits:
            if benefit <= 0:
                # No benefit to moving to CPU, stop
                break

            # Try moving this expert to CPU
            new_cpu_cost = cpu_cost_total + cpu_cost
            new_gpu_cost = gpu_cost_total - gpu_cost

            new_bottleneck = max(new_cpu_cost, new_gpu_cost)

            # Only move if it reduces the bottleneck
            if new_bottleneck < current_bottleneck:
                cpu_experts.append(expert_idx)
                gpu_experts.remove(expert_idx)
                cpu_cost_total = new_cpu_cost
                gpu_cost_total = new_gpu_cost
                current_bottleneck = new_bottleneck

        return cpu_experts, gpu_experts

    def _run_expert_on_cpu(self, layer_idx, expert_idx, input_tensor, routing_weights):
        """
        Execute expert on CPU and return results.

        Args:
            layer_idx: layer index
            expert_idx: expert index
            input_tensor: input tensor (on GPU)
            routing_weights: routing weights for this expert (on GPU)

        Returns:
            expert_output: output tensor (on GPU, ready for accumulation)
        """
        moe_layer = self.model.model.layers[layer_idx].mlp
        cpu_expert = moe_layer.experts[expert_idx]

        # Move input to CPU
        input_cpu = input_tensor.cpu()

        # Execute expert on CPU
        with torch.no_grad():
            expert_output_cpu = cpu_expert(input_cpu)

        # Move output back to GPU
        expert_output_gpu = expert_output_cpu.to(self.device, dtype=self.dtype)

        # Apply routing weights (on GPU)
        weighted_output = expert_output_gpu * routing_weights.to(expert_output_gpu.device)

        return weighted_output

    def generate(self, text=None, output_token=20, input_token=None):
        """
        Generate with prefetch-aware processing and metrics tracking.

        Args:
            text: Input text. Can be:
                  - Single string: "The capital of France is"
                  - List of strings: ["The capital of France is", "The theory of relativity"]
                  - None: Use default text
            output_token: Number of tokens to generate
            input_token: Limit on input token count

        Returns:
            Tuple of (prefill_time, decode_time_per_token, prefill_hit_rate, decode_hit_rate)
        """
        # Handle text input - support both single and batched inputs
        if text is None:
            text = "The capital of France is"

        # Tokenize input - handle both single strings and lists of strings
        if isinstance(text, str):
            # Single input - use existing behavior
            inputs = self.tokenizer(text, return_tensors="pt")
        elif isinstance(text, list):
            # Batched input - tokenize all together with padding
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        else:
            raise ValueError(f"text must be a string or list of strings, got {type(text)}")

        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device) if inputs.attention_mask is not None else None

        # Limit input tokens if specified
        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :input_token]

        # Reset statistics
        self.expert_fetch_count = 0
        self.expert_hit_count = 0
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.current_expert = None

        # Reset prefetch metrics
        self.metrics = PrefetchMetrics()

        # Reset CPU offloading statistics
        self.cpu_expert_count = 0
        self.gpu_expert_count = 0
        self.cpu_execution_time = 0.0
        self.gpu_execution_time = 0.0

        # Reset profiler token position for new generation
        if not self.collection_mode:
            self.profiler.current_token_pos = 0

        # CRITICAL: Clear prefetch caches to avoid stale experts from previous generation
        self.prefetch_cache_A.clear()
        self.prefetch_cache_B.clear()
        self.expert_ready_events.clear()

        # Reset timing statistics
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.decode_token_count = 0

        start_time = time.time()

        with torch.no_grad():
            # Reset cache
            self.model.generation_config.use_cache = True
            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            # Generate with the built-in generation method but with cache reset after each token
            # This ensures our hooks are called appropriately
            # IMPORTANT: Set eos_token_id=None to force exactly output_token generations
            # This ensures pattern collection captures all token positions for benchmarking
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_token,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=None,  # Force exact token count
                use_cache=True
            )

            # Token position is advanced in MoE forward during generation

        total_time = time.time() - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate hit rates
        baseline_hit_rate = self.cnt_expert_hit / self.cnt_expert_all if self.cnt_expert_all > 0 else 0.0
        overall_hit_rate = self.metrics.get_hit_rate()
        prefill_hit_rate = self.metrics.get_prefill_hit_rate()
        decode_hit_rate = self.metrics.get_decode_hit_rate()

        # Store for comparison
        self.last_generated_text = generated_text

        # Save patterns if in collection mode
        if self.collection_mode:
            self.save_expert_patterns()

        # Use actual measured times from MoE layer tracking
        # Note: self.prefill_time and self.decode_time are accumulated across all MoE layers
        prefill_time = self.prefill_time
        decode_time = self.decode_time / self.decode_token_count if self.decode_token_count else 0.0

        print(f"Generated: {generated_text}")
        print(f"🎯 Prefetch hit rate - Overall: {overall_hit_rate:.1%}, Prefill: {prefill_hit_rate:.1%}, Decode: {decode_hit_rate:.1%}")
        print(f"⏱️  Prefill: {prefill_time:.3f}s, Decode/token: {decode_time:.3f}s")

        # Print CPU offloading statistics if enabled
        if self.enable_cpu_offload:
            total_experts = self.cpu_expert_count + self.gpu_expert_count
            cpu_pct = (self.cpu_expert_count / total_experts * 100) if total_experts > 0 else 0
            gpu_pct = (self.gpu_expert_count / total_experts * 100) if total_experts > 0 else 0
            print(f"🖥️  CPU offloading: {self.cpu_expert_count} CPU ({cpu_pct:.1f}%), {self.gpu_expert_count} GPU ({gpu_pct:.1f}%)")
            print(f"⚡ CPU exec time: {self.cpu_execution_time:.3f}s, GPU exec time: {self.gpu_execution_time:.3f}s")

    # Return separate hit rates for prefill and decode (decode time is per token)
        return (prefill_time, decode_time, prefill_hit_rate, decode_hit_rate)

    def get_prefetch_stats(self):
        """Get detailed prefetch statistics (token-weighted)."""
        return {
            'overall_hit_rate': self.metrics.get_hit_rate(),
            'prefill_hit_rate': self.metrics.get_prefill_hit_rate(),
            'decode_hit_rate': self.metrics.get_decode_hit_rate(),
            'total_tokens': self.metrics.total_tokens,
            'prefetch_hit_tokens': self.metrics.prefetch_hit_tokens,
            'prefetch_miss_tokens': self.metrics.prefetch_miss_tokens,
            'prefill_hit_tokens': self.metrics.prefill_hit_tokens,
            'prefill_miss_tokens': self.metrics.prefill_miss_tokens,
            'decode_hit_tokens': self.metrics.decode_hit_tokens,
            'decode_miss_tokens': self.metrics.decode_miss_tokens
        }

    def get_cpu_offload_stats(self):
        """Get detailed CPU offloading statistics."""
        total_experts = self.cpu_expert_count + self.gpu_expert_count
        return {
            'cpu_expert_count': self.cpu_expert_count,
            'gpu_expert_count': self.gpu_expert_count,
            'total_expert_count': total_experts,
            'cpu_percentage': (self.cpu_expert_count / total_experts * 100) if total_experts > 0 else 0,
            'gpu_percentage': (self.gpu_expert_count / total_experts * 100) if total_experts > 0 else 0,
            'cpu_execution_time': self.cpu_execution_time,
            'gpu_execution_time': self.gpu_execution_time,
            'enabled': self.enable_cpu_offload
        }