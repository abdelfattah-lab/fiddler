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
    """Tracks prefetch hit rates and performance metrics."""

    def __init__(self):
        self.total_expert_requests = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

        # Detailed tracking by phase
        self.prefill_hits = 0
        self.prefill_misses = 0
        self.decode_hits = 0
        self.decode_misses = 0

        # Track current phase
        self.current_phase = "prefill"

    def set_phase(self, phase):
        """Set current phase (prefill or decode)."""
        self.current_phase = phase

    def record_expert_access(self, layer_id, expert_id, was_prefetched):
        """Record whether an expert access was a hit or miss."""
        self.total_expert_requests += 1

        if was_prefetched:
            self.prefetch_hits += 1
            if self.current_phase == "prefill":
                self.prefill_hits += 1
            else:
                self.decode_hits += 1
        else:
            self.prefetch_misses += 1
            if self.current_phase == "prefill":
                self.prefill_misses += 1
            else:
                self.decode_misses += 1

    def get_hit_rate(self):
        """Get overall prefetch hit rate."""
        if self.total_expert_requests == 0:
            return 0.0
        return self.prefetch_hits / self.total_expert_requests

    def get_prefill_hit_rate(self):
        """Get prefill-phase prefetch hit rate."""
        total_prefill = self.prefill_hits + self.prefill_misses
        if total_prefill == 0:
            return 0.0
        return self.prefill_hits / total_prefill

    def get_decode_hit_rate(self):
        """Get decode-phase prefetch hit rate."""
        total_decode = self.decode_hits + self.decode_misses
        if total_decode == 0:
            return 0.0
        return self.decode_hits / total_decode


class FiddlerQwenWithPrefetch(FiddlerQwen):
    """Qwen implementation with prefetch capabilities."""

    def __init__(self, args, num_experts_to_prefetch=1):
        # Initialize base class (includes Fiddler mode support)
        super().__init__(args)

        # Configure number of experts to prefetch per layer (0-16)
        self.num_experts_to_prefetch = max(0, min(16, num_experts_to_prefetch))

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

        # Process each active expert (adapted from baseline)
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
                self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=True)
                self.cnt_expert_hit += len(top_x)
            # Check if expert was prefetched, use prefetched version if available
            elif self._is_expert_prefetched(layer_idx, expert_idx):
                # REMOVED: Event wait - this was the PRIMARY CAUSE of no parallelism
                # Prefetch runs truly async, GPU will naturally wait when accessing tensor if needed

                # Use prefetched expert
                if NVTX_AVAILABLE:
                    range_id = nvtx.start_range(f"PREFETCH_HIT: Layer{layer_idx}_Expert{expert_idx}")
                expert_buffer = self._get_expert_for_execution_with_prefetch(layer_idx, expert_idx, was_prefetched=True)
                if NVTX_AVAILABLE:
                    nvtx.end_range(range_id)
                self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=True)
                self.cnt_expert_hit += len(top_x)
            else:
                # Load expert on-demand using the base class method
                if NVTX_AVAILABLE:
                    range_id = nvtx.start_range(f"EXPERT_LOAD_ON_DEMAND: Layer{layer_idx}_Expert{expert_idx}")
                expert_buffer = self._get_expert_for_execution(layer_idx, expert_idx)
                if NVTX_AVAILABLE:
                    nvtx.end_range(range_id)
                self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=False)

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

    def generate(self, text=None, output_token=20, input_token=None):
        """Generate with prefetch-aware processing and metrics tracking."""
        # Handle text input
        if text is None:
            text = "The capital of France is"

        inputs = self.tokenizer(text, return_tensors="pt")
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

        # Reset profiler token position for new generation
        if not self.collection_mode:
            self.profiler.current_token_pos = 0

        # Reset timing statistics
        self.prefill_time = 0.0
        self.decode_time = 0.0

        start_time = time.time()

        with torch.no_grad():
            # Reset cache
            self.model.generation_config.use_cache = True
            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            # Generate with the built-in generation method but with cache reset after each token
            # This ensures our hooks are called appropriately
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_token,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
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
        decode_time = self.decode_time

        print(f"Generated: {generated_text}")
        print(f"🎯 Prefetch hit rate - Overall: {overall_hit_rate:.1%}, Prefill: {prefill_hit_rate:.1%}, Decode: {decode_hit_rate:.1%}")
        print(f"⏱️  Prefill: {prefill_time:.3f}s, Decode: {decode_time:.3f}s")

        # Return separate hit rates for prefill and decode
        return (prefill_time, decode_time, prefill_hit_rate, decode_hit_rate)

    def get_prefetch_stats(self):
        """Get detailed prefetch statistics."""
        return {
            'overall_hit_rate': self.metrics.get_hit_rate(),
            'prefill_hit_rate': self.metrics.get_prefill_hit_rate(),
            'decode_hit_rate': self.metrics.get_decode_hit_rate(),
            'total_requests': self.metrics.total_expert_requests,
            'prefetch_hits': self.metrics.prefetch_hits,
            'prefetch_misses': self.metrics.prefetch_misses,
            'prefill_hits': self.metrics.prefill_hits,
            'prefill_misses': self.metrics.prefill_misses,
            'decode_hits': self.metrics.decode_hits,
            'decode_misses': self.metrics.decode_misses
        }