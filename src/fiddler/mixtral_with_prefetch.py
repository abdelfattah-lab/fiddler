import copy
import json
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers

from .mixtral import FiddlerMixtral


class ExpertBuffer:
    """Buffer to hold 2 experts for a specific layer type (even/odd)."""

    def __init__(self, expert_placeholder):
        self.experts = [None, None]  # Hold 2 experts
        self.expert_ids = [None, None]  # Track which experts are loaded
        self.layer_id = None  # Which layer this buffer serves
        self.is_ready = False
        self.placeholder = expert_placeholder

    def initialize_experts(self):
        """Initialize expert placeholders."""
        for i in range(2):
            self.experts[i] = copy.deepcopy(self.placeholder)
            self.expert_ids[i] = None
        self.is_ready = False


class AsyncPrefetcher:
    """Handles asynchronous expert prefetching using CUDA streams."""

    def __init__(self):
        self.prefetch_stream = torch.cuda.Stream()

    def prefetch_experts_async(self, target_buffer, layer_id, expert_ids, model_layers):
        """Asynchronously prefetch experts into the target buffer."""
        # Mark buffer as not ready during loading
        target_buffer.is_ready = False
        target_buffer.layer_id = layer_id

        with torch.cuda.stream(self.prefetch_stream):
            for i, expert_id in enumerate(expert_ids):
                if i < len(target_buffer.experts) and expert_id is not None:
                    # Load expert weights into buffer slot
                    target_buffer.experts[i].load_state_dict(
                        model_layers[layer_id].block_sparse_moe.experts[expert_id].state_dict()
                    )
                    target_buffer.expert_ids[i] = expert_id

            # Mark buffer as ready after loading completes
            target_buffer.is_ready = True


class ExpertUsageProfiler:
    """Records and provides expert usage patterns for prediction."""

    def __init__(self, n_layers, n_experts):
        # Use direct O(1) lookup: token_position -> layer_id -> [expert_id1, expert_id2]
        self.expert_patterns = {}  # token_pos -> {layer_id: [expert1, expert2]}
        self.current_token_pos = 0
        self.collection_mode = True

    def record_layer_experts(self, layer_id, selected_experts):
        """Record the exact 2 experts for this layer at current token position."""
        # Convert to string keys for JSON compatibility
        token_key = str(self.current_token_pos)
        layer_key = str(layer_id)

        if token_key not in self.expert_patterns:
            self.expert_patterns[token_key] = {}

        top_2_experts = selected_experts.flatten().tolist()[:2]  # Take exactly top-2
        self.expert_patterns[token_key][layer_key] = top_2_experts

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
        self.layer_metrics = {}  # per-layer hit rates

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

        # Track per-layer metrics
        if layer_id not in self.layer_metrics:
            self.layer_metrics[layer_id] = {'hits': 0, 'total': 0}

        self.layer_metrics[layer_id]['total'] += 1
        if was_prefetched:
            self.layer_metrics[layer_id]['hits'] += 1

    def get_hit_rate(self):
        """Get overall prefetch hit rate."""
        if self.total_expert_requests == 0:
            return 0.0
        return self.prefetch_hits / self.total_expert_requests

    def get_decode_hit_rate(self):
        """Get decode-phase prefetch hit rate."""
        total_decode = self.decode_hits + self.decode_misses
        if total_decode == 0:
            return 0.0
        return self.decode_hits / total_decode

    def get_stats_summary(self):
        """Get comprehensive statistics summary."""
        total_prefill = self.prefill_hits + self.prefill_misses
        total_decode = self.decode_hits + self.decode_misses

        return {
            'overall_hit_rate': self.get_hit_rate(),
            'decode_hit_rate': self.get_decode_hit_rate(),
            'prefill_hit_rate': self.prefill_hits / total_prefill if total_prefill > 0 else 0.0,
            'total_requests': self.total_expert_requests,
            'prefill_requests': total_prefill,
            'decode_requests': total_decode,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'prefill_hits': self.prefill_hits,
            'prefill_misses': self.prefill_misses,
            'decode_hits': self.decode_hits,
            'decode_misses': self.decode_misses
        }


class FiddlerMixtralWithPrefetch(FiddlerMixtral):
    """Mixtral implementation with dual-buffer prefetching system."""

    def __init__(self, args):
        super().__init__(args)

        # Initialize prefetch components
        self.buffer_a = ExpertBuffer(self.expert_placeholder)  # Even layers
        self.buffer_b = ExpertBuffer(self.expert_placeholder)  # Odd layers
        self.profiler = ExpertUsageProfiler(self.n_layer, self.n_expert)
        self.prefetcher = AsyncPrefetcher()
        self.metrics = PrefetchMetrics()

        # Initialize buffers
        self.buffer_a.initialize_experts()
        self.buffer_b.initialize_experts()

        # Load or initialize expert usage patterns
        self.expert_patterns = self.load_expert_patterns()
        self.collection_mode = (self.expert_patterns is None)

        if not self.collection_mode:
            # Load patterns into profiler
            self.profiler.expert_patterns = self.expert_patterns
            self.profiler.collection_mode = False

    def get_buffer_for_layer(self, layer_id):
        """Get the appropriate buffer for the given layer."""
        return self.buffer_a if layer_id % 2 == 0 else self.buffer_b

    def load_expert_patterns(self):
        """Load expert usage patterns from file."""
        pattern_file = "expert_usage_patterns.json"
        if os.path.exists(pattern_file):
            with open(pattern_file, 'r') as f:
                return json.load(f)
        return None

    def save_expert_patterns(self):
        """Save expert usage patterns to file."""
        if self.collection_mode:
            with open("expert_usage_patterns.json", 'w') as f:
                json.dump(self.profiler.expert_patterns, f, indent=2)

    def get_expert_for_execution(self, layer_id, expert_id):
        """Get expert for execution, trying prefetch buffer first, then fallback."""
        target_buffer = self.get_buffer_for_layer(layer_id)

        if (target_buffer.is_ready and
            target_buffer.layer_id == layer_id and
            expert_id in target_buffer.expert_ids):
            # Prefetch hit - use buffered expert
            buffer_index = target_buffer.expert_ids.index(expert_id)
            self.metrics.record_expert_access(layer_id, expert_id, was_prefetched=True)
            return target_buffer.experts[buffer_index]
        else:
            # Prefetch miss - fallback to on-demand loading
            self.metrics.record_expert_access(layer_id, expert_id, was_prefetched=False)
            return self.load_expert_on_demand(layer_id, expert_id)

    def load_expert_on_demand(self, layer_id, expert_id):
        """Load expert on-demand using the expert placeholder."""
        self.expert_placeholder.load_state_dict(
            self.model.layers[layer_id].block_sparse_moe.experts[expert_id].state_dict()
        )
        return self.expert_placeholder

    def trigger_prefetch_for_layer(self, layer_id, token_pos):
        """Trigger asynchronous prefetch for the specified layer at given token position."""
        if layer_id >= self.n_layer:
            return

        # Get expected experts for this layer from patterns
        if not self.collection_mode:
            expected_experts = self.profiler.get_experts_for_layer(layer_id, token_pos)
            if expected_experts:
                target_buffer = self.get_buffer_for_layer(layer_id)
                self.prefetcher.prefetch_experts_async(
                    target_buffer, layer_id, expected_experts, self.model.layers
                )

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
        """Modified mixtral forward with prefetch integration."""
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps)

        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape

            inps_residual = inps
            inps = layer.input_layernorm(inps)
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(-1, hidden_dim)

            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # Record expert usage if in collection mode
            if self.collection_mode:
                self.profiler.record_layer_experts(i_layer, selected_experts)

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            # Always use prefetch-aware expert processing (ignore cpu_offload parameter)
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=8
            ).permute(2, 1, 0)

            for i_expert in range(len(experts)):
                is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                idx, top_2 = torch.where(expert_mask[i_expert])

                if top_2.shape[0] == 0:
                    continue

                if is_cuda:
                    current_state = inps[None, top_2].reshape(-1, hidden_dim)
                    current_state = experts[i_expert](
                        current_state, routing_weights[top_2, idx, None]
                    )
                    inps_after_experts.index_add_(0, top_2, current_state)
                    self.cnt_expert_hit += top_2.shape[0]
                else:
                    # Use prefetch-aware expert retrieval
                    current_state = inps[None, top_2].reshape(-1, hidden_dim)
                    prefetch_expert = self.get_expert_for_execution(i_layer, i_expert)
                    current_state = prefetch_expert(
                        current_state, routing_weights[top_2, idx, None]
                    )
                    inps_after_experts.index_add_(0, top_2, current_state)

                self.cnt_expert_all += top_2.shape[0]

            # Trigger prefetch for next layers at current token position
            if i_layer + 2 < self.n_layer:
                self.trigger_prefetch_for_layer(i_layer + 2, self.profiler.current_token_pos)

            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis

    def generate(self, text=None, output_token=20, input_token=None):
        """Generate with prefetch-aware processing and metrics tracking."""
        torch.set_num_threads(16)

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        # Reset metrics for this generation
        self.metrics = PrefetchMetrics()

        # Reset profiler token position for new generation
        if not self.collection_mode:
            self.profiler.current_token_pos = 0

        input_ids, position_ids = self.tokenize(text)

        # Initialize cache AFTER tokenization
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        tick = time.time()
        is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)

        for i_token in range(output_token):
            if self.beam_width == 1:
                print(self.tokenizer.decode(input_ids[0]))

            if is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])

            # Set metrics phase
            self.metrics.set_phase("decode" if is_decode else "prefill")

            logits = self.mixtral_forward(input_ids, position_ids, is_decode)

            # Advance token position after each forward pass (prefill and decode)
            self.profiler.advance_token_position()

            logits = logits.to("cpu")
            logits = F.softmax(logits, dim=-1)

            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True

            probs = probs * new_probs
            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .view(-1, 1)
            )

            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True

        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        print("--------------------")
        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")

        # Store the generated text for comparison
        self.last_generated_text = decode_strings[max_ids[0]]

        # Save patterns if in collection mode
        if self.collection_mode:
            self.save_expert_patterns()

        # Return hit rate from prefetch metrics instead of expert cache hit rate
        prefetch_hit_rate = self.metrics.get_hit_rate()
        return (prefill_time, decode_time, prefetch_hit_rate)

    def get_prefetch_stats(self):
        """Get detailed prefetch statistics."""
        return self.metrics.get_stats_summary()