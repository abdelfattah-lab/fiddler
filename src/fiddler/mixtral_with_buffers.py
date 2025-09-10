import copy
import random
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers


class MixtralWithBuffers:
    def __init__(self, args, num_buffer_sets=2, prefetch_percentage=100):
        # DUAL BUFFER OPTIMIZATION: Use 2 buffers for true compute/memory overlap
        self.num_buffer_sets = 2
        
        # HYBRID PREFETCHING: Configurable percentage of layers to prefetch
        self.prefetch_percentage = prefetch_percentage  # 0-100%
        
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            # device_map='cpu',
            use_cache=True,
        )
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        # DUAL BUFFER: Two buffer sets for true compute/memory overlap
        self.buffer_a = []  # Buffer A: current layer being processed
        self.buffer_b = []  # Buffer B: next layer being prefetched
        for j in range(8):  # 8 experts per layer
            expert_buffer_a = copy.deepcopy(
                self.model.layers[0].block_sparse_moe.experts[0]
            ).to(self.dev)
            expert_buffer_b = copy.deepcopy(
                self.model.layers[0].block_sparse_moe.experts[0]
            ).to(self.dev)
            self.buffer_a.append(expert_buffer_a)
            self.buffer_b.append(expert_buffer_b)
            
        # FALLBACK: Expert placeholder for on-demand loading like baseline
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
        
        
        # Store max_experts_gpu if provided for testing
        if hasattr(args, 'max_experts_gpu'):
            self.max_experts_gpu = args.max_experts_gpu
       

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        # INSTRUMENTATION: Track prefetch vs fallback behavior
        self.prefetch_hits = 0
        self.prefetch_waits = 0  
        self.fallback_loads = 0
        self.prefetch_wait_times = []
        
        # HYBRID PREFETCHING: Track prefetching decisions
        self.prefetch_decisions = 0  # Total prefetch decisions made
        self.prefetch_skipped = 0    # Prefetches skipped due to percentage
        
        # DEBUG: Track prefetch triggering
        self.prefetch_triggers = []  # List of (completed_layer, target_layer, buffer) tuples
        self.layer_access_log = []   # List of (layer, token, hit_type) tuples

        self.bring_non_expert_to_gpu()

        # DUAL BUFFER: Track which layers are in each buffer
        self.buffer_a_layer = 0  # Buffer A starts with layer 0
        self.buffer_b_layer = 1  # Buffer B starts with layer 1
        
        import concurrent.futures
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.prefetch_shutdown = False
        
        # DUAL BUFFER: Streams for both buffers
        self.compute_stream = torch.cuda.Stream()
        self.memory_stream = torch.cuda.Stream()
        
        # DUAL BUFFER: Separate locks for each buffer (must be defined before loading)
        self.buffer_a_lock = threading.RLock()
        self.buffer_b_lock = threading.RLock()
        
        # Load first two layers into dual buffers
        self.load_layer_to_buffer_a(0)  # Load layer 0 into buffer A
        self.load_layer_to_buffer_b(1)  # Load layer 1 into buffer B
        
        # PHASE 2 OPTIMIZATION: Event pool for reuse instead of creating/destroying
        self.event_pool = []  # Reusable events
        self.event_pool_lock = threading.Lock()
        self.layer_prefetch_events = {}  # layer_idx -> threading.Event
        self.prefetch_tracking_lock = threading.RLock()  # Use RLock for nested locking
        
        print(f"Loaded layers 0 and 1 into dual buffers ({self.n_expert} experts per buffer)")

        print("Model is ready.")
        
    def __del__(self):
        """Cleanup ThreadPoolExecutor when object is destroyed"""
        self.shutdown_prefetch_threads()
        
    def shutdown_prefetch_threads(self):
        """Gracefully shutdown background prefetch threads"""
        self.prefetch_shutdown = True
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
            
    def get_reusable_event(self):
        # TODO: Is this really necessary? I don't like that we're adding more synchronization areas.
        """PHASE 2 OPTIMIZATION: Get a reusable event from pool or create new one"""
        with self.event_pool_lock:
            if self.event_pool:
                event = self.event_pool.pop()
                event.clear()  # Reset event state
                return event
            else:
                return threading.Event()
                
    def return_event_to_pool(self, event):
        """PHASE 2 OPTIMIZATION: Return event to pool for reuse"""
        with self.event_pool_lock:
            if len(self.event_pool) < 10:  # Limit pool size to prevent memory bloat
                self.event_pool.append(event)
            
    # Removed the adaptive prefetching that was essentially cheating by disabling prefetching

    def load_layer_to_buffer_a(self, layer_idx):
        """Load all experts of a layer into buffer A"""
        if layer_idx >= self.n_layer:
            raise ValueError(f"layer_idx L{layer_idx} >= n_layer {self.n_layer}")
            
        source_experts = self.model.layers[layer_idx].block_sparse_moe.experts
        
        with self.buffer_a_lock:
            with torch.cuda.stream(self.memory_stream):
                with torch.no_grad():
                    for i_expert in range(8):
                        buffer_expert = self.buffer_a[i_expert]
                        source_expert = source_experts[i_expert]
                        
                        for (buffer_name, buffer_param), (source_name, source_param) in zip(
                            buffer_expert.named_parameters(), source_expert.named_parameters()):
                            buffer_param.copy_(source_param, non_blocking=True)
            
            self.memory_stream.synchronize()
            self.buffer_a_layer = layer_idx

    def load_layer_to_buffer_b(self, layer_idx):
        """Load all experts of a layer into buffer B"""
        if layer_idx >= self.n_layer:
            raise ValueError(f"layer_idx L{layer_idx} >= n_layer {self.n_layer}")
            
        source_experts = self.model.layers[layer_idx].block_sparse_moe.experts
        
        with self.buffer_b_lock:
            with torch.cuda.stream(self.memory_stream):
                with torch.no_grad():
                    for i_expert in range(8):
                        buffer_expert = self.buffer_b[i_expert]
                        source_expert = source_experts[i_expert]
                        
                        for (buffer_name, buffer_param), (source_name, source_param) in zip(
                            buffer_expert.named_parameters(), source_expert.named_parameters()):
                            buffer_param.copy_(source_param, non_blocking=True)
            
            self.memory_stream.synchronize()
            self.buffer_b_layer = layer_idx

    def prefetch_layer_to_buffer_a_with_event(self, layer_to_prefetch_idx):
        """Prefetch layer to buffer A and signal event when complete"""
        if self.prefetch_shutdown:
            return
            
        self.load_layer_to_buffer_a(layer_to_prefetch_idx)
        
        # Signal completion and remove event
        event_to_signal = None
        with self.prefetch_tracking_lock:
            event_to_signal = self.layer_prefetch_events.pop(layer_to_prefetch_idx, None)

        if event_to_signal:
            event_to_signal.set()
            self.return_event_to_pool(event_to_signal)

    def prefetch_layer_to_buffer_b_with_event(self, layer_to_prefetch_idx):
        """Prefetch layer to buffer B and signal event when complete"""
        if self.prefetch_shutdown:
            return
            
        self.load_layer_to_buffer_b(layer_to_prefetch_idx)
        
        # Signal completion and remove event
        event_to_signal = None
        with self.prefetch_tracking_lock:
            event_to_signal = self.layer_prefetch_events.pop(layer_to_prefetch_idx, None)

        if event_to_signal:
            event_to_signal.set()
            self.return_event_to_pool(event_to_signal)

    def should_prefetch_layer(self, layer_idx):
        """Decide whether to prefetch this layer based on percentage"""
        self.prefetch_decisions += 1
        
        if self.prefetch_percentage == 0:
            self.prefetch_skipped += 1
            return False
        elif self.prefetch_percentage == 100:
            return True
        else:
            # Deterministic random decision: seed random generator with layer index
            # This creates true switching between prefetching and not prefetching
            # while maintaining deterministic behavior for consistent outputs
            random.seed(layer_idx + 42)  # +42 for additional randomness
            random_value = random.randint(1, 100)
            should_prefetch = random_value <= self.prefetch_percentage
            if not should_prefetch:
                self.prefetch_skipped += 1
            return should_prefetch

    def is_layer_in_buffer(self, layer_idx):
        """Check if the layer is in either buffer A or B"""
        return (self.buffer_a_layer == layer_idx or self.buffer_b_layer == layer_idx)
    
    def get_buffer_for_layer(self, layer_idx):
        """Get the buffer that contains the specified layer"""
        if self.buffer_a_layer == layer_idx:
            return self.buffer_a
        elif self.buffer_b_layer == layer_idx:
            return self.buffer_b
        else:
            return None

    def prefetch_layer_to_buffer(self, layer_to_prefetch_idx):
        """Prefetch a layer to the single buffer"""
        if self.prefetch_shutdown:
            return
        
        try:
            with self.buffer_lock:
                # Load the new layer into the single buffer
                self.load_layer_to_buffer(layer_to_prefetch_idx)
                
        finally:
            # PHASE 2 OPTIMIZATION: Signal completion and reuse events efficiently
            event_to_signal = None
            with self.prefetch_tracking_lock:
                event_to_signal = self.layer_prefetch_events.pop(layer_to_prefetch_idx, None)
            
            # Signal outside of lock to reduce contention
            if event_to_signal:
                event_to_signal.set()
                # PHASE 2 OPTIMIZATION: Return event to pool for reuse
                self.return_event_to_pool(event_to_signal)

    def start_prefetch_thread_to_buffer_a(self, target_layer_idx):
        """Start a prefetch thread to load layer into buffer A"""
        if self.prefetch_shutdown:
            return
        
        with self.prefetch_tracking_lock:
            self.layer_prefetch_events[target_layer_idx] = self.get_reusable_event()
            
        self.thread_pool.submit(
            self.prefetch_layer_to_buffer_a_with_event,
            target_layer_idx
        )

    def start_prefetch_thread_to_buffer_b(self, target_layer_idx):
        """Start a prefetch thread to load layer into buffer B"""
        if self.prefetch_shutdown:
            return
        
        with self.prefetch_tracking_lock:
            self.layer_prefetch_events[target_layer_idx] = self.get_reusable_event()
            
        self.thread_pool.submit(
            self.prefetch_layer_to_buffer_b_with_event,
            target_layer_idx
        )

    def trigger_prefetch_for_layer_completion(self, completed_layer_idx):
        """Trigger prefetching when a layer completes processing - dual buffer approach"""
        if self.prefetch_shutdown:
            return
            
        # DUAL BUFFER: Prefetch layer N+2 into the buffer that's not being used
        next_layer_to_prefetch = (completed_layer_idx + 2) % self.n_layer
        
        # HYBRID PREFETCHING: Check if we should prefetch this layer
        if not self.should_prefetch_layer(next_layer_to_prefetch):
            return  # Skip prefetching for this layer
        
        # Determine which buffer to use for prefetching
        # FIX: Use consistent even/odd buffer assignment
        # Even layers (0, 2, 4, ...) → Buffer A
        # Odd layers (1, 3, 5, ...) → Buffer B
        if next_layer_to_prefetch % 2 == 0:
            # Prefetch even layer into buffer A
            self.prefetch_triggers.append((completed_layer_idx, next_layer_to_prefetch, 'A'))
            self.start_prefetch_thread_to_buffer_a(next_layer_to_prefetch)
        else:
            # Prefetch odd layer into buffer B
            self.prefetch_triggers.append((completed_layer_idx, next_layer_to_prefetch, 'B'))
            self.start_prefetch_thread_to_buffer_b(next_layer_to_prefetch)

    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            # only model.layers[i].block_sparse_moe.experts is on CPU

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU (in single buffer)"""
        return self.is_layer_in_buffer(i_layer)

    def initial_beam_tensor(self, input_tensor):
        # transpose tensor of shape (beam_width, seq_len, beam_width) to (beam_width, 1) properly
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(16) # TODO: set appropriately
        
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        
        input_ids, position_ids = self.tokenize(text)
        
        # FIXED: Initialize cache AFTER tokenization so we know the batch size
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
                # TODO: streaming output for beam search
            if is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])

            logits = self.mixtral_forward(input_ids, position_ids, is_decode)

            logits = logits.to("cpu")
            # logits.shape: (batch_size, seq_len, vocab_size)

            # normalize logits
            logits = F.softmax(logits, dim=-1)

            # greedy search:
            # output = torch.argmax(logits, dim=-1)

            # beam_search:
            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            # new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
            probs = probs * new_probs

            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
            # input_ids.shape: (batch_size, seq_len=1)

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
            # position_ids.shape: (1, 1)
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

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all if self.cnt_expert_all > 0 else 0.0,
        )

    def tokenize(self, text):
        # Handle both single text and batch of texts
        if isinstance(text, list):
            # Batch processing: multiple texts
            input_ids = []
            encodings = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids_batch = encodings.input_ids.to(self.dev)
            
            # For each text in the batch, replicate for beam width
            for i in range(input_ids_batch.shape[0]):  # For each text in batch
                for j in range(self.beam_width):  # For each beam
                    input_ids.append(input_ids_batch[i])
        else:
            # Single text processing (original behavior)
            input_ids = []
            encodings = self.tokenizer(text, return_tensors="pt")
            input_id = encodings.input_ids.to(self.dev)
            for i in range(self.beam_width):
                input_ids.append(input_id[0])
        
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.dev)

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
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
            # inps.shape: (batch_size, seq_len/token_num, embed_dim)
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(-1, hidden_dim)
            # inps.shape: (batch_size*seq_len*embed_dim/hidden_dim, hidden_dim)
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            # routing_weights.shape: (batch_size*seq_len, num_experts)
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            # routing_weights.shape: (batch_size*seq_len, 2)
            # selected_experts.shape: (batch_size*seq_len, 2)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # OPTIMIZATION: Pre-allocate output tensor more efficiently
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            # OPTIMIZATION: Compute expert mask more efficiently 
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=8
            ).permute(2, 1, 0)
            
            # EXACT MATCH: Create top_2s list like in original FiddlerMixtral
            top_2s = []
            for i_expert in range(len(experts)):
                _, top_2 = torch.where(expert_mask[i_expert])
                top_2s.append(top_2)

            # DUAL BUFFER: Determine which buffer to use for current layer
            current_buffer_set = self.get_buffer_for_layer(i_layer)
            use_prefetched_buffer = current_buffer_set is not None
            
            if use_prefetched_buffer:
                self.prefetch_hits += 1
                self.layer_access_log.append((i_layer, 'direct_hit'))
            else:
                # Check if prefetch is in progress and ready
                prefetch_event = None
                if i_layer in self.layer_prefetch_events:
                    with self.prefetch_tracking_lock:
                        prefetch_event = self.layer_prefetch_events.get(i_layer)
                
                if prefetch_event is not None:
                    # INSTRUMENTATION: Time the wait operation
                    import time
                    wait_start = time.time()
                    
                    # NON-BLOCKING CHECK: Only use if prefetch is already complete
                    if prefetch_event.wait(timeout=0):  # Non-blocking check
                        wait_time = time.time() - wait_start
                        self.prefetch_wait_times.append(wait_time)
                        current_buffer_set = self.get_buffer_for_layer(i_layer)
                        use_prefetched_buffer = current_buffer_set is not None
                        if use_prefetched_buffer:
                            self.prefetch_waits += 1
                            self.layer_access_log.append((i_layer, 'prefetch_wait'))
            
            if not use_prefetched_buffer:
                # FALLBACK: Use same approach as baseline FiddlerMixtral
                current_buffer_set = [self.expert_placeholder] * 8
                self.fallback_loads += 1
                self.layer_access_log.append((i_layer, 'fallback'))
            
            # SINGLE BUFFER: Process experts on dedicated compute stream
            with torch.cuda.stream(self.compute_stream):
                # OPTIMIZATION: Process all experts more efficiently
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        continue

                    # EXACT MATCH: Use same tensor operations as original FiddlerMixtral
                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()
                    
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    
                    if use_prefetched_buffer:
                        # Use prefetched buffer directly
                        current_state = current_buffer_set[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        # FALLBACK: Load expert on-demand like baseline FiddlerMixtral
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    
                    # OPTIMIZATION: Batch update counters outside critical path
                    batch_hit_count = top_2.shape[0]
                    self.cnt_expert_hit += batch_hit_count
                    self.cnt_expert_all += batch_hit_count
                        
                    inps_after_experts.index_add_(
                        0, 
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True)
                    )
            
            # SINGLE BUFFER: Use stream event instead of blocking synchronize
            compute_event = torch.cuda.Event()
            compute_event.record(self.compute_stream)
            compute_event.wait()  # Non-blocking wait for compute completion

            # Trigger background prefetching after expert processing completes
            self.trigger_prefetch_for_layer_completion(i_layer)

            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

            # end of one layer

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis
    
    def print_prefetch_stats(self):
        """Print detailed prefetch behavior statistics"""
        total = self.prefetch_hits + self.prefetch_waits + self.fallback_loads
        print(f"\n=== HYBRID PREFETCHING ANALYSIS (Prefetch: {self.prefetch_percentage}%) ===")
        print(f"Direct buffer hits: {self.prefetch_hits}/{total} ({self.prefetch_hits/total*100:.1f}%)")  
        print(f"Prefetch waits (succeeded): {self.prefetch_waits}/{total} ({self.prefetch_waits/total*100:.1f}%)")
        print(f"Fallback loads: {self.fallback_loads}/{total} ({self.fallback_loads/total*100:.1f}%)")
        if self.prefetch_wait_times:
            avg_wait = sum(self.prefetch_wait_times) / len(self.prefetch_wait_times)
            print(f"Average wait time: {avg_wait*1000:.3f}ms")
        print(f"Total layer accesses: {total}")
        
        # HYBRID PREFETCHING: Show prefetch decision statistics
        print(f"\n=== PREFETCH DECISION ANALYSIS ===")
        print(f"Prefetch decisions made: {self.prefetch_decisions}")
        print(f"Prefetches skipped: {self.prefetch_skipped}/{self.prefetch_decisions} ({self.prefetch_skipped/self.prefetch_decisions*100:.1f}%)")
        print(f"Actual prefetch rate: {(self.prefetch_decisions - self.prefetch_skipped)/self.prefetch_decisions*100:.1f}%")
        
        # DEBUG: Print prefetch triggers
        print(f"\n=== PREFETCH TRIGGERS DEBUG ===")
        print(f"Total prefetch triggers: {len(self.prefetch_triggers)}")
        if len(self.prefetch_triggers) > 0:
            print("First 10 triggers: (completed_layer → target_layer, buffer)")
            for i, (completed, target, buffer) in enumerate(self.prefetch_triggers[:10]):
                print(f"  {i+1}: Layer {completed} → Layer {target} (Buffer {buffer})")
        
        # DEBUG: Analyze layer access patterns  
        print(f"\n=== LAYER ACCESS PATTERNS ===")
        hit_layers = [layer for layer, access_type in self.layer_access_log if access_type == 'direct_hit']
        fallback_layers = [layer for layer, access_type in self.layer_access_log if access_type == 'fallback']
        
        if hit_layers:
            print(f"Hit layers (first 20): {hit_layers[:20]}")
        if fallback_layers:
            print(f"Fallback layers (first 20): {fallback_layers[:20]}")
            
        print("="*40)

