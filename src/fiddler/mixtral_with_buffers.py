import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers


class MixtralWithBuffers:
    def __init__(self, args, num_buffer_sets=2):
        # Validate buffer sets configuration
        if num_buffer_sets < 2:
            raise ValueError("num_buffer_sets must be at least 2")
        self.num_buffer_sets = num_buffer_sets
        
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
        # Create buffer sets - each buffer set contains 8 expert slots (one per expert in a layer)
        self.buffer_sets = []
        for i in range(self.num_buffer_sets):
            buffer_set = []
            for j in range(8):  # 8 experts per layer
                expert_buffer = copy.deepcopy(
                    self.model.layers[0].block_sparse_moe.experts[0]
                ).to(self.dev)
                buffer_set.append(expert_buffer)
            self.buffer_sets.append(buffer_set)

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

        self.bring_non_expert_to_gpu()

        # Initialize expert location tracking
        # Values: -1=CPU, 0=buffer_set_0, 1=buffer_set_1, etc.
        self.expert_loc = np.full((self.n_layer, self.n_expert), -1, dtype=int)
        
        # Load first 2 layers into buffer sets
        self.load_layer_to_buffer_set(0, 0)  # Load layer 0 into buffer set 0
        self.load_layer_to_buffer_set(1, 1)  # Load layer 1 into buffer set 1
        
        # Update expert_loc to reflect buffer set assignments
        for i_expert in range(self.n_expert):
            self.expert_loc[0, i_expert] = 0  # Layer 0 in buffer set 0
            self.expert_loc[1, i_expert] = 1  # Layer 1 in buffer set 1
            
        # Initialize threading infrastructure for background prefetching
        # Use only 2 threads: one for even layers, one for odd layers
        # With even/odd thread separation, we can reduce lock contention significantly
        self.even_buffer_lock = threading.Lock()  # Lock for even layers (buffer sets 0, 2, 4, ...)
        self.odd_buffer_lock = threading.Lock()   # Lock for odd layers (buffer sets 1, 3, 5, ...)
        self.even_thread = None  # Thread for even layer numbers (0, 2, 4, ...)
        self.odd_thread = None   # Thread for odd layer numbers (1, 3, 5, ...)
        self.prefetch_shutdown = False
        # Track which layers are currently being prefetched with events
        self.layer_prefetch_events = {}  # layer_idx -> threading.Event
        self.prefetch_tracking_lock = threading.Lock()
        
        print(f"Loaded layers 0-1 into buffer sets 0-1 ({self.n_expert * 2} experts total)")

        print("Model is ready.")

    def load_layer_to_buffer_set(self, layer_idx, buffer_set_idx):
        """Load all experts of a layer into a specific buffer set"""
        if buffer_set_idx >= self.num_buffer_sets:
            raise ValueError(f"buffer_set_idx {buffer_set_idx} >= num_buffer_sets {self.num_buffer_sets}")
        if layer_idx >= self.n_layer:
            raise ValueError(f"layer_idx L{layer_idx} >= n_layer {self.n_layer}")
            
        buffer_set = self.buffer_sets[buffer_set_idx]
        source_experts = self.model.layers[layer_idx].block_sparse_moe.experts
        
        for i_expert in range(self.n_expert):
            buffer_set[i_expert].load_state_dict(source_experts[i_expert].state_dict())

    def is_layer_in_buffer_set(self, layer_idx, buffer_set_idx):
        """Check if a layer is currently loaded in a specific buffer set"""
        if buffer_set_idx >= self.num_buffer_sets or layer_idx >= self.n_layer:
            return False
        return self.expert_loc[layer_idx, 0] == buffer_set_idx

    def get_buffer_set_for_layer(self, layer_idx):
        """Get the buffer set index for a layer, or -1 if not in any buffer set"""
        if layer_idx >= self.n_layer:
            return -1
        # Check the first expert of the layer to determine which buffer set it's in
        expert_loc = self.expert_loc[layer_idx, 0]
        if expert_loc >= 0:  # In a buffer set
            return expert_loc
        return -1  # On CPU

    def prefetch_layer_to_buffer_set(self, completing_layer_idx, layer_to_prefetch_idx, buffer_set_idx):
        """Prefetch a layer to a buffer set, replacing the completing layer"""
        if self.prefetch_shutdown:
            return
            
        # Use appropriate lock based on buffer set parity to minimize contention
        buffer_lock = self.even_buffer_lock if buffer_set_idx % 2 == 0 else self.odd_buffer_lock
        
        try:
            with buffer_lock:
                # Clear the completing layer's location
                for i_expert in range(self.n_expert):
                    self.expert_loc[completing_layer_idx, i_expert] = -1
                
                # Load the new layer into the buffer set
                self.load_layer_to_buffer_set(layer_to_prefetch_idx, buffer_set_idx)
                
                # Update expert_loc for the new layer
                for i_expert in range(self.n_expert):
                    self.expert_loc[layer_to_prefetch_idx, i_expert] = buffer_set_idx
        finally:
            # Signal completion and remove event
            with self.prefetch_tracking_lock:
                if layer_to_prefetch_idx in self.layer_prefetch_events:
                    self.layer_prefetch_events[layer_to_prefetch_idx].set()
                    del self.layer_prefetch_events[layer_to_prefetch_idx]

    def start_prefetch_thread(self, completing_layer_idx, target_layer_idx, buffer_set_idx):
        """Start a prefetch thread using even/odd thread allocation"""
        if self.prefetch_shutdown:
            return
        
        # Create event for this layer before starting thread
        with self.prefetch_tracking_lock:
            self.layer_prefetch_events[target_layer_idx] = threading.Event()
            
        thread = threading.Thread(
            target=self.prefetch_layer_to_buffer_set,
            args=(completing_layer_idx, target_layer_idx, buffer_set_idx),
            daemon=True
        )
        
        # Assign thread based on target layer parity (even/odd)
        if target_layer_idx % 2 == 0:  # Even layer
            # Wait for previous even thread to complete if it exists
            if self.even_thread is not None and self.even_thread.is_alive():
                self.even_thread.join()
            self.even_thread = thread
        else:  # Odd layer
            # Wait for previous odd thread to complete if it exists
            if self.odd_thread is not None and self.odd_thread.is_alive():
                self.odd_thread.join()
            self.odd_thread = thread
        
        thread.start()

    def trigger_prefetch_for_layer_completion(self, completed_layer_idx):
        """Trigger prefetching when a layer completes processing"""
        if self.prefetch_shutdown:
            return
            
        # Calculate which layer to prefetch and which buffer set to use
        next_layer_to_prefetch = (completed_layer_idx + 2) % self.n_layer
        buffer_set_to_use = completed_layer_idx % self.num_buffer_sets
        
        # Start prefetching in the background
        self.start_prefetch_thread(completed_layer_idx, next_layer_to_prefetch, buffer_set_to_use)

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

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """Set the location of experts - DISABLED: keeping all experts on CPU initially"""
        # This method is now disabled to keep all experts on CPU initially
        # Only buffer sets will contain experts on GPU
        pass

    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU - DISABLED: keeping all experts on CPU initially"""
        # This method is now disabled to keep all experts on CPU initially
        # Only buffer sets will contain experts on GPU
        pass

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU (either in buffer sets or old GPU location)"""
        return self.expert_loc[i_layer, i_expert] >= 0

    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # Check if max_experts_gpu is specified (for testing purposes)
        if hasattr(self, 'max_experts_gpu') and self.max_experts_gpu:
            return min(self.max_experts_gpu, self.n_layer * self.n_expert)
        
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev) # TODO: magic number
        return int((free_mem) // (n_param * 2))

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

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
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

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            # Run everything on GPU
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=8
            ).permute(2, 1, 0)

            # Only wait for prefetch if layer is not already available in buffer sets
            layer_buffer_set = self.get_buffer_set_for_layer(i_layer)
            if layer_buffer_set == -1:  # Layer not in any buffer set, may need to wait for prefetch
                print("Prefetching is not done. Waiting...")
                with self.prefetch_tracking_lock:
                    prefetch_event = self.layer_prefetch_events.get(i_layer)
                if prefetch_event is not None:
                    prefetch_event.wait()

            for i_expert in range(len(experts)):
                is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                idx, top_2 = torch.where(expert_mask[i_expert])

                if top_2.shape[0] == 0:
                    continue

                top_2_list = top_2.tolist()
                idx_list = idx.tolist()

                current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                if not is_cuda:
                    raise ValueError("Expert is not on the GPU")
                
                # Expert is on GPU - use buffer set expert (all experts should be in buffer sets now)
                buffer_set_idx = self.expert_loc[i_layer, i_expert]
                current_state = self.buffer_sets[buffer_set_idx][i_expert](
                    current_state, routing_weights[top_2_list, idx_list, None]
                )
                self.cnt_expert_hit += top_2.shape[0]
                self.cnt_expert_all += top_2.shape[0]
                    
                inps_after_experts.index_add_(
                    0, top_2, current_state.to(inps.dtype)
                )

            # Trigger background prefetching after expert processing completes
            self.trigger_prefetch_for_layer_completion(i_layer)

            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

            # end of one layer

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis

