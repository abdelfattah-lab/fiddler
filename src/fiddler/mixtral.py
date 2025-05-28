"""
FiddlerMixtral: Optimized Mixtral Model with CPU Offloading

This module implements an optimized version of the Mixtral model (a Mixture of Experts language model)
with intelligent CPU/GPU memory management. The key innovation is the ability to dynamically offload
expert networks between CPU and GPU memory to handle models larger than GPU memory.

Key Features:
- Selective GPU placement of frequently-used experts based on popularity profiling
- Dynamic CPU/GPU offloading during inference to maximize throughput
- Beam search generation support
- Memory-efficient inference for large MoE models

Usage:
    args = YourArgsClass(
        model="mistralai/Mixtral-8x7B-v0.1",
        cpu_offload=True,
        beam_width=4
    )
    model = FiddlerMixtral(args)
    prefill_time, decode_time, hit_rate = model.generate("Your prompt here", output_token=50)

Architecture Overview:
    The Mixtral model consists of:
    - Non-expert layers: embeddings, attention, layer norms (always on GPU)
    - Expert layers: 8 experts per layer, 32 layers total = 256 experts
    - Only top-2 experts are activated per token (sparse MoE)
    
    This implementation keeps popular experts on GPU and dynamically manages the rest.
"""

import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers


class FiddlerMixtral:
    """
    Optimized Mixtral model wrapper with CPU offloading capabilities.
    
    This class manages a Mixtral model by intelligently distributing expert networks
    between CPU and GPU memory. It profiles expert usage patterns and keeps frequently
    accessed experts on GPU while offloading others to CPU.
    
    Attributes:
        dtype: Model precision (default: bfloat16)
        dev: CUDA device for GPU operations
        model: The core Mixtral model (without LM head)
        lm_head: Language modeling head for token prediction
        expert_placeholder: Template expert network for CPU inference
        tokenizer: Tokenizer for text processing
        beam_width: Number of beams for beam search generation
        n_layer: Number of transformer layers (32 for Mixtral)
        n_expert: Number of experts per layer (8 for Mixtral)
        expert_loc: 2D array tracking expert locations (0=CPU, 1=GPU)
        latency_cpu: Estimated latency per token on CPU (ms)
        latency_gpu: Estimated latency for GPU transfer (ms)
    """
    
    def __init__(self, args):
        """
        Initialize the FiddlerMixtral model with optimized expert placement.
        
        Args:
            args: Configuration object with:
                - model: HuggingFace model name/path
                - cpu_offload: Whether to enable CPU offloading
                - beam_width: Beam width for generation
        """
        # Model configuration
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        
        # Load the pretrained Mixtral model
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            # device_map='cpu',  # Initially load to CPU to manage memory
            use_cache=True,
        )
        
        # Separate LM head from main model for easier management
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        
        # Create a template expert on GPU for CPU inference
        # This avoids repeated CPU->GPU transfers of expert weights (I think this comment is wrong. Weights would still need to be transferred when we know the correct expert to use)
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        
        # Initialize tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # KV cache for autoregressive generation
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        
        # Configuration
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
       
        # TODO: find this value based on device config
        # Latency estimates for offloading decisions
        self.latency_cpu = 7  # ms per token on CPU
        self.latency_gpu = 70  # ms for CPU->GPU transfer

        # Expert hit rate tracking for profiling
        self.cnt_expert_hit = 0  # Count of tokens processed by GPU experts
        self.cnt_expert_all = 0  # Total token count

        # Step 1: Move all non-expert components to GPU
        self.bring_non_expert_to_gpu()

        # Step 2: Determine expert placement strategy
        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )

        # Step 3: Mark popular experts for GPU placement
        self.set_expert_loc(n_expert_on_gpu)
        # print(self.expert_loc)

        # Step 4: Move selected experts to GPU
        self.bring_expert_to_gpu()

        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        """
        Move all non-expert model components to GPU.
        
        This includes embeddings, attention layers, layer norms, and gating networks.
        Only the expert FFN networks remain on CPU (if offloading is enabled).
        """
        # Move token prediction head
        self.lm_head.to(self.dev)
        
        # Move embeddings and final norm
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        
        # Move each layer's non-expert components
        for i in range(len(self.model.layers)):
            # Attention components
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            
            # MoE gating network (router)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            
            # Note: model.layers[i].block_sparse_moe.experts remains on CPU

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """
        Determine which experts should be placed on GPU based on usage patterns.
        
        Uses profiling data to identify the most frequently accessed experts
        and marks them for GPU placement.
        
        Args:
            n_expert_on_gpu: Number of experts that fit in GPU memory
            popular_experts: Optional pre-computed popularity ranking.
                            Format: List of (layer_idx, expert_idx) tuples
                            ordered by usage frequency
        """
        if popular_experts is None:
            # Pre-computed popularity ranking from profiling runs
            # Format: (layer_index, expert_index) ordered by frequency of use
            # This was determined by running the model on representative data
            popular_experts = [
                (9, 5),
                (11, 2),
                (10, 4),
                (28, 0),
                (13, 1),
                (17, 7),
                (12, 1),
                (8, 6),
                (16, 1),
                (9, 0),
                (14, 5),
                (19, 5),
                (26, 2),
                (30, 7),
                (7, 1),
                (3, 7),
                (23, 4),
                (22, 1),
                (29, 3),
                (1, 5),
                (13, 0),
                (5, 1),
                (18, 0),
                (4, 7),
                (10, 3),
                (1, 2),
                (3, 0),
                (8, 3),
                (11, 0),
                (11, 5),
                (11, 1),
                (31, 4),
                (21, 0),
                (25, 1),
                (15, 5),
                (22, 4),
                (27, 5),
                (16, 7),
                (15, 1),
                (13, 2),
                (15, 4),
                (21, 1),
                (27, 7),
                (9, 7),
                (7, 4),
                (31, 5),
                (2, 1),
                (11, 6),
                (12, 3),
                (2, 4),
                (24, 2),
                (28, 2),
                (0, 2),
                (30, 2),
                (6, 0),
                (6, 7),
                (15, 6),
                (6, 2),
                (14, 2),
                (2, 0),
                (17, 2),
                (19, 2),
                (24, 0),
                (10, 0),
                (19, 4),
                (1, 4),
                (26, 3),
                (31, 7),
                (17, 6),
                (25, 3),
                (12, 6),
                (0, 0),
                (26, 0),
                (29, 7),
                (27, 2),
                (19, 6),
                (5, 0),
                (18, 2),
                (20, 1),
                (12, 4),
                (17, 5),
                (5, 4),
                (30, 6),
                (20, 5),
                (24, 6),
                (25, 2),
                (28, 4),
                (4, 6),
                (7, 2),
                (20, 3),
                (23, 2),
                (8, 4),
                (30, 0),
                (3, 4),
                (12, 5),
                (23, 7),
                (1, 7),
                (22, 5),
                (18, 4),
                (31, 0),
                (17, 0),
                (0, 5),
                (14, 6),
                (0, 3),
                (15, 7),
                (5, 6),
                (4, 4),
                (24, 7),
                (31, 1),
                (27, 6),
                (22, 2),
                (14, 1),
                (1, 0),
                (29, 1),
                (21, 3),
                (25, 7),
                (22, 3),
                (7, 3),
                (2, 6),
                (29, 5),
                (28, 3),
                (6, 6),
                (7, 5),
                (5, 7),
                (8, 5),
                (20, 4),
                (21, 5),
                (18, 7),
                (27, 0),
                (16, 0),
                (24, 5),
                (12, 2),
                (2, 2),
                (24, 3),
                (4, 1),
                (29, 0),
                (3, 1),
                (21, 6),
                (10, 2),
                (20, 7),
                (19, 0),
                (26, 7),
                (20, 6),
                (23, 3),
                (4, 3),
                (30, 1),
                (1, 6),
                (29, 2),
                (30, 3),
                (0, 6),
                (8, 1),
                (25, 6),
                (29, 4),
                (16, 2),
                (23, 1),
                (26, 1),
                (26, 6),
                (16, 4),
                (2, 5),
                (0, 4),
                (7, 6),
                (14, 4),
                (3, 6),
                (20, 0),
                (18, 3),
                (4, 5),
                (17, 4),
                (0, 1),
                (16, 5),
                (19, 3),
                (23, 0),
                (30, 4),
                (20, 2),
                (13, 6),
                (18, 6),
                (15, 2),
                (3, 5),
                (22, 0),
                (10, 1),
                (9, 6),
                (10, 5),
                (25, 4),
                (9, 2),
                (18, 1),
                (6, 4),
                (4, 2),
                (23, 5),
                (6, 5),
                (21, 2),
                (5, 5),
                (6, 1),
                (26, 5),
                (12, 0),
                (25, 0),
                (4, 0),
                (14, 0),
                (16, 6),
                (31, 2),
                (8, 0),
                (21, 7),
                (14, 3),
                (31, 6),
                (28, 1),
                (5, 3),
                (23, 6),
                (6, 3),
                (18, 5),
                (25, 5),
                (27, 1),
                (11, 7),
                (11, 4),
                (24, 1),
                (0, 7),
                (8, 7),
                (13, 3),
                (21, 4),
                (27, 4),
                (13, 7),
                (3, 2),
                (9, 1),
                (2, 7),
                (7, 0),
                (2, 3),
                (28, 5),
                (27, 3),
                (15, 0),
                (24, 4),
                (5, 2),
                (22, 6),
                (3, 3),
                (28, 6),
                (14, 7),
                (13, 4),
                (28, 7),
                (22, 7),
                (13, 5),
                (19, 1),
                (26, 4),
                (1, 1),
                (17, 1),
                (16, 3),
                (10, 7),
                (29, 6),
                (19, 7),
                (31, 3),
                (7, 7),
                (1, 3),
                (8, 2),
                (9, 4),
                (17, 3),
                (30, 5),
                (15, 3),
                (9, 3),
                (10, 6),
                (12, 7),
                (11, 3),
            ]

        # Mark the top-N most popular experts for GPU placement
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def bring_expert_to_gpu(self):
        """
        Move experts marked for GPU placement from CPU to GPU memory.
        
        This is called after set_expert_loc() has determined which experts
        should reside on GPU based on available memory and usage patterns.
        """
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)

    def is_expert_in_gpu(self, i_layer, i_expert):
        """
        Check if a specific expert is currently on GPU.
        
        Args:
            i_layer: Layer index (0-31)
            i_expert: Expert index within layer (0-7)
            
        Returns:
            bool: True if expert is on GPU, False if on CPU
        """
        return self.expert_loc[i_layer, i_expert] == 1

    def calc_n_expert_on_gpu(self):
        """
        Calculate how many experts can fit in available GPU memory.
        
        This method:
        1. Calculates the memory footprint of a single expert
        2. Checks available GPU memory after loading non-expert components
        3. Returns the maximum number of experts that can fit
        
        Returns:
            int: Number of experts that can be placed on GPU
        """
        # Calculate parameter count for one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        
        # Get available GPU memory
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        # Reserve 5% as buffer, account for already allocated memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev)  # TODO: magic number
        
        # Each parameter uses 2 bytes (bfloat16)
        return int((free_mem) // (n_param * 2))

    def initial_beam_tensor(self, input_tensor):
        """
        Initialize beam search tensors for the first generation step.
        
        Transforms the initial logits tensor to properly initialize multiple beams
        from a single input sequence.
        
        Args:
            input_tensor: Tensor of shape (beam_width, seq_len, beam_width)
            
        Returns:
            torch.Tensor: Initialized beam tensor of shape (beam_width, 1)
        """
        # Extract last token probabilities
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        
        # Select top beam from each sequence
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        """
        Generate text using the model with beam search.
        
        This is the main entry point for text generation. It handles:
        - Tokenization of input text
        - Prefill phase (processing input tokens)
        - Decode phase (generating new tokens)
        - Beam search for better quality outputs
        
        Args:
            text: Input prompt text
            output_token: Number of tokens to generate
            input_token: Optional limit on input tokens (for testing)
            
        Returns:
            tuple: (prefill_time, decode_time, expert_hit_rate)
                - prefill_time: Time to process input prompt (seconds)
                - decode_time: Time to generate output tokens (seconds)
                - expert_hit_rate: Fraction of expert calls that hit GPU cache
        """
        # Configure PyTorch for CPU inference
        torch.set_num_threads(16)  # TODO: set appropriately
        
        # Reset KV cache for new generation
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        # Reset profiling counters
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        
        # Tokenize input
        input_ids, position_ids = self.tokenize(text)

        # Optionally truncate input for testing
        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        # Generation loop
        tick = time.time()
        is_decode = False  # False = prefill phase, True = decode phase
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False  # Beam search initialization flag
        probs = torch.full((input_ids.shape[0], 1), 1.0)  # Cumulative beam probabilities

        for i_token in range(output_token):
            if self.beam_width == 1:
                # Greedy decoding - show progress
                print(self.tokenizer.decode(input_ids[0]))
                # TODO: streaming output for beam search
            if is_decode:
                # Track generated text for each beam
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])

            # Forward pass through model
            logits = self.mixtral_forward(input_ids, position_ids, is_decode)

            # Move logits to CPU for token selection
            logits = logits.to("cpu")
            # logits.shape: (batch_size, seq_len, vocab_size)

            # Convert logits to probabilities
            logits = F.softmax(logits, dim=-1)

            # Token selection strategy
            # greedy search:
            # output = torch.argmax(logits, dim=-1)

            # Beam search implementation
            self.past_key_values_length += logits.shape[1]
            if search_start:
                # Continue beam search - select top-1 from each beam
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                # Initialize beam search - expand to beam_width candidates
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            
            # Update cumulative probabilities
            # new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
            probs = probs * new_probs

            # Prepare next input
            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
            # input_ids.shape: (batch_size, seq_len=1)

            # Update position IDs for next token
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
            
            # Track timing for prefill vs decode phases
            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
            
        decode_time = time.time() - tick
        
        # Select best beam based on cumulative probability
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        # Print results
        print("--------------------")
        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
        )

    def tokenize(self, text):
        """
        Tokenize input text and prepare for beam search.
        
        Creates multiple copies of the input for parallel beam processing.
        
        Args:
            text: Input text string
            
        Returns:
            tuple: (input_ids, position_ids)
                - input_ids: Token IDs tensor [beam_width, seq_len]
                - position_ids: Position encodings [beam_width, seq_len]
        """
        input_ids = []
        
        # Tokenize text
        encodings = self.tokenizer(text, return_tensors="pt")
        input_id = encodings.input_ids.to(self.dev)
        
        # Duplicate for each beam
        for i in range(self.beam_width):
            input_ids.append(input_id[0])
        
        # Pad sequences to same length
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.dev)

        # Create position IDs
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
        """
        Custom forward pass through Mixtral model with expert routing.
        
        This method implements the core MoE logic:
        1. Process tokens through embeddings and attention
        2. Route each token to top-2 experts based on gating network
        3. Dynamically offload expert computation between CPU/GPU
        4. Combine expert outputs and continue to next layer
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position encodings
            is_decode: Whether in decode phase (affects caching)
            
        Returns:
            torch.Tensor: Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        hidden_dim = self.model.config.hidden_size
        
        # Embed input tokens
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps)

        # Process through transformer layers
        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape

            # Self-attention block with residual connection
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
            
            # MoE FFN block
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            
            # Flatten for expert routing
            inps = inps.view(-1, hidden_dim)
            # inps.shape: (batch_size*seq_len, hidden_dim)
            
            # Expert routing
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            # routing_weights.shape: (batch_size*seq_len, num_experts)
            
            # Select top-2 experts per token
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            # routing_weights.shape: (batch_size*seq_len, 2)
            # selected_experts.shape: (batch_size*seq_len, 2)
            
            # Normalize routing weights
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # Buffer for expert outputs
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts


            if self.cpu_offload == 0:
                # Baseline: all experts on GPU (no offloading)
                
                # Create one-hot mask for expert assignment
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                # Process each expert
                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    
                    # Find tokens assigned to this expert
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        # No tokens assigned to this expert
                        continue

                    # torch.cuda.synchronize()
                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    # Extract tokens for this expert
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    
                    if not is_cuda:
                        # Expert on CPU - use placeholder
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        # Expert on GPU - direct computation
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    
                    # Accumulate weighted expert outputs
                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        # Ensure expert stays on CPU
                        experts[i_expert] = experts[i_expert].to("cpu")

                    # end of one expert

            else:
                # Advanced: Dynamic CPU/GPU offloading based on workload
                
                # Create expert assignment mask
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                # Step 1: Calculate workload and cost for each expert
                idxs, top_2s = [], []
                cost_per_expert = np.zeros(
                    (len(experts), 2), dtype=float
                )  # columns: [CPU_cost, GPU_cost]
                
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])
                    idxs.append(idx)
                    top_2s.append(top_2)
                    
                    # CPU cost: proportional to number of tokens
                    cost_per_expert[i_expert, 0] = top_2.shape[0] * self.latency_cpu
                    
                    # GPU cost: transfer overhead (constant) or zero if already on GPU
                    cost_per_expert[i_expert, 1] = self.latency_gpu
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # Expert already on GPU - no transfer cost
                        cost_per_expert[i_expert, 1] = 0
                        self.cnt_expert_hit += top_2.shape[0]
                    self.cnt_expert_all += top_2.shape[0]
                
                # Step 2: Find optimal CPU/GPU assignment to minimize total latency
                # We want to minimize: max(sum_cpu_costs, sum_gpu_costs)
                # Using exhaustive search since we only have 8 experts
                best_config = -1
                best_cost = float("inf")
                
                # Try all 2^8 possible configurations
                for config in range(1 << len(experts)):
                    sum_cost = 0
                    for i_expert in range(len(experts)):
                        if (config >> i_expert) & 1:
                            # Expert assigned to CPU
                            sum_cost += cost_per_expert[i_expert, 0]
                        else:
                            # Expert assigned to GPU
                            sum_cost += cost_per_expert[i_expert, 1]
                    if sum_cost < best_cost:
                        best_cost = sum_cost
                        best_config = config

                # Step 3: Partition experts based on optimal configuration
                cpu_experts = []
                gpu_experts = []
                for i_expert in range(8):
                    if (best_config >> i_expert) & 1:
                        cpu_experts.append(i_expert)
                    else:
                        gpu_experts.append(i_expert)

                # Step 4: Process GPU experts first (can overlap with CPU)
                for i_expert in gpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # Direct GPU computation
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        # Load to GPU placeholder and compute
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    
                    # Non-blocking accumulation
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

                # Step 5: Process CPU experts
                for i_expert in cpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    
                    # CPU computation
                    current_state = self.run_expert_at_cpu(
                        i_layer,
                        i_expert,
                        current_state.to("cpu"),
                        routing_weights[top_2_list, idx_list, None].to("cpu"),
                    )
                    
                    # Transfer back and accumulate
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

            # Apply residual connection over MoE layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

            # end of one layer

        # Final layer norm and output projection
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        # Update KV cache for next token
        self.present_key_value = present_key_value
        return lm_logis

    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """
        Execute an expert network on CPU.
        
        This is used when an expert is offloaded to CPU memory to save GPU space.
        The computation happens on CPU and results are transferred back.
        
        Args:
            i_layer: Layer index
            i_expert: Expert index within the layer
            inps: Input activations (on CPU)
            routing_weights: Routing weights for this expert (on CPU)
            
        Returns:
            torch.Tensor: Expert output (on CPU)
        """
        return self.model.layers[i_layer].block_sparse_moe.experts[i_expert](
            inps, routing_weights
        )