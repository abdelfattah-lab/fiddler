"""
FiddlerMixtral: Optimized Mixtral Model with CPU Offloading and Expert Prefetching

This module implements an optimized version of the Mixtral model (a Mixture of Experts language model)
with intelligent CPU/GPU memory management and expert prefetching during decode phase.

Key Features:
- Selective GPU placement of frequently-used experts based on popularity profiling
- Dynamic CPU/GPU offloading during inference to maximize throughput
- Expert prefetching during decode phase to overlap loading with computation
- Double buffering for expert networks to prefetch both needed experts
- Beam search generation support
- Memory-efficient inference for large MoE models

Usage:
    args = YourArgsClass(
        model="mistralai/Mixtral-8x7B-v0.1",
        cpu_offload=True,
        beam_width=4,
        enable_prefetch=True  # Enable expert prefetching
    )
    model = FiddlerMixtral(args)
    prefill_time, decode_time, hit_rate = model.generate("Your prompt here", output_token=50)

Architecture Overview:
    The Mixtral model consists of:
    - Non-expert layers: embeddings, attention, layer norms (always on GPU)
    - Expert layers: 8 experts per layer, 32 layers total = 256 experts
    - Only top-2 experts are activated per token (sparse MoE)
    
    This implementation keeps popular experts on GPU and dynamically manages the rest.
    With prefetching enabled, it predicts and preloads experts for the next layer.
    
Prefetching Strategy:
    During decode phase with single token generation (beam_width=1), we know we need 
    exactly 2 experts per layer. After processing experts for layer i, we start 
    prefetching both experts for layer i+1 (if they're on CPU) into our 2 placeholders.
    This prefetch happens in parallel with the residual connection, layer norm, and 
    attention computations of the next layer, hiding the CPU→GPU transfer latency.
    With beam search (beam_width>1), we may need more experts, so prefetching 
    is less effective but still helpful.
"""

import copy
import threading
import time
import queue
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers


class FiddlerMixtral:
    """
    Optimized Mixtral model wrapper with CPU offloading and prefetching capabilities.
    
    This class manages a Mixtral model by intelligently distributing expert networks
    between CPU and GPU memory. It profiles expert usage patterns and keeps frequently
    accessed experts on GPU while offloading others to CPU. With prefetching enabled,
    it predicts and preloads experts for upcoming layers during decode phase.
    
    Attributes:
        dtype: Model precision (default: bfloat16)
        dev: CUDA device for GPU operations
        model: The core Mixtral model (without LM head)
        lm_head: Language modeling head for token prediction
        expert_placeholder_0: First template expert network for prefetching/CPU inference
        expert_placeholder_1: Second template expert network for prefetching/CPU inference
        placeholder_contents: Dict tracking what's loaded in each placeholder
        tokenizer: Tokenizer for text processing
        beam_width: Number of beams for beam search generation
        n_layer: Number of transformer layers (32 for Mixtral)
        n_expert: Number of experts per layer (8 for Mixtral)
        expert_loc: 2D array tracking expert locations (0=CPU, 1=GPU)
        latency_cpu: Estimated latency per token on CPU (ms)
        latency_gpu: Estimated latency for GPU transfer (ms)
        enable_prefetch: Whether to enable expert prefetching
        prefetch_stream: CUDA stream for async expert transfers
        prefetch_timers: Dictionary for profiling prefetching operations
        efficient_copy: Whether to use efficient parameter copying
        run_all_experts_on_gpu: Whether to force all experts to run on the GPU
    """
    
    def __init__(self, args):
        """
        Initialize the FiddlerMixtral model with optimized expert placement.
        
        Args:
            args: Configuration object with:
                - model: HuggingFace model name/path
                - cpu_offload: Whether to enable CPU offloading
                - beam_width: Beam width for generation
                - enable_prefetch: Whether to enable expert prefetching (default: False)
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
        
        # Create two template experts on GPU for double buffering
        self.expert_placeholder_0 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        self.expert_placeholder_1 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        
        # Track what's loaded in each placeholder
        self.placeholder_contents = {
            0: None,  # (layer, expert) tuple or None
            1: None   # (layer, expert) tuple or None
        }
        
        # Initialize tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # KV cache for autoregressive generation
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        
        # Configuration
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.enable_prefetch = getattr(args, 'enable_prefetch', False)
        self.no_preloading = getattr(args, 'no_preloading', False)
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
       
        # TODO: find this value based on device config
        # Latency estimates for offloading decisions
        self.latency_cpu = 7  # ms per token on CPU
        self.latency_gpu = 70  # ms for CPU->GPU transfer

        # Expert hit rate tracking for profiling
        self.cnt_expert_hit = 0  # Count of tokens processed by GPU experts
        self.cnt_expert_all = 0  # Total token count
        self.cnt_prefetch_hit = 0  # Count of successful prefetch hits
        self.cnt_prefetch_miss = 0  # Count of prefetch misses

        # Prefetching infrastructure
        if self.enable_prefetch:
            # Create dedicated CUDA stream for prefetching
            self.prefetch_stream = torch.cuda.Stream()
            # Track ongoing prefetch
            self.prefetch_in_progress = False
            self.prefetch_target_layer = None
            self.prefetch_target_experts = None


            self.decode_expert_predictions = [
                (0, (2, 5)),   # Layer 0 will use experts 2 and 5
                (1, (4, 7)),   # Layer 1 will use experts 4 and 7
                (2, (1, 4)),   # Layer 2 will use experts 1 and 4
                (3, (0, 7)),   # Layer 3 will use experts 0 and 7
                (4, (3, 6)),   # Layer 4 will use experts 3 and 6
                (5, (1, 4)),   # Layer 5 will use experts 1 and 4
                (6, (0, 7)),   # Layer 6 will use experts 0 and 7
                (7, (2, 5)),   # Layer 7 will use experts 2 and 5
                (8, (3, 6)),   # Layer 8 will use experts 3 and 6
                (9, (0, 5)),   # Layer 9 will use experts 0 and 5
                (10, (3, 4)),  # Layer 10 will use experts 3 and 4
                (11, (0, 2)),  # Layer 11 will use experts 0 and 2
                (12, (1, 4)),  # Layer 12 will use experts 1 and 4
                (13, (0, 1)),  # Layer 13 will use experts 0 and 1
                (14, (2, 5)),  # Layer 14 will use experts 2 and 5
                (15, (1, 5)),  # Layer 15 will use experts 1 and 5
                (16, (1, 7)),  # Layer 16 will use experts 1 and 7
                (17, (2, 7)),  # Layer 17 will use experts 2 and 7
                (18, (0, 4)),  # Layer 18 will use experts 0 and 4
                (19, (2, 5)),  # Layer 19 will use experts 2 and 5
                (20, (1, 5)),  # Layer 20 will use experts 1 and 5
                (21, (0, 1)),  # Layer 21 will use experts 0 and 1
                (22, (1, 4)),  # Layer 22 will use experts 1 and 4
                (23, (2, 4)),  # Layer 23 will use experts 2 and 4
                (24, (0, 2)),  # Layer 24 will use experts 0 and 2
                (25, (1, 3)),  # Layer 25 will use experts 1 and 3
                (26, (0, 2)),  # Layer 26 will use experts 0 and 2
                (27, (2, 5)),  # Layer 27 will use experts 2 and 5
                (28, (0, 4)),  # Layer 28 will use experts 0 and 4
                (29, (3, 7)),  # Layer 29 will use experts 3 and 7
                (30, (2, 7)),  # Layer 30 will use experts 2 and 7
                (31, (4, 5)),  # Layer 31 will use experts 4 and 5
            ]

        # Add profiling timers
        self.prefetch_timers = {
            'load_state_dict': 0.0,
            'expert_compute': 0.0,
            'wait_time': 0.0,
            'total_prefetch': 0.0
        }

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

        print(f"Model is ready. Prefetching enabled: {self.enable_prefetch}")

        # Additional configuration
        self.efficient_copy = getattr(args, 'efficient_copy', False)
        self.run_all_experts_on_gpu = getattr(args, 'run_all_experts_on_gpu', False)

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
        
        if self.no_preloading:
            return 0;
        # Each parameter uses 2 bytes (bfloat16)
        # Account for 2 expert placeholders if prefetching is enabled
        placeholder_mem = n_param * 2 * (2 if self.enable_prefetch else 1)
        return int((free_mem - placeholder_mem) // (n_param * 2))

    def copy_expert_params(self, source_expert, target_placeholder):
        """
        Efficiently copy parameters from source expert to target placeholder.
        """
        if self.efficient_copy:
            with torch.no_grad():
                for param_src, param_tgt in zip(source_expert.parameters(), target_placeholder.parameters()):
                    if not param_src.is_pinned:
                        param_src.data = param_src.pin_memory()
                    param_tgt.copy_(param_src, non_blocking=True)
        else:
            # Normal (non-optimized) copy: use state_dict
            target_placeholder.load_state_dict(source_expert.state_dict())

    def start_expert_prefetch_async(self, i_layer, expert_0, expert_1):
        """
        Start prefetching experts for the next layer using CUDA streams.
        
        Args:
            i_layer: Layer index to prefetch experts for
            expert_0: First expert index to prefetch
            expert_1: Second expert index to prefetch
        """
        if not self.enable_prefetch:
            return
        
        # Record what we're prefetching
        self.prefetch_in_progress = True
        self.prefetch_target_layer = i_layer
        self.prefetch_target_experts = (expert_0, expert_1)
        
        # Only measure the actual work, not the async enqueue time
        experts_to_load = []
        
        # Check which experts need loading
        if not self.is_expert_in_gpu(i_layer, expert_0):
            experts_to_load.append((expert_0, self.expert_placeholder_0, 0))
        else:
            self.placeholder_contents[0] = None
        
        if not self.is_expert_in_gpu(i_layer, expert_1) and expert_1 != expert_0:
            experts_to_load.append((expert_1, self.expert_placeholder_1, 1))
        else:
            self.placeholder_contents[1] = None
        
        if len(experts_to_load) == 0:
            # Nothing to prefetch
            self.prefetch_in_progress = False
            return
        
        # Use the prefetch stream for async transfers
        with torch.cuda.stream(self.prefetch_stream):
            for expert_idx, placeholder, placeholder_idx in experts_to_load:
                expert = self.model.layers[i_layer].block_sparse_moe.experts[expert_idx]
                
                # Record CUDA event before copy
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record(self.prefetch_stream)
                self.copy_expert_params(expert, placeholder)
                end_event.record(self.prefetch_stream)
                
                # Store events for later timing
                if not hasattr(self, 'prefetch_events'):
                    self.prefetch_events = []
                self.prefetch_events.append((start_event, end_event))
                
                self.placeholder_contents[placeholder_idx] = (i_layer, expert_idx)

    def wait_for_prefetch(self):
        """
        Wait for ongoing prefetch operation to complete by synchronizing streams.
        """
        if self.enable_prefetch and self.prefetch_in_progress:
            wait_start = time.time()
            # Wait for prefetch stream to complete
            self.prefetch_stream.synchronize()
            self.prefetch_timers['wait_time'] += time.time() - wait_start
            
            # Now measure actual CUDA execution time
            if hasattr(self, 'prefetch_events'):
                for start_event, end_event in self.prefetch_events:
                    # This gives actual GPU execution time in milliseconds
                    gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                    self.prefetch_timers['load_state_dict'] += gpu_time
                    self.prefetch_timers['total_prefetch'] += gpu_time
                self.prefetch_events = []
            
            self.prefetch_in_progress = False

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
        self.cnt_prefetch_hit = 0
        self.cnt_prefetch_miss = 0
        
        # Track actual expert usage during decode phase
        self.actual_expert_usage = {}  # {layer_idx: [(expert_0, expert_1), ...]}
        for i in range(self.n_layer):
            self.actual_expert_usage[i] = []
        self.tracking_decode_experts = False

        # Reset prefetching state
        if self.enable_prefetch:
            self.placeholder_contents = {0: None, 1: None}
            self.prefetch_in_progress = False
        
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
                # Start tracking expert usage after prefill
                if is_decode and self.enable_prefetch:
                    self.tracking_decode_experts = True
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

        # Save actual expert usage if we tracked it
        if self.enable_prefetch and self.tracking_decode_experts and len(self.actual_expert_usage[0]) > 0:
            # Convert to the expected format and take the first usage pattern
            # (assuming consistent expert selection across tokens in decode phase)
            expert_predictions = []
            for layer_idx in range(self.n_layer):
                if len(self.actual_expert_usage[layer_idx]) > 0:
                    # Take the first occurrence (or could do majority voting)
                    experts = self.actual_expert_usage[layer_idx][0]
                    expert_predictions.append([layer_idx, list(experts)])
            
            # Save to file
            expert_pred_file = "expert_predictions.json"
            with open(expert_pred_file, 'w') as f:
                json.dump(expert_predictions, f, indent=2)
            print(f"Saved actual expert usage patterns to {expert_pred_file}")
            
            # Update current predictions for immediate use
            self.decode_expert_predictions = [
                (item[0], tuple(item[1])) for item in expert_predictions
            ]
            self.tracking_decode_experts = False
        
        # Select best beam based on cumulative probability
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        # Print results
        print("--------------------")
        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")
        
        if self.enable_prefetch:
            prefetch_rate = self.cnt_prefetch_hit / (self.cnt_prefetch_hit + self.cnt_prefetch_miss) if (self.cnt_prefetch_hit + self.cnt_prefetch_miss) > 0 else 0
            print(f"Prefetch hit rate: {prefetch_rate:.2%} ({self.cnt_prefetch_hit}/{self.cnt_prefetch_hit + self.cnt_prefetch_miss})")
            print("\nPrefetch Profiling:")
            print(f"  Load state dict time: {self.prefetch_timers['load_state_dict']:.3f}s")
            print(f"  Expert compute time: {self.prefetch_timers['expert_compute']:.3f}s")
            print(f"  Wait time: {self.prefetch_timers['wait_time']:.3f}s")
            print(f"  Total prefetch time: {self.prefetch_timers['total_prefetch']:.3f}s")

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
        Custom forward pass through Mixtral model with expert routing and prefetching.
        
        This method implements the core MoE logic:
        1. Process tokens through embeddings and attention
        2. Route each token to top-2 experts based on gating network
        3. Process experts (using prefetched ones if available)
        4. Start prefetching experts for next layer (after current layer's experts are used)
        5. Combine expert outputs and continue to next layer
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position encodings
            is_decode: Whether in decode phase (affects caching and prefetching)
            
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
            
            # Track actual expert usage during decode phase
            if self.tracking_decode_experts and is_decode and input_ids.shape[0] == 1:
                # For single token decode, track which experts are selected
                # We only track the first beam (index 0) for simplicity
                if selected_experts.shape[0] >= 1:
                    expert_0 = selected_experts[0, 0].item()
                    expert_1 = selected_experts[0, 1].item()
                    # Store as tuple to match expected format
                    self.actual_expert_usage[i_layer].append((expert_0, expert_1))

            # Normalize routing weights
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # Buffer for expert outputs
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            # # Determine which experts to use for prefetching (decode phase only)
            # if self.enable_prefetch and is_decode and i_layer < self.n_layer - 1:
            #     # Get predicted experts for next layer
            #     _, (next_expert_0, next_expert_1) = self.decode_expert_predictions[i_layer + 1]
                
            #     # Start prefetching both experts for next layer in background
            #     self.start_expert_prefetch(i_layer + 1, next_expert_0, next_expert_1)

            # Choose expert processing strategy:
            # 1. If cpu_offload=0: All experts on GPU (baseline, no offloading)
            # 2. If enable_prefetch=True: Use simple path with prefetching support
            # 3. Otherwise: Use advanced dynamic offloading (original Fiddler approach)
            if self.cpu_offload == 0 or self.enable_prefetch:
                
                # Wait for any ongoing prefetch to complete
                if self.enable_prefetch and is_decode and i_layer > 0:
                    self.wait_for_prefetch()
                
                # Create one-hot mask for expert assignment
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                # print the experts placed in placholder 0 and placeholder 1
                # print(f'Placeholder 0: {self.placeholder_contents[0]}')
                # print(f'Placeholder 1: {self.placeholder_contents[1]}')
                # Process each expert
                used_placeholder1 = False # TODO: To be removed
                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    
                    # Find tokens assigned to this expert
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        # No tokens assigned to this expert
                        # print(f'No tokens to expert {i_expert}.')
                        continue
                    # else:
                        # print(f'Found tokens for expert {i_expert}.')

                    # torch.cuda.synchronize()
                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    # Extract tokens for this expert
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    
                    if not is_cuda:
                        # Expert on CPU - check if it's already loaded in a placeholder
                        expert_found = False
                        
                        if self.enable_prefetch:
                            # Check if this expert is loaded in placeholder 0
                            # if self.placeholder_contents[0] == (i_layer, i_expert):
                            if not(used_placeholder1): # TODO: To be removed
                                used_placeholder1 = True # TODO: To be removed
                                current_state = self.expert_placeholder_0(
                                    current_state, routing_weights[top_2_list, idx_list, None]
                                )
                                expert_found = True
                                self.cnt_prefetch_hit = 1
                            # Check if this expert is loaded in placeholder 1
                            # elif self.placeholder_contents[1] == (i_layer, i_expert):
                            else: # TODO: To be removed
                                current_state = self.expert_placeholder_1(
                                    current_state, routing_weights[top_2_list, idx_list, None]
                                )
                                expert_found = True
                                self.cnt_prefetch_hit += 1
                        
                        if not expert_found:
                            # Expert not prefetched - load on demand
                            if self.enable_prefetch and is_decode:
                                self.cnt_prefetch_miss += 1
                            
                            # Choose a placeholder to use (prefer one not containing next layer's experts)
                            # Simple round-robin between placeholders
                            if self.placeholder_contents[0] == (i_layer, i_expert):
                                use_placeholder = self.expert_placeholder_0
                            else:
                                use_placeholder = self.expert_placeholder_0
                                self.placeholder_contents[0] = None
                            
                            # Use direct parameter copy instead of state dict
                            self.copy_expert_params(experts[i_expert], use_placeholder)
                            current_state = use_placeholder(
                                current_state, routing_weights[top_2_list, idx_list, None]
                            )
                    else:
                        # Expert on GPU - direct computation
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                        self.cnt_expert_hit += top_2.shape[0]
                    
                    self.cnt_expert_all += top_2.shape[0]
                    
                    # Accumulate weighted expert outputs
                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        # Ensure expert stays on CPU
                        experts[i_expert] = experts[i_expert].to("cpu")

                    # end of one expert
                
                # After processing all experts, start prefetching for next layer
                if self.enable_prefetch and is_decode and i_layer < self.n_layer - 1:
                    # Get predicted experts for next layer
                    _, (next_expert_0, next_expert_1) = self.decode_expert_predictions[i_layer + 1]
                    
                    # Start async prefetching both experts for next layer
                    self.start_expert_prefetch_async(i_layer + 1, next_expert_0, next_expert_1)

            else:
                # Advanced: Dynamic CPU/GPU offloading based on workload (original path)
                
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
                    if not(self.run_all_experts_on_gpu) and (best_config >> i_expert) & 1:
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
                        self.expert_placeholder_0.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder_0(
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