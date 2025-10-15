#!/usr/bin/env python3
"""
FiddlerQwen - Baseline implementation with single expert buffer
Uses device_map="auto" for correct placement, then adds expert management
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import copy
from types import MethodType


class FiddlerQwen:
    def __init__(self, args):
        """
        Initialize Qwen model with single expert buffer.
        Uses device_map="auto" for correct device placement, then adds expert management.
        """
        self.model_name = args.model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.beam_width = args.beam_width
        self.cpu_offload = args.cpu_offload

        # For testing - respect max_experts_gpu=0
        if hasattr(args, 'max_experts_gpu'):
            self.max_experts_gpu = args.max_experts_gpu

        # Fiddler mode: execute experts on CPU for small batches
        self.use_fiddler_mode = getattr(args, 'use_fiddler_mode', False)
        self.fiddler_batch_threshold = getattr(args, 'fiddler_batch_threshold', 8)  # Default: use CPU for batch < 8

        # Cost model parameters (mirrors Mixtral implementation behaviour)
        latency_cpu = getattr(args, 'latency_cpu', None)
        latency_gpu = getattr(args, 'latency_gpu', None)
        self.latency_cpu = float(latency_cpu) if latency_cpu is not None else 0.05  # ms per token on CPU
        self.latency_gpu = float(latency_gpu) if latency_gpu is not None else 5.0   # ms constant transfer cost

        print(f"🚀 Loading {self.model_name} with Fiddler expert management")

        # Load model with correct device placement
        self._load_model()

        # Analyze structure
        self._analyze_model_structure()

        # Hook MoE layers for expert fetching
        self._hook_moe_layers()

        # Move non-expert parts to GPU (similar to Mixtral approach)
        self._bring_non_expert_to_gpu()

        # Set up expert management AFTER moving non-experts to GPU
        self._setup_expert_management()

        print("✅ Model ready with Fiddler expert management")

    def _load_model(self):
        """Load model on CPU, preserving integrated structure."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='left'  # Use left-padding for decoder-only models
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model entirely on CPU to preserve integrated structure
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            use_cache=True
        )

        print("✅ Model loaded on CPU (preserving structure integrity)")



    def _analyze_model_structure(self):
        """Analyze model structure."""
        self.n_layers = len(self.model.model.layers)
        self.moe_layers = []

        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                self.moe_layers.append(i)
                if not hasattr(self, 'n_expert'):
                    self.n_expert = len(layer.mlp.experts)

        print(f"📊 Found {len(self.moe_layers)} MoE layers with {self.n_expert} experts each")

    def _bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU (similar to Mixtral approach)."""
        print("🔄 Moving non-expert layers to GPU...")

        # Move top-level model parts to GPU
        self.model.lm_head.to(self.device)
        self.model.model.embed_tokens.to(self.device)
        self.model.model.norm.to(self.device)

        # Move layer components to GPU, but keep experts on CPU
        for i in range(len(self.model.model.layers)):
            layer = self.model.model.layers[i]
            # Move attention and normalization layers to GPU
            layer.self_attn.to(self.device)
            layer.input_layernorm.to(self.device)
            layer.post_attention_layernorm.to(self.device)

            # For MoE layers, move gate and shared expert to GPU, but keep experts on CPU
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                layer.mlp.gate.to(self.device)
                layer.mlp.shared_expert.to(self.device)
                layer.mlp.shared_expert_gate.to(self.device)
                # layer.mlp.experts remains on CPU

        print("✅ Non-expert layers moved to GPU, experts remain on CPU")

    def _setup_expert_management(self):
        """Set up expert management system with GPU buffer."""
        print("🔧 Setting up expert management...")

        if len(self.moe_layers) > 0:
            # Create GPU buffer for single expert
            # Get a sample expert (now on CPU) to create the buffer template
            sample_layer = self.model.model.layers[self.moe_layers[0]]
            sample_expert_cpu = sample_layer.mlp.experts[0]

            # Create expert buffer on GPU by copying CPU expert structure and moving to GPU
            import copy
            self.expert_buffer = copy.deepcopy(sample_expert_cpu)
            self.expert_buffer.to(self.device, dtype=self.dtype)
            print(f"Created GPU expert buffer on {self.device} with dtype {self.dtype}")

            # Pin all CPU expert parameters for faster async transfers
            self._pin_cpu_experts()

        # Statistics
        self.expert_fetch_count = 0
        self.expert_hit_count = 0
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.current_expert = None  # Track which expert is loaded in buffer

        # Timing statistics for prefill vs decode
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.decode_token_count = 0
        self.is_first_forward = True  # Track if we're in prefill phase

        print("✅ Expert management ready")

    def _pin_cpu_experts(self):
        """Pin all CPU expert parameters in memory for faster transfers."""
        print("📌 Pinning CPU expert parameters...")
        pinned_count = 0

        for layer_idx in self.moe_layers:
            layer = self.model.model.layers[layer_idx]
            for expert_idx, expert in enumerate(layer.mlp.experts):
                for param in expert.parameters():
                    if param.device.type == 'cpu':
                        # Create pinned memory tensor and copy data
                        pinned_param = torch.empty_like(param, pin_memory=True)
                        pinned_param.copy_(param)
                        # Replace the parameter data with pinned version
                        param.data = pinned_param
                        pinned_count += 1

        print(f"✅ Pinned {pinned_count} expert parameters")

    def _hook_moe_layers(self):
        """Hook into MoE layers to add expert management."""
        print("🔗 Hooking into MoE layers...")

        for layer_idx in self.moe_layers:
            layer = self.model.model.layers[layer_idx]

            # Save original forward
            layer.mlp.original_forward = layer.mlp.forward

            def create_hooked_forward(layer_idx):
                def hooked_forward(hidden_states):
                    return self._moe_forward_with_management(hidden_states, layer_idx)
                return hooked_forward

            # Replace forward method
            layer.mlp.forward = create_hooked_forward(layer_idx)

        print("✅ MoE layers hooked")

    def _moe_forward_with_management(self, hidden_states, layer_idx):
        """MoE forward with expert management and CPU-to-GPU loading."""
        # Start timing for this layer
        layer_start_time = time.time()

        moe_layer = self.model.model.layers[layer_idx].mlp

        # Get dimensions
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Determine if we're in prefill (sequence_length > 1) or decode (sequence_length == 1) phase
        is_prefill = sequence_length > 1

        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router computation (this stays on CPU since gate is on CPU)
        router_logits = moe_layer.gate(hidden_states_flat)

        # Routing weights computation
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        if moe_layer.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Convert routing weights to the correct dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Initialize output tensor with correct dtype
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # One-hot encode selected experts
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=moe_layer.num_experts).permute(2, 1, 0)

        # Find active experts
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Gather expert data for potential CPU/GPU partitioning
        expert_data = {}
        expert_token_counts = {}

        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor.item()

            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            if len(top_x) == 0:
                continue

            current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
            expert_routing_weights = routing_weights[top_x, idx, None]

            expert_data[expert_idx] = {
                'idx': idx,
                'top_x': top_x,
                'input': current_state,
                'weights': expert_routing_weights
            }
            expert_token_counts[expert_idx] = top_x.shape[0]

        cpu_expert_list = []
        gpu_expert_list = list(expert_data.keys())

        # Use cost model to decide CPU/GPU assignment when Fiddler mode is enabled
        if expert_data and self.use_fiddler_mode and batch_size < self.fiddler_batch_threshold:
            cost_per_expert = self._calculate_expert_costs(expert_token_counts)
            if cost_per_expert:
                cpu_expert_list, gpu_expert_list = self._partition_experts_greedy(cost_per_expert)
                # Ensure GPU list includes any experts not present in the cost map (safety guard)
                gpu_expert_list = [idx for idx in expert_data.keys() if idx not in cpu_expert_list]

        # Execute experts assigned to CPU
        for expert_idx in cpu_expert_list:
            data = expert_data[expert_idx]
            top_x = data['top_x']

            cpu_output = self._run_expert_on_cpu(
                layer_idx,
                expert_idx,
                data['input'],
                data['weights']
            )

            if cpu_output.device != final_hidden_states.device:
                cpu_output = cpu_output.to(final_hidden_states.device)
            cpu_output = cpu_output.to(final_hidden_states.dtype)
            final_hidden_states.index_add_(0, top_x, cpu_output)

            # Note: CPU experts are NOT counted in cnt_expert_all since hit rate only tracks GPU experts

        # Execute experts assigned to GPU (default path)
        for expert_idx in gpu_expert_list:
            data = expert_data[expert_idx]
            top_x = data['top_x']

            current_state = data['input']
            expert_routing_weights = data['weights']

            expert_buffer = self._get_expert_for_execution(layer_idx, expert_idx)

            if current_state.device != expert_buffer.gate_proj.weight.device:
                current_state = current_state.to(expert_buffer.gate_proj.weight.device)

            expert_output = expert_buffer(current_state)
            current_hidden_states = expert_output * expert_routing_weights.to(expert_output.device)

            if current_hidden_states.device != final_hidden_states.device:
                current_hidden_states = current_hidden_states.to(final_hidden_states.device)

            current_hidden_states = current_hidden_states.to(final_hidden_states.dtype)

            final_hidden_states.index_add_(0, top_x, current_hidden_states)

            self.cnt_expert_all += len(top_x)
            if self.current_expert == (layer_idx, expert_idx):
                self.cnt_expert_hit += len(top_x)

        # Add shared expert (shared expert stays on CPU since model is on CPU)
        shared_expert_output = moe_layer.shared_expert(hidden_states_flat)
        shared_expert_gate = F.sigmoid(moe_layer.shared_expert_gate(hidden_states_flat))
        shared_expert_output = shared_expert_gate * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        # Reshape back to original dimensions
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

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

        return final_hidden_states, router_logits

    def _moe_forward_cpu(self, hidden_states, layer_idx, moe_layer, batch_size, sequence_length, hidden_dim):
        """Execute MoE forward entirely on CPU for small batches (Fiddler mode)."""
        # Save original device
        original_device = hidden_states.device

        # Move hidden states to CPU if needed
        hidden_states_cpu = hidden_states.cpu() if hidden_states.device != torch.device('cpu') else hidden_states
        hidden_states_flat = hidden_states_cpu.view(-1, hidden_dim)

        # Router computation on CPU (gate is on GPU, so move input temporarily)
        router_logits = moe_layer.gate(hidden_states_flat.to(moe_layer.gate.weight.device)).cpu()

        # Routing weights computation
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        if moe_layer.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states_cpu.dtype)

        # Initialize output tensor on CPU
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states_cpu.dtype,
            device='cpu'
        )

        # One-hot encode selected experts
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=moe_layer.num_experts).permute(2, 1, 0)

        # Find active experts
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Process each active expert on CPU
        for i, expert_idx_tensor in enumerate(expert_hit):
            expert_idx = expert_idx_tensor.item()

            # Find tokens assigned to this expert
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            if len(top_x) == 0:
                continue

            # Get input for this expert
            current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)

            # Get routing weights for this expert
            expert_routing_weights = routing_weights[top_x, idx, None]

            # Execute expert on CPU directly
            cpu_expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]
            expert_output = cpu_expert(current_state)

            # Apply routing weights and accumulate
            current_hidden_states = expert_output * expert_routing_weights
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(final_hidden_states.dtype))

        # Add shared expert (shared expert is on GPU, so need to move data temporarily)
        hidden_states_for_shared = hidden_states_flat.to(moe_layer.shared_expert.gate_proj.weight.device)
        shared_expert_output = moe_layer.shared_expert(hidden_states_for_shared)
        shared_expert_gate = F.sigmoid(moe_layer.shared_expert_gate(hidden_states_for_shared))
        shared_expert_output = (shared_expert_gate * shared_expert_output).cpu()

        final_hidden_states = final_hidden_states + shared_expert_output

        # Reshape back to original dimensions
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        # Move result back to original device
        if original_device != torch.device('cpu'):
            final_hidden_states = final_hidden_states.to(original_device)

        router_logits = router_logits.view(batch_size, sequence_length, -1)
        # Move router_logits back to original device too
        if original_device != torch.device('cpu'):
            router_logits = router_logits.to(original_device)

        return final_hidden_states, router_logits

    def _calculate_expert_costs(self, expert_token_counts):
        """Compute CPU and GPU cost estimates for each expert."""
        latency_cpu = self.latency_cpu if self.latency_cpu is not None else 0.05
        latency_gpu = self.latency_gpu if self.latency_gpu is not None else 5.0

        cost_per_expert = {}
        for expert_idx, num_tokens in expert_token_counts.items():
            if num_tokens <= 0:
                continue

            cpu_cost = float(num_tokens) * latency_cpu
            gpu_cost = latency_gpu
            cost_per_expert[expert_idx] = (cpu_cost, gpu_cost)

        return cost_per_expert

    def _partition_experts_greedy(self, cost_per_expert):
        """Greedy partitioning that mirrors the Mixtral implementation."""
        if not cost_per_expert:
            return [], []

        benefits = []
        for expert_idx, (cpu_cost, gpu_cost) in cost_per_expert.items():
            benefit = gpu_cost - cpu_cost
            benefits.append((benefit, expert_idx, cpu_cost, gpu_cost))

        benefits.sort(reverse=True)

        cpu_experts = []
        gpu_experts = list(cost_per_expert.keys())

        cpu_cost_total = 0.0
        gpu_cost_total = sum(cost_per_expert[idx][1] for idx in gpu_experts)
        current_bottleneck = max(cpu_cost_total, gpu_cost_total)

        for benefit, expert_idx, cpu_cost, gpu_cost in benefits:
            if benefit <= 0:
                break

            new_cpu_cost = cpu_cost_total + cpu_cost
            new_gpu_cost = gpu_cost_total - gpu_cost
            new_bottleneck = max(new_cpu_cost, new_gpu_cost)

            if new_bottleneck < current_bottleneck:
                cpu_experts.append(expert_idx)
                gpu_experts.remove(expert_idx)
                cpu_cost_total = new_cpu_cost
                gpu_cost_total = new_gpu_cost
                current_bottleneck = new_bottleneck

        return cpu_experts, gpu_experts

    def _run_expert_on_cpu(self, layer_idx, expert_idx, input_tensor, routing_weights):
        """Execute a single expert on CPU and return its weighted contribution."""
        moe_layer = self.model.model.layers[layer_idx].mlp
        cpu_expert = moe_layer.experts[expert_idx]

        input_cpu = input_tensor.cpu()
        with torch.no_grad():
            expert_output_cpu = cpu_expert(input_cpu)

        expert_output_gpu = expert_output_cpu.to(self.device, dtype=self.dtype)
        weighted_output = expert_output_gpu * routing_weights.to(expert_output_gpu.device)

        return weighted_output

    def _get_expert_for_execution(self, layer_idx, expert_idx):
        """Get expert for execution, loading from CPU to GPU buffer if needed."""
        expert_key = (layer_idx, expert_idx)

        # Check if expert is already loaded in buffer
        if self.current_expert == expert_key:
            return self.expert_buffer
        else:
            # Load expert from CPU to GPU buffer
            cpu_expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]

            # Copy CPU expert weights to GPU buffer with correct dtype and pinned memory
            state_dict = cpu_expert.state_dict()
            converted_state_dict = {}

            for key, tensor in state_dict.items():
                # CRITICAL: state_dict() returns copies that might not be pinned
                # Always ensure pinned memory for transfers
                if not tensor.is_pinned():
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

            self.expert_buffer.load_state_dict(converted_state_dict)
            self.current_expert = expert_key

        return self.expert_buffer

    def generate(self, text=None, output_token=20, input_token=None):
        """
        Generate text with expert management tracking.

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

        input_ids = inputs.input_ids.to(self.device)  # Move to GPU since model is now on GPU
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None

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

        # Reset timing statistics
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.decode_token_count = 0

        start_time = time.time()

        with torch.no_grad():
            # Generate with our hooked model
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_token,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=None,  # Force exact token count
                use_cache=True
            )

        total_time = time.time() - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate hit rate
        hit_rate = self.cnt_expert_hit / self.cnt_expert_all if self.cnt_expert_all > 0 else 0.0

        # Store for comparison
        self.last_generated_text = generated_text

        # Use actual measured times from MoE layer tracking
        # Note: self.prefill_time and self.decode_time are accumulated across all MoE layers
        prefill_time = self.prefill_time
        decode_time = self.decode_time / self.decode_token_count if self.decode_token_count else 0.0

        print(f"Generated: {generated_text}")
        print(f"⏱️  Prefill: {prefill_time:.3f}s, Decode/token: {decode_time:.3f}s")

    # Return format: (prefill_time, decode_time_per_token, prefill_hit_rate, decode_hit_rate)
        # Baseline doesn't track phase-specific hit rates, so return same rate for both
        return (prefill_time, decode_time, hit_rate, hit_rate)

    def tokenize(self, text):
        """Tokenize text - for interface compatibility."""
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # Create position_ids for compatibility
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.device
        )
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        return input_ids, position_ids

    def mixtral_forward(self, input_ids, position_ids, is_decode):
        """Forward method for interface compatibility."""
        # Use the model's forward method directly
        outputs = self.model(input_ids=input_ids, use_cache=True)
        return outputs.logits

    def get_memory_usage(self):
        """Get GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() // 1024**2,
                'cached_mb': torch.cuda.memory_reserved() // 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() // 1024**2
            }
        return {}