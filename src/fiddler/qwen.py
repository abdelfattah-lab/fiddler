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
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Set padding side to left for decoder-only models
        self.tokenizer.padding_side = 'left'

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

        # Keep first 2 MoE layers (0-1) permanently on GPU
        # Since layer N predicts layer N+2 in prefetch, layers 0-1 are never prefetched
        # So it makes sense to keep them on GPU for faster access
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

        print("✅ Expert management ready")

    def _pin_cpu_experts(self):
        """Pin all CPU expert parameters in memory for faster transfers."""
        print("📌 Pinning CPU expert parameters...")
        pinned_count = 0

        for layer_idx in self.moe_layers:
            # Skip GPU-resident layers (0-1)
            if layer_idx in self.gpu_resident_layers:
                continue

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

        print(f"✅ Pinned {pinned_count} expert pa rameters")

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
        moe_layer = self.model.model.layers[layer_idx].mlp

        # Get dimensions
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Fiddler mode: execute on CPU for small batches
        if self.use_fiddler_mode and batch_size < self.fiddler_batch_threshold:
            return self._moe_forward_cpu(hidden_states, layer_idx, moe_layer, batch_size, sequence_length, hidden_dim)

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

        # Process each active expert
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

            # Check if this is a GPU-resident layer (0-1)
            if layer_idx in self.gpu_resident_layers:
                # Use expert directly from GPU (no loading needed)
                expert_buffer = moe_layer.experts[expert_idx]
                # GPU-resident layers are always "hits" (no loading needed)
                self.cnt_expert_hit += len(top_x)
                self.cnt_expert_all += len(top_x)
            else:
                # Load expert to GPU buffer and execute
                expert_buffer = self._get_expert_for_execution(layer_idx, expert_idx)
                # Update statistics for buffer-loaded experts
                self.cnt_expert_all += len(top_x)
                if self.current_expert == (layer_idx, expert_idx):
                    self.cnt_expert_hit += len(top_x)

            # Execute expert computation on GPU
            if current_state.device != expert_buffer.gate_proj.weight.device:
                current_state = current_state.to(expert_buffer.gate_proj.weight.device)

            expert_output = expert_buffer(current_state)

            # Apply routing weights
            current_hidden_states = expert_output * expert_routing_weights.to(expert_output.device)

            # Move back to original device if needed and accumulate
            if current_hidden_states.device != final_hidden_states.device:
                current_hidden_states = current_hidden_states.to(final_hidden_states.device)

            # Ensure dtype consistency before accumulation
            current_hidden_states = current_hidden_states.to(final_hidden_states.dtype)

            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        # Add shared expert (shared expert stays on CPU since model is on CPU)
        shared_expert_output = moe_layer.shared_expert(hidden_states_flat)
        shared_expert_gate = F.sigmoid(moe_layer.shared_expert_gate(hidden_states_flat))
        shared_expert_output = shared_expert_gate * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        # Reshape back to original dimensions
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        # Return in the same format as original forward (output, router_logits)
        router_logits = router_logits.view(batch_size, sequence_length, -1)
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

            # Check if this layer's experts are on GPU (GPU-resident layers)
            expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]
            if layer_idx in self.gpu_resident_layers:
                # Expert is on GPU, need to move data to GPU and back to CPU
                current_state_gpu = current_state.to(self.device)
                expert_output = expert(current_state_gpu).cpu()
            else:
                # Execute expert on CPU directly
                expert_output = expert(current_state)

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
        """Generate text with expert management tracking."""
        # Handle text input
        if text is None:
            text = "The capital of France is"

        # Handle batched inputs with padding
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
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

        start_time = time.time()

        with torch.no_grad():
            # Generate with our hooked model
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_token,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        total_time = time.time() - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate hit rate
        hit_rate = self.cnt_expert_hit / self.cnt_expert_all if self.cnt_expert_all > 0 else 0.0

        # Store for comparison
        self.last_generated_text = generated_text

        # For now, approximate prefill vs decode timing
        prefill_time = total_time * 0.3  # Rough approximation
        decode_time = total_time * 0.7

        print(f"Generated: {generated_text}")

        return (prefill_time, decode_time, hit_rate)

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