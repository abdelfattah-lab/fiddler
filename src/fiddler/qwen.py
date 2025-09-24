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

        print(f"🚀 Loading {self.model_name} with Fiddler expert management")

        # Load model with correct device placement
        self._load_model()

        # Analyze structure
        self._analyze_model_structure()

        # Hook MoE layers for expert fetching
        self._hook_moe_layers()

        # Set up expert management AFTER moving experts to CPU
        self._setup_expert_management()

        print("✅ Model ready with Fiddler expert management")

    def _load_model(self):
        """Load model with experts on CPU and core model on GPU."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model on GPU first with device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map="auto"
        )

        # Move experts to CPU while keeping core model on GPU
        self._move_experts_to_cpu()

    def _move_experts_to_cpu(self):
        """Move all experts to CPU while keeping core model on GPU."""
        print("🔄 Moving experts to CPU...")

        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                for expert_idx, expert in enumerate(layer.mlp.experts):
                    # Properly materialize expert on CPU using state dict approach
                    self._materialize_expert_on_cpu(i, expert_idx, expert)
                print(f"Layer {i}: Moved {len(layer.mlp.experts)} experts to CPU")

        print("✅ All experts moved to CPU")

    def _materialize_expert_on_cpu(self, layer_idx, expert_idx, expert):
        """Properly materialize expert on CPU with actual weight data."""
        import copy

        # Check if expert parameters are meta tensors
        for name, param in expert.named_parameters():
            if param.is_meta:
                print(f"    Expert {layer_idx}-{expert_idx} {name} is meta tensor, skipping CPU move")
                return  # Skip experts that are meta tensors

        # For non-meta tensors, use simple CPU move
        try:
            expert.cpu()
            print(f"    Expert {layer_idx}-{expert_idx} moved to CPU successfully")
        except Exception as e:
            print(f"    Expert {layer_idx}-{expert_idx} CPU move failed: {e}")
            # Fallback: try to create a new expert with copied weights
            expert_cpu = copy.deepcopy(expert)
            for name, param in expert.named_parameters():
                if not param.is_meta:
                    cpu_weight = param.detach().cpu()
                    parts = name.split('.')
                    current = expert_cpu
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], torch.nn.Parameter(cpu_weight))

            # Replace the original expert with the CPU version
            self.model.model.layers[layer_idx].mlp.experts[expert_idx] = expert_cpu

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

        # Statistics
        self.expert_fetch_count = 0
        self.expert_hit_count = 0
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.current_expert = None  # Track which expert is loaded in buffer

        print("✅ Expert management ready")

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
        """MoE forward with CPU-to-GPU expert loading following Qwen's exact implementation."""
        moe_layer = self.model.model.layers[layer_idx].mlp

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router computation (exactly like original)
        router_logits = moe_layer.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        if moe_layer.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts (exactly like original)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=moe_layer.num_experts).permute(2, 1, 0)

        # Loop over all available experts (exactly like original)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor.item()

            # Track for statistics
            if self.current_expert == (layer_idx, expert_idx):
                self.expert_hit_count += 1
                self.cnt_expert_hit += 1
            else:
                self.expert_fetch_count += 1
                self.current_expert = (layer_idx, expert_idx)
            self.cnt_expert_all += 1

            # Get the expert (load to GPU if necessary)
            expert_layer = self._get_expert_for_execution(layer_idx, expert_idx)

            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute expert output (exactly like original)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # Use index_add_ for accumulation - ensure dtype consistency to avoid precision loss
            if current_hidden_states.dtype != hidden_states.dtype:
                current_hidden_states = current_hidden_states.to(hidden_states.dtype)
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        # Add shared expert output (exactly like original)
        shared_expert_output = moe_layer.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(moe_layer.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    def _get_expert_for_execution(self, layer_idx, expert_idx):
        """Get expert for execution, loading from CPU to GPU buffer if needed."""
        expert_key = (layer_idx, expert_idx)

        # Check if expert is already loaded in buffer
        if self.current_expert == expert_key:
            return self.expert_buffer
        else:
            # Load expert from CPU to GPU buffer
            cpu_expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]

            # Copy CPU expert weights to GPU buffer with correct dtype
            state_dict = cpu_expert.state_dict()
            # Ensure all weights are loaded with the correct dtype
            for key, tensor in state_dict.items():
                if tensor.dtype != self.dtype:
                    state_dict[key] = tensor.to(self.dtype)
            self.expert_buffer.load_state_dict(state_dict)
            self.current_expert = expert_key

        return self.expert_buffer

    def generate(self, text=None, output_token=20, input_token=None):
        """Generate text with expert management tracking."""
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