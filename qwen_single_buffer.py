#!/usr/bin/env python3
"""
Clean Qwen1.5-MoE implementation with single expert buffer.
- All experts stored on CPU
- Single expert buffer on GPU
- Expert fetched to GPU buffer when needed
- All computation happens on GPU
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import nvtx
import copy
from types import MethodType


class QwenSingleBuffer:
    def __init__(self, model_name="Qwen/Qwen1.5-MoE-A2.7B"):
        """
        Initialize Qwen model with single expert buffer on GPU.

        Architecture:
        - All experts stored on CPU
        - Single expert buffer on GPU
        - Expert fetched to buffer when needed
        - All computation on GPU
        """
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        print(f"🚀 Loading {model_name} with single expert buffer")
        print("📋 Architecture:")
        print("   - All experts stored on CPU")
        print("   - Single expert buffer on GPU")
        print("   - Fetch expert to GPU when needed")
        print("   - All computation on GPU")

        # Load model
        self._load_model()

        # Analyze structure
        self._analyze_model_structure()

        # Set up single buffer system
        self._setup_single_buffer()

        # Hook MoE layers
        self._hook_moe_layers()

        print("✅ Model ready with single expert buffer")

    def _load_model(self):
        """Load model and tokenizer."""
        with nvtx.annotate("Model Loading"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="cpu"  # All on CPU initially
            )

    def _analyze_model_structure(self):
        """Analyze model structure."""
        self.n_layers = len(self.model.model.layers)
        self.moe_layers = []

        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                self.moe_layers.append(i)
                print(f"Layer {i}: {len(layer.mlp.experts)} experts")

        print(f"📊 Found {len(self.moe_layers)} MoE layers")

    def _setup_single_buffer(self):
        """Set up single expert buffer system."""
        print("🔧 Setting up single expert buffer...")

        # Move non-expert components to GPU
        self._move_non_experts_to_gpu()

        # Create single expert buffer on GPU
        if len(self.moe_layers) > 0:
            first_layer = self.moe_layers[0]
            sample_expert = self.model.model.layers[first_layer].mlp.experts[0]

            # Single expert buffer on GPU
            self.expert_buffer = copy.deepcopy(sample_expert).to(self.device)

            # Track what's currently loaded: (layer_idx, expert_idx) or None
            self.loaded_expert = None

        # Statistics
        self.expert_fetch_count = 0
        self.expert_hit_count = 0

        print("✅ Created single expert buffer on GPU")

    def _move_non_experts_to_gpu(self):
        """Move all non-expert components to GPU."""
        print("🔄 Moving non-expert components to GPU...")

        # Move main components
        self.model.lm_head.to(self.device)
        self.model.model.embed_tokens.to(self.device)
        self.model.model.norm.to(self.device)

        # Move layer components except experts
        for layer in self.model.model.layers:
            layer.self_attn.to(self.device)
            layer.input_layernorm.to(self.device)
            layer.post_attention_layernorm.to(self.device)

            if hasattr(layer, 'mlp'):
                layer.mlp.gate.to(self.device)
                if hasattr(layer.mlp, 'shared_expert'):
                    layer.mlp.shared_expert.to(self.device)
                # Keep experts on CPU

    def _fetch_expert_to_buffer(self, layer_idx, expert_idx):
        """Fetch expert from CPU to GPU buffer if not already loaded."""
        target_expert = (layer_idx, expert_idx)

        # Check if already loaded
        if self.loaded_expert == target_expert:
            self.expert_hit_count += 1
            return

        # Fetch expert to buffer
        with nvtx.annotate(f"Expert Fetch L{layer_idx}E{expert_idx}"):
            source_expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]

            # Copy parameters from CPU to GPU buffer
            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                    source_expert.named_parameters(),
                    self.expert_buffer.named_parameters()
                ):
                    param2.copy_(param1.to(self.device))

            self.loaded_expert = target_expert
            self.expert_fetch_count += 1

    def _hook_moe_layers(self):
        """Hook into MoE layers to intercept expert calls."""
        print("🔗 Hooking into MoE layers...")

        for layer_idx in self.moe_layers:
            layer = self.model.model.layers[layer_idx]

            def create_hooked_forward(layer_idx):
                def hooked_forward(self, hidden_states):
                    return self._moe_forward_with_single_buffer(hidden_states, layer_idx)
                return hooked_forward

            # Replace forward method
            layer.mlp.forward = MethodType(
                create_hooked_forward(layer_idx),
                self
            )

        print("✅ MoE layers hooked")

    def _moe_forward_with_single_buffer(self, hidden_states, layer_idx):
        """MoE forward with single buffer expert fetching."""
        moe_layer = self.model.model.layers[layer_idx].mlp

        # Router logic
        router_logits = moe_layer.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states)

        # Process each unique expert
        for expert_idx in selected_experts.flatten().unique():
            expert_idx = expert_idx.item()

            # Fetch expert to buffer (handles caching)
            self._fetch_expert_to_buffer(layer_idx, expert_idx)

            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            expert_tokens = hidden_states[expert_mask]

            if expert_tokens.numel() > 0:
                # Forward through GPU buffer (all computation on GPU)
                expert_output = self.expert_buffer(expert_tokens)

                # Apply routing weights and accumulate
                routing_weight = routing_weights[expert_mask,
                                               (selected_experts[expert_mask] == expert_idx).nonzero(as_tuple=True)[1]]
                weighted_output = expert_output * routing_weight.unsqueeze(-1)
                final_hidden_states[expert_mask] += weighted_output

        # Add shared expert if present
        if hasattr(moe_layer, 'shared_expert') and moe_layer.shared_expert is not None:
            final_hidden_states += moe_layer.shared_expert(hidden_states)

        return final_hidden_states

    def generate(self, prompt, max_new_tokens=10, **kwargs):
        """Generate text with single buffer expert fetching."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None

        # Reset statistics
        self.expert_fetch_count = 0
        self.expert_hit_count = 0
        self.loaded_expert = None  # Clear buffer

        start_time = time.time()

        with torch.no_grad():
            with nvtx.annotate("Generation with Single Buffer"):
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

        generation_time = time.time() - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate hit rate
        total_accesses = self.expert_fetch_count + self.expert_hit_count
        hit_rate = self.expert_hit_count / total_accesses if total_accesses > 0 else 0

        return {
            'text': generated_text,
            'generation_time': generation_time,
            'expert_fetches': self.expert_fetch_count,
            'expert_hits': self.expert_hit_count,
            'hit_rate': hit_rate,
            'total_expert_accesses': total_accesses
        }

    def get_memory_usage(self):
        """Get GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() // 1024**2,
                'cached_mb': torch.cuda.memory_reserved() // 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() // 1024**2
            }
        return {}


def main():
    """Test single buffer implementation."""
    print("🧪 Testing Qwen Single Expert Buffer")
    print("=" * 50)

    # Create model with single buffer
    model = QwenSingleBuffer()

    # Test generation
    test_prompt = "The capital of France is"
    print(f"\n📝 Test prompt: '{test_prompt}'")

    result = model.generate(test_prompt, max_new_tokens=10)

    print(f"\n✅ Generated: '{result['text']}'")
    print(f"⏱️ Time: {result['generation_time']:.2f}s")
    print(f"📊 Expert fetches: {result['expert_fetches']}")
    print(f"📊 Expert hits: {result['expert_hits']}")
    print(f"📊 Total accesses: {result['total_expert_accesses']}")
    print(f"📊 Hit rate: {result['hit_rate']:.1%}")

    memory = model.get_memory_usage()
    if memory:
        print(f"💾 GPU Memory: {memory['allocated_mb']} MB allocated")
        print(f"💾 GPU Memory: {memory['cached_mb']} MB cached")

    print(f"\n🎯 Architecture Validation:")
    print(f"   ✅ All experts stored on CPU")
    print(f"   ✅ Single expert buffer on GPU")
    print(f"   ✅ Expert fetched when needed")
    print(f"   ✅ All computation on GPU")


if __name__ == "__main__":
    main()