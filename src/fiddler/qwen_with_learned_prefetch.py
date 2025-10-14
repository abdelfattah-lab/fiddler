#!/usr/bin/env python3
"""
FiddlerQwenWithLearnedPrefetch - Learned expert prefetching
Uses trained attention-based predictor instead of token position patterns
"""

import torch
import os
import sys

# Import the base prefetch implementation
try:
    from .qwen_with_prefetch import FiddlerQwenWithPrefetch, PrefetchMetrics
except ImportError:
    # If running as script, use absolute import
    from qwen_with_prefetch import FiddlerQwenWithPrefetch, PrefetchMetrics

# Import predictor model from project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from train_predictor import AttentionBasedExpertPredictor


class FiddlerQwenWithLearnedPrefetch(FiddlerQwenWithPrefetch):
    """
    Qwen implementation with learned expert prefetching.
    Uses attention-based predictor instead of token position patterns.
    """

    def __init__(
        self,
        args,
        predictor_path="predictor_checkpoints/best_model.pt",
        num_experts_to_prefetch=8,
        enable_cpu_offload=False,
        latency_cpu=0.1,
        latency_gpu=10.0,
        n_gpu_resident_experts=0,
        prefill_aggregation='frequency'  # 'frequency', 'mean', or 'max'
    ):
        """
        Initialize model with learned prefetching.

        Args:
            args: Model arguments (model path, etc.)
            predictor_path: Path to trained predictor checkpoint
            num_experts_to_prefetch: Number of experts to prefetch per layer
            enable_cpu_offload: Whether to use Fiddler CPU offloading
            latency_cpu: CPU expert latency for Fiddler cost model
            latency_gpu: GPU expert latency for Fiddler cost model
            n_gpu_resident_experts: Number of experts to keep on GPU permanently
        """
        print("="*80)
        print("INITIALIZING LEARNED PREFETCH MODEL")
        print("="*80)

        # Store predictor path and set device before loading
        self.predictor_path = predictor_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Load predictor first to ensure it's available
        print(f"📖 Loading predictor from: {predictor_path}")
        self._load_predictor(predictor_path)

        # Initialize parent class (FiddlerQwenWithPrefetch)
        # This will set collection_mode=False since we're providing predictor
        super().__init__(
            args,
            num_experts_to_prefetch=num_experts_to_prefetch,
            enable_cpu_offload=enable_cpu_offload,
            latency_cpu=latency_cpu,
            latency_gpu=latency_gpu,
            n_gpu_resident_experts=n_gpu_resident_experts
        )

        # Override collection mode - we use learned predictor, not patterns
        # Force prediction mode since we have a learned predictor
        self.collection_mode = False
        self.profiler.collection_mode = False
        self.expert_patterns = {}  # Empty patterns - we use predictor instead

        # Register attention capture hook on layer 0
        print("🔗 Registering attention capture hook on layer 0...")
        self._register_attention_hook()

        # Storage for current attention output
        self.current_attention_output = None

        # Cache for predictor outputs (to avoid running predictor multiple times per forward pass)
        # Maps layer_idx -> list of expert indices
        self.cached_predictions = {}

        # Create separate CUDA stream for asynchronous predictor execution
        self.predictor_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.predictor_ready_event = None  # Event to track when predictor completes

        # Aggregation strategy for prefill phase
        self.prefill_aggregation = prefill_aggregation

        print("✅ Learned prefetch model initialized")
        print(f"🔮 Using learned predictor with {num_experts_to_prefetch} experts per layer")
        print(f"🔮 Prefill aggregation strategy: {prefill_aggregation}")
        if self.predictor_stream:
            print(f"⚡ Async predictor stream initialized (non-blocking execution)")
        print("="*80)

    def _load_predictor(self, predictor_path):
        """Load trained predictor model."""
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"Predictor checkpoint not found: {predictor_path}\n"
                f"Please run Phase 2 (train_predictor.py) first."
            )

        checkpoint = torch.load(predictor_path, map_location=self.device)
        config = checkpoint['config']

        self.predictor = AttentionBasedExpertPredictor(
            hidden_dim=config['hidden_dim'],
            n_moe_layers=config['n_moe_layers'],
            n_experts=config['n_experts'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)

        self.predictor.load_state_dict(checkpoint['model_state_dict'])
        self.predictor.eval()

        # Convert predictor to the same dtype as the model
        self.predictor = self.predictor.to(dtype=self.dtype)

        print(f"✅ Loaded predictor from epoch {checkpoint['epoch']}")
        print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
        print(f"   Validation Top-4 Accuracy: {checkpoint['val_acc']*100:.2f}%")

    def _register_attention_hook(self):
        """Register hook to capture first layer attention output and run predictor asynchronously."""
        def attention_hook(module, input, output):
            # Capture attention output for predictor
            # output is a tuple: (hidden_states, attention_weights, ...)
            # We want hidden_states which is output[0]
            self.current_attention_output = output[0].detach()  # [batch, seq_len, hidden_dim]

            # EAGER ASYNC EXECUTION: Run predictor immediately in separate stream
            # This allows predictions to compute in parallel with layer processing
            # The main thread is NOT blocked - predictor runs asynchronously
            self._run_predictor_and_cache_async()

        # Register on layer 0 self-attention
        self.model.model.layers[0].self_attn.register_forward_hook(attention_hook)
        print("✅ Attention hook registered on layer 0 (eager async predictor execution)")

    def _run_predictor_and_cache_async(self):
        """
        Run predictor asynchronously in separate CUDA stream.
        This is called eagerly in the attention hook and does NOT block the main thread.
        """
        # Check if we have attention output from layer 0
        if self.current_attention_output is None:
            # No attention output yet - cannot make predictions
            return

        # If we don't have a separate stream, fall back to synchronous execution
        if self.predictor_stream is None:
            self._run_predictor_and_cache()
            return

        # Run predictor in separate stream (non-blocking)
        with torch.cuda.stream(self.predictor_stream):
            self._run_predictor_and_cache()

            # Record event to track completion
            self.predictor_ready_event = torch.cuda.Event()
            self.predictor_ready_event.record(self.predictor_stream)

    def _run_predictor_and_cache(self):
        """
        Run predictor once and cache predictions for all layers.
        Called either synchronously (if no stream) or asynchronously (in predictor_stream).
        """
        # Check if we have attention output from layer 0
        if self.current_attention_output is None:
            # No attention output yet - cannot make predictions
            return

        # Run predictor once for all layers
        with torch.no_grad():
            attention_output = self.current_attention_output  # [batch, seq_len, 2048]
            predicted_logits = self.predictor(attention_output)  # [seq_len, 22, 60] or [batch, seq_len, 22, 60]

        # Handle both 3D and 4D output from predictor
        if len(predicted_logits.shape) == 3:
            # Output is [seq_len, 22, 60] - batch dimension was removed
            # Add batch dimension for uniform handling
            predicted_logits = predicted_logits.unsqueeze(0)  # [1, seq_len, 22, 60]

        # Now predicted_logits is always [batch, seq_len, 22, 60]
        batch_size, seq_len, n_layers, n_experts = predicted_logits.shape

        # Determine if we're in decode or prefill phase
        is_decode_phase = (seq_len == 1)

        # Process predictions for each layer and cache them
        for output_idx in range(n_layers):
            layer_logits = predicted_logits[:, :, output_idx, :]  # [batch, seq_len, 60]

            # Determine aggregation strategy based on phase
            if is_decode_phase:
                # Decode phase: aggregate across batch elements using frequency counting
                predicted_experts = self._aggregate_batch_predictions(
                    layer_logits.squeeze(1),  # [batch, 60]
                    self.num_experts_to_prefetch
                )
            else:
                # Prefill phase: aggregate across both batch and sequence dimensions
                predicted_experts = self._aggregate_prefill_predictions(
                    layer_logits,  # [batch, seq_len, 60]
                    self.num_experts_to_prefetch
                )

            # Map output_idx to actual layer_idx
            # output_idx=0 -> layer_idx=moe_layers[2]
            # output_idx=1 -> layer_idx=moe_layers[3], etc.
            layer_idx = self.moe_layers[output_idx + 2]
            self.cached_predictions[layer_idx] = predicted_experts

    def _predict_experts_for_layer(self, layer_idx, token_pos):
        """
        Predict which experts will be needed for this layer using learned predictor.
        Returns cached predictions that were eagerly computed in the attention hook.
        Synchronizes with async predictor if needed.

        Args:
            layer_idx: MoE layer index (2-23 for Qwen)
            token_pos: Token position (for decode phase)

        Returns:
            List of expert indices to prefetch
        """
        # Check if this layer is one we predict for (layers 2-23)
        if layer_idx not in self.moe_layers[2:]:
            return []

        # CRITICAL: Synchronize with predictor stream if predictions are still computing
        # This ensures we wait for the async predictor to finish before using results
        if self.predictor_ready_event is not None:
            self.predictor_ready_event.synchronize()
            # Clear event after synchronization (only wait once per forward pass)
            self.predictor_ready_event = None

        # Return cached prediction (computed eagerly in attention hook)
        # Predictions are now guaranteed to be available since we synchronized
        return self.cached_predictions.get(layer_idx, [])

    def _aggregate_batch_predictions(self, layer_logits, k):
        """
        Aggregate expert predictions across batch elements during decode phase.

        For each batch element, get the top-k experts, then count how many times
        each expert appears across all batch elements, and select the k most
        frequent experts.

        Args:
            layer_logits: [batch, 60] - predicted logits for each batch element
            k: number of experts to select

        Returns:
            List of k expert indices to prefetch
        """
        batch_size, n_experts = layer_logits.shape

        # For each batch element, get top-4 experts (matching Qwen's gating top_k=4)
        top_k_per_element = 4  # Match Qwen's gating top-k
        _, top_experts_per_element = torch.topk(layer_logits, k=top_k_per_element, dim=1)  # [batch, 4]

        # Count frequency of each expert across all batch elements
        expert_counts = torch.zeros(n_experts, dtype=torch.long, device=layer_logits.device)
        for batch_experts in top_experts_per_element:
            for expert_idx in batch_experts:
                expert_counts[expert_idx] += 1

        # Select top-k most frequent experts
        _, top_k_by_frequency = torch.topk(expert_counts, k=min(k, n_experts))
        return top_k_by_frequency.cpu().tolist()

    def _aggregate_prefill_predictions(self, layer_logits, k):
        """
        Aggregate expert predictions across batch and sequence dimensions during prefill.

        Args:
            layer_logits: [batch, seq_len, 60] - predicted logits for each batch and token
            k: number of experts to select

        Returns:
            List of k expert indices to prefetch
        """
        batch_size, seq_len, n_experts = layer_logits.shape

        if self.prefill_aggregation == 'frequency':
            # Strategy 1: Frequency-based (most principled)
            # For each token in each batch, get top-4 experts (matching Qwen's top_k=4)
            # Then select the k most frequent experts across all batch elements and tokens
            top_k_per_token = 4  # Match Qwen's gating top-k
            # Reshape to [batch*seq_len, n_experts] for easier processing
            layer_logits_flat = layer_logits.reshape(-1, n_experts)  # [batch*seq_len, 60]
            _, top_experts_per_token = torch.topk(layer_logits_flat, k=top_k_per_token, dim=1)  # [batch*seq_len, 4]

            # Count frequency of each expert across all batch elements and tokens
            expert_counts = torch.zeros(n_experts, dtype=torch.long, device=layer_logits.device)
            for token_experts in top_experts_per_token:
                for expert_idx in token_experts:
                    expert_counts[expert_idx] += 1

            # Select top-k most frequent experts
            _, top_k_by_frequency = torch.topk(expert_counts, k=min(k, n_experts))
            return top_k_by_frequency.cpu().tolist()

        elif self.prefill_aggregation == 'mean':
            # Strategy 2: Mean pooling (average scores across batch and tokens)
            mean_logits = layer_logits.mean(dim=(0, 1))  # [60]
            _, top_k_indices = torch.topk(mean_logits, k=min(k, n_experts))
            return top_k_indices.cpu().tolist()

        elif self.prefill_aggregation == 'max':
            # Strategy 3: Max pooling (experts needed by ANY batch element or token)
            # First max over seq_len, then max over batch
            max_logits = layer_logits.max(dim=1)[0]  # [batch, 60]
            max_logits = max_logits.max(dim=0)[0]  # [60]
            _, top_k_indices = torch.topk(max_logits, k=min(k, n_experts))
            return top_k_indices.cpu().tolist()

        else:
            # Fallback to mean pooling
            mean_logits = layer_logits.mean(dim=(0, 1))  # [60]
            _, top_k_indices = torch.topk(mean_logits, k=min(k, n_experts))
            return top_k_indices.cpu().tolist()

    def generate(self, text=None, output_token=20, input_token=None):
        """Generate with learned prefetching and metrics tracking."""
        # Reset attention output and cached predictions at start of generation
        self.current_attention_output = None
        self.cached_predictions = {}
        self.predictor_ready_event = None  # Reset event for new generation

        # Call parent generate which handles all the generation logic
        return super().generate(text=text, output_token=output_token, input_token=input_token)

    def load_expert_patterns(self):
        """
        Override parent method to skip loading patterns file.
        We use learned predictor instead of patterns.
        """
        # Return empty dict to indicate we have "patterns" (actually predictor)
        # This prevents collection_mode from being set to True
        return {}

    def save_expert_patterns(self):
        """
        Override parent method to skip saving patterns.
        We use learned predictor instead of patterns.
        """
        # Don't save patterns - we use learned predictor
        pass
