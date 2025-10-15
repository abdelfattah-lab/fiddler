#!/usr/bin/env python3
"""Oracle variant of the Qwen MoE model with all experts resident on GPU."""

import copy

from .qwen import FiddlerQwen


class FiddlerQwenOracle(FiddlerQwen):
    """Loads every expert onto the GPU to provide an oracle upper bound."""

    def __init__(self, args):
        super().__init__(args)
        self._preload_all_experts_to_gpu()

    def _preload_all_experts_to_gpu(self) -> None:
        """Clone every expert onto the GPU upfront."""
        self.gpu_expert_cache = {}
        total_loaded = 0

        for layer_idx in self.moe_layers:
            moe_layer = self.model.model.layers[layer_idx].mlp
            for expert_idx, expert in enumerate(moe_layer.experts):
                gpu_expert = copy.deepcopy(expert).to(self.device, dtype=self.dtype)
                self.gpu_expert_cache[(layer_idx, expert_idx)] = gpu_expert
                total_loaded += 1

        print(f"✅ Oracle mode: cached {total_loaded} experts on GPU")

    def _get_expert_for_execution(self, layer_idx: int, expert_idx: int):
        """Return the preloaded GPU expert instead of performing CPU copies."""
        expert_key = (layer_idx, expert_idx)
        gpu_expert = self.gpu_expert_cache.get(expert_key)
        if gpu_expert is None:
            raise KeyError(f"Expert {expert_idx} for layer {layer_idx} not preloaded on GPU")

        # Reuse base-class accounting so hit statistics read as 100%.
        self.current_expert = expert_key
        self.expert_buffer = gpu_expert
        return gpu_expert
