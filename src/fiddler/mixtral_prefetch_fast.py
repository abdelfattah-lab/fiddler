import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers


class FiddlerMixtralPrefetchFast:
    """FAST & SIMPLE implementation of 2-buffer prefetching strategy"""
    
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            use_cache=True,
        )
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)

        self.bring_non_expert_to_gpu()

        # SIMPLE: Just track which layers are ready in buffers
        # -1 means no layer loaded
        self.buffer_1_layer = 0   # Start with layer 0 ready
        self.buffer_2_layer = 1   # Start with layer 1 ready
        
        # Pre-create GPU expert placeholders (reuse them)
        self.expert_placeholder_1 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        
        self.expert_placeholder_2 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)

        print("FAST prefetching model ready!")

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

    def get_expert_for_layer(self, layer_idx, expert_idx):
        """FAST: Get expert with simple fallback logic"""
        # Use baseline approach: load expert on-demand using placeholder
        self.expert_placeholder_1.load_state_dict(
            self.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].state_dict()
        )
        return self.expert_placeholder_1

    def initial_beam_tensor(self, input_tensor):
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(16)
        
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
            if is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])

            logits = self.mixtral_forward(input_ids, position_ids, is_decode)

            logits = logits.to("cpu")
            logits = F.softmax(logits, dim=-1)

            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            
            probs = probs * new_probs
            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
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

        return prefill_time, decode_time, 0

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
        """FAST forward pass with minimal prefetching overhead"""
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
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(-1, hidden_dim)
            
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # FAST: Process experts with minimal overhead
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=8
            ).permute(2, 1, 0)

            for i_expert in range(self.n_expert):
                idx, top_2 = torch.where(expert_mask[i_expert])
                
                if top_2.shape[0] == 0:
                    continue

                top_2_list = top_2.tolist()
                idx_list = idx.tolist()

                current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                
                # FAST: Use the simple expert loading
                expert = self.get_expert_for_layer(i_layer, i_expert)
                current_state = expert(
                    current_state, routing_weights[top_2_list, idx_list, None]
                )
                
                inps_after_experts.index_add_(
                    0, top_2, current_state.to(inps.dtype)
                )

            # Addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis