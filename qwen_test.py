#!/usr/bin/env python3
"""Simple test script for Qwen1.5-MoE-A2.7B profiling."""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import nvtx

def main():
    print("🚀 Loading Qwen/Qwen1.5-MoE-A2.7B...")

    # Load model and tokenizer
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"

    with nvtx.annotate("Model Loading"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            pad_token="<|endoftext|>"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # Use 8-bit to reduce memory usage
        )

    # Test prompt
    test_prompt = "The capital of France is"
    print(f"📝 Test prompt: '{test_prompt}'")

    # Tokenize input
    with nvtx.annotate("Tokenization"):
        inputs = tokenizer(test_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

    print(f"🔢 Input tokens: {input_ids.shape[1]}")

    # Generation with NVTX markers for profiling
    with torch.no_grad():
        with nvtx.annotate("Generation"):
            # Generate 10 tokens to capture both prefill and decode phases
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

    # Decode output
    with nvtx.annotate("Output Decoding"):
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"✅ Generated: '{generated_text}'")
    print("🏁 Profiling completed")

if __name__ == "__main__":
    main()
