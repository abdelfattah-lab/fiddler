#!/usr/bin/env python3
"""
Profile Qwen/Qwen1.5-MoE-A2.7B using Nsight Systems to analyze
compute vs memory transfer timing for active experts.

This script reuses the existing profile_nsight.py infrastructure.
"""

import sys
import os
import time
from datetime import datetime

# Import the generic profiling function
from profile_nsight import profile_program

def create_qwen_test_script():
    """Create a simple test script for Qwen1.5-MoE-A2.7B."""
    script_content = '''#!/usr/bin/env python3
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
'''

    with open("qwen_test.py", "w") as f:
        f.write(script_content)

    print("📝 Created qwen_test.py")

def profile_qwen_moe():
    """Profile Qwen1.5-MoE-A2.7B focusing on compute vs memory timing."""
    print("🚀 Qwen1.5-MoE-A2.7B Profiling")
    print("=" * 50)

    # Create test script
    create_qwen_test_script()

    # Create profile directory
    profile_dir = f"qwen_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(profile_dir, exist_ok=True)
    print(f"📁 Profile directory: {profile_dir}")

    # Profile the Qwen model using the conda environment
    command_args = ["bash", "-c", "source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen_profiling && python qwen_test.py"]
    scenario_name = "Qwen1.5-MoE-A2.7B"
    output_name = "qwen_moe_profile"

    print(f"🔍 Starting profiling...")
    print(f"📊 Focus: Compute vs Memory transfer timing for expert loading")

    success, profile_path = profile_program(
        command_args=command_args,
        scenario_name=scenario_name,
        output_name=output_name,
        profile_dir=profile_dir,
        duration=120,  # 2 minutes should be enough for this test
        timeout=180    # 3 minutes timeout
    )

    if success:
        print(f"\n✅ Qwen1.5-MoE profiling completed successfully!")
        print(f"📊 Profile saved to: {profile_path}")

        print(f"\n🔍 ANALYSIS COMMANDS:")
        print(f"# Memory operations analysis:")
        print(f"nsys stats --report cuda_gpu_mem_time_sum {profile_path}")
        print(f"nsys stats --report cuda_gpu_mem_size_sum {profile_path}")

        print(f"\n# CUDA kernel analysis:")
        print(f"nsys stats --report cuda_gpu_kern_sum {profile_path}")
        print(f"nsys stats --report cuda_api_sum {profile_path}")

        print(f"\n# Expert-specific analysis (look for MoE patterns):")
        print(f"nsys stats --report nvtx_sum {profile_path}")

        print(f"\n# GUI analysis:")
        print(f"nsight-sys {profile_path}")

        print(f"\n📈 KEY METRICS TO ANALYZE:")
        print(f"1. Memory Transfer Time: Look for cuda_gpu_mem_time_sum")
        print(f"2. Compute Time: Look for kernel execution times")
        print(f"3. Expert Loading Patterns: Check NVTX markers and memory transfers")
        print(f"4. Total Execution: Compare compute vs memory overhead")

    else:
        print(f"\n❌ Qwen1.5-MoE profiling failed")
        print(f"Check that the model can be loaded and CUDA is available")

    return success, profile_path if success else None

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("🚀 Qwen1.5-MoE-A2.7B Profiling Script")
        print("=" * 45)
        print("This script profiles Qwen/Qwen1.5-MoE-A2.7B using Nsight Systems")
        print("to analyze compute vs memory transfer timing for active experts.")
        print()
        print("Usage:")
        print("  python profile_qwen_moe.py")
        print()
        print("Output:")
        print("  - Creates qwen_test.py (test script)")
        print("  - Generates .nsys-rep profile file")
        print("  - Provides analysis commands")
        return

    success, profile_path = profile_qwen_moe()

    if success:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Analyze the profile using the commands above")
        print(f"2. Compare compute time vs memory transfer time")
        print(f"3. Look for expert loading patterns in memory operations")
        print(f"4. Identify optimization opportunities")
    else:
        print(f"\n🚨 TROUBLESHOOTING:")
        print(f"1. Ensure CUDA is available: nvidia-smi")
        print(f"2. Check model access: huggingface-cli login")
        print(f"3. Verify dependencies: pip install transformers torch nvtx")

if __name__ == "__main__":
    main()