import itertools
import time
import sys
import os

sys.path.append("../src")
from fiddler.mixtral import FiddlerMixtral

# Fixed values
PROMPT = "The answer to life is "
N_TOKENS = 10
MODEL_PATH = "mistralai/Mixtral-8x7B-v0.1"
BEAM_WIDTH = 1
NO_PRELOADING = True

# All combinations of options
# options = list(itertools.product([True, False], repeat=3))
options = [
# (efficient_copy, enable_prefetch, run_all_experts_on_gpu, cpu_offload)
    (False,         False,          True,                   True), 
    (False,         False,          True,                   False), 
    (False,         True,           True,                   False), 
    (True,          True,           True,                   False), 
]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

results = []

for efficient_copy, enable_prefetch, run_all_experts_on_gpu, cpu_offload in options:
    print("\n==============================")
    print(f"efficient_copy={efficient_copy}, enable_prefetch={enable_prefetch}, run_all_experts_on_gpu={run_all_experts_on_gpu}")
    class Args:
        model = MODEL_PATH
        cpu_offload = cpu_offload
        beam_width = BEAM_WIDTH
        enable_prefetch = enable_prefetch
        efficient_copy = efficient_copy
        run_all_experts_on_gpu = run_all_experts_on_gpu
        no_preloading = NO_PRELOADING
    args = Args()
    model = FiddlerMixtral(args)
    prefill_time, decode_time, hit_rate = model.generate(PROMPT, output_token=N_TOKENS)
    print(f"Decode time: {decode_time:.3f}s")
    results.append({
        'efficient_copy': efficient_copy,
        'enable_prefetch': enable_prefetch,
        'run_all_experts_on_gpu': run_all_experts_on_gpu,
        'decode_time': decode_time,
    })

# Print summary table
print("\n===== SUMMARY =====")
print("efficient_copy | enable_prefetch | run_all_experts_on_gpu | decode (s)")
for r in results:
    print(f"{r['efficient_copy']:>13} | {r['enable_prefetch']:>15} | {r['run_all_experts_on_gpu']:>17} | {r['decode_time']:.3f}")

# Compute speedup vs slowest config
slowest = max(r['decode_time'] for r in results)
print("\nSpeedup vs slowest config (Baseline):")
for r in results:
    speedup = slowest / r['decode_time']
    print(f"Config: efficient_copy={r['efficient_copy']}, enable_prefetch={r['enable_prefetch']}, run_all_experts_on_gpu={r['run_all_experts_on_gpu']} | Speedup: {speedup:.2f}x") 