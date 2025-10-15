#!/bin/bash
# Run benchmark with each config in isolated subprocess to avoid OOM

set -e

OUTPUT_DIR="expert_loading_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run each configuration in separate process
echo "=== Running Baseline ==="
python3 run_single_config.py --config baseline --tokens 20 --trials 1 --output "$OUTPUT_DIR/baseline_results.json"
echo ""

echo "=== Running Oracle ==="
python3 run_single_config.py --config oracle --tokens 20 --trials 1 --output "$OUTPUT_DIR/oracle_results.json"
echo ""

for COUNT in 2 4 8 16; do
    echo "=== Running Prefetch-$COUNT ==="
    python3 run_single_config.py --config prefetch --prefetch-count $COUNT --tokens 20 --trials 1 --output "$OUTPUT_DIR/prefetch_${COUNT}_results.json"
    echo ""
done

# Combine results and create plot
echo "=== Creating plot ==="
python3 - "$OUTPUT_DIR" << 'PYEOF'
import json
import os
import sys
from glob import glob
import matplotlib.pyplot as plt

output_dir = sys.argv[1]

# Load all results
all_results = []
for json_file in sorted(glob(f"{output_dir}/*_results.json")):
    with open(json_file) as f:
        all_results.extend(json.load(f))

# Plot
style_map = {
    "Baseline (On-Demand)": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "Oracle (All Experts)": {"color": "#2ca02c", "linestyle": "-", "marker": "s"},
}

# Add prefetch styles
prefetch_configs = [r for r in all_results if "Prefetch" in r["config"]]
unique_prefetch = sorted(set(r["config"] for r in prefetch_configs))
for idx, config in enumerate(unique_prefetch):
    style_map[config] = {
        "color": plt.cm.tab10((idx + 2) % 10),
        "linestyle": "--",
        "marker": "o",
    }

plt.figure(figsize=(10, 6))
for result in all_results:
    config = result["config"]
    style = style_map.get(config, {})
    plt.scatter(
        [result["tokens"]],
        [result["total_time_mean"]],
        label=config,
        color=style.get("color"),
        marker=style.get("marker", "o"),
        s=100
    )

plt.xlabel("Output Tokens")
plt.ylabel("Total Time (s)")
plt.title("Qwen Expert Loading Strategies")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = f"{output_dir}/expert_loading_strategies.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"📊 Plot saved to: {plot_path}")

# Save combined JSON
with open(f"{output_dir}/expert_loading_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"🗂️  JSON saved to: {output_dir}/expert_loading_results.json")
print(f"\nResults in: {output_dir}")
PYEOF
