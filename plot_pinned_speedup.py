#!/usr/bin/env python3
"""
Plot comparison showing speedup from pinned memory in baseline.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
baseline_without_pinned = 2.908  # From guide.md
baseline_with_pinned = 2.182     # From our benchmark
speedup = baseline_without_pinned / baseline_with_pinned

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Time comparison
configs = ['Baseline\n(no pinned)', 'Baseline\n(with pinned)']
times = [baseline_without_pinned, baseline_with_pinned]
colors = ['#e74c3c', '#3498db']

bars1 = ax1.bar(configs, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Baseline Performance: Pinned vs Non-Pinned Memory', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, time in zip(bars1, times):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.3f}s',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 2: Speedup
configs_speedup = ['Speedup from\nPinned Memory']
speedup_values = [speedup]
colors_speedup = ['#27ae60']

bars2 = ax2.bar(configs_speedup, speedup_values, color=colors_speedup, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax2.set_title('Speedup from Pinned Memory', fontsize=14, fontweight='bold')
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline (1.0x)')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend()

# Add value labels on bars
for bar, spd in zip(bars2, speedup_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{spd:.2f}x',
             ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkgreen')

# Add summary text
summary_text = f"""
Baseline Comparison:
• Without pinned: {baseline_without_pinned:.3f}s
• With pinned:    {baseline_with_pinned:.3f}s
• Speedup:        {speedup:.2f}x ({((speedup - 1) * 100):.1f}% faster)
"""

fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         family='monospace')

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig('pinned_baseline_benchmark_20250930_185406/pinned_memory_speedup.png', dpi=300, bbox_inches='tight')
print("Plot saved to: pinned_baseline_benchmark_20250930_185406/pinned_memory_speedup.png")
print(f"\nSpeedup from pinned memory: {speedup:.2f}x")
print(f"Time reduction: {baseline_without_pinned - baseline_with_pinned:.3f}s ({((speedup - 1) * 100):.1f}% faster)")
