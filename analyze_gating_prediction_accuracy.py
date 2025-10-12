#!/usr/bin/env python3
"""
Analyze the accuracy of using layer X's gating to predict layer X+1 and X+2 expert selections.
This helps understand how well expert usage patterns propagate across layers.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import os


class GatingAnalyzer:
    """Analyzes gating patterns and prediction accuracy across MoE layers."""

    def __init__(self, model_name="Qwen/Qwen1.5-MoE-A2.7B"):
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Find MoE layers
        self.moe_layers = []
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts'):
                self.moe_layers.append(idx)

        print(f"Found {len(self.moe_layers)} MoE layers: {self.moe_layers}")

        # Storage for gating decisions
        self.gating_decisions = {}  # token_pos -> layer_idx -> selected_experts (shape: [batch*seq, top_k])
        self.current_token_pos = 0

        # Register hooks to capture gating decisions
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture gating decisions."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Extract gating information
                hidden_states = input[0]
                batch_size, sequence_length, hidden_dim = hidden_states.shape
                hidden_states_flat = hidden_states.view(-1, hidden_dim)

                # Get router logits
                router_logits = module.gate(hidden_states_flat)

                # Get top-k experts
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                _, selected_experts = torch.topk(routing_weights, module.top_k, dim=-1)

                # Store decisions for each token in the sequence
                # For prefill: multiple tokens, for decode: single token
                for seq_idx in range(batch_size * sequence_length):
                    token_key = self.current_token_pos + seq_idx
                    if token_key not in self.gating_decisions:
                        self.gating_decisions[token_key] = {}

                    # Store the selected experts for this token (as a list)
                    self.gating_decisions[token_key][layer_idx] = selected_experts[seq_idx].cpu().tolist()

                # Update token position if this is the last MoE layer
                if layer_idx == self.moe_layers[-1]:
                    self.current_token_pos += batch_size * sequence_length

                return output

            return hook

        # Register hooks for all MoE layers
        for layer_idx in self.moe_layers:
            layer = self.model.model.layers[layer_idx].mlp
            layer.register_forward_hook(make_hook(layer_idx))

    def generate_and_collect(self, text="The capital of France is", num_tokens=20):
        """Generate tokens and collect gating decisions."""
        print(f"\nGenerating {num_tokens} tokens from prompt: '{text}'")

        # Reset collection
        self.gating_decisions = {}
        self.current_token_pos = 0

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                eos_token_id=None,  # Force exact token count
                use_cache=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        print(f"Collected gating decisions for {len(self.gating_decisions)} token positions")

        return generated_text

    def calculate_prediction_accuracy(self, lookahead=1):
        """
        Calculate how well layer X's gating predicts layer X+lookahead's gating.

        Args:
            lookahead: How many layers ahead to predict (1 for X->X+1, 2 for X->X+2)

        Returns:
            Dictionary with accuracy metrics per layer
        """
        print(f"\nCalculating prediction accuracy for lookahead={lookahead} (X -> X+{lookahead})")

        accuracies = {}

        for i, source_layer in enumerate(self.moe_layers):
            # Check if target layer exists
            if i + lookahead >= len(self.moe_layers):
                continue

            target_layer = self.moe_layers[i + lookahead]

            # Calculate metrics across all token positions
            exact_matches = 0
            intersection_counts = []
            jaccard_scores = []
            total_positions = 0

            for token_pos in self.gating_decisions:
                if source_layer not in self.gating_decisions[token_pos]:
                    continue
                if target_layer not in self.gating_decisions[token_pos]:
                    continue

                source_experts = set(self.gating_decisions[token_pos][source_layer])
                target_experts = set(self.gating_decisions[token_pos][target_layer])

                # Exact match: all experts are the same
                if source_experts == target_experts:
                    exact_matches += 1

                # Intersection: how many experts overlap
                intersection = len(source_experts & target_experts)
                intersection_counts.append(intersection)

                # Jaccard similarity: intersection / union
                union = len(source_experts | target_experts)
                jaccard = intersection / union if union > 0 else 0
                jaccard_scores.append(jaccard)

                total_positions += 1

            if total_positions > 0:
                accuracies[source_layer] = {
                    'target_layer': target_layer,
                    'exact_match_rate': exact_matches / total_positions,
                    'avg_intersection': np.mean(intersection_counts),
                    'avg_jaccard': np.mean(jaccard_scores),
                    'total_positions': total_positions
                }

        return accuracies

    def visualize_prediction_accuracy(self, accuracies_x1, accuracies_x2, output_dir="gating_analysis"):
        """Create comprehensive visualizations for both X+1 and X+2 predictions."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create figure with 2 rows x 3 columns for both prediction scenarios
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gating Prediction Accuracy Analysis for Qwen1.5-MoE-A2.7B',
                     fontsize=16, fontweight='bold')

        # Helper function to plot metrics
        def plot_metrics(accuracies, row, title_prefix):
            if not accuracies:
                for col in range(3):
                    axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[row, col].set_title(f'{title_prefix}: N/A')
                return

            source_layers = sorted(accuracies.keys())

            # Extract metrics
            exact_match_rates = [accuracies[l]['exact_match_rate'] for l in source_layers]
            avg_intersections = [accuracies[l]['avg_intersection'] for l in source_layers]
            avg_jaccards = [accuracies[l]['avg_jaccard'] for l in source_layers]

            # Plot 1: Exact Match Rate
            axes[row, 0].plot(source_layers, exact_match_rates, marker='o', linewidth=2, markersize=8)
            axes[row, 0].set_xlabel('Source Layer (X)', fontsize=11)
            axes[row, 0].set_ylabel('Exact Match Rate', fontsize=11)
            axes[row, 0].set_title(f'{title_prefix}: Exact Match Rate', fontsize=12, fontweight='bold')
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 0].set_ylim([0, 1])

            # Plot 2: Average Intersection (out of top-k=4)
            axes[row, 1].plot(source_layers, avg_intersections, marker='s', linewidth=2,
                            markersize=8, color='orange')
            axes[row, 1].set_xlabel('Source Layer (X)', fontsize=11)
            axes[row, 1].set_ylabel('Average Intersection Count', fontsize=11)
            axes[row, 1].set_title(f'{title_prefix}: Avg. Expert Overlap (out of 4)', fontsize=12, fontweight='bold')
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].set_ylim([0, 4])
            axes[row, 1].axhline(y=4, color='g', linestyle='--', alpha=0.5, label='Perfect (4/4)')
            axes[row, 1].axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Random (~2/4)')
            axes[row, 1].legend(fontsize=9)

            # Plot 3: Jaccard Similarity
            axes[row, 2].plot(source_layers, avg_jaccards, marker='^', linewidth=2,
                            markersize=8, color='green')
            axes[row, 2].set_xlabel('Source Layer (X)', fontsize=11)
            axes[row, 2].set_ylabel('Jaccard Similarity', fontsize=11)
            axes[row, 2].set_title(f'{title_prefix}: Avg. Jaccard Similarity', fontsize=12, fontweight='bold')
            axes[row, 2].grid(True, alpha=0.3)
            axes[row, 2].set_ylim([0, 1])

        # Plot X -> X+1 predictions (row 0)
        plot_metrics(accuracies_x1, 0, 'Layer X → X+1')

        # Plot X -> X+2 predictions (row 1)
        plot_metrics(accuracies_x2, 1, 'Layer X → X+2')

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(output_dir, f'gating_prediction_accuracy_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")

        return output_path

    def save_results(self, accuracies_x1, accuracies_x2, output_dir="gating_analysis"):
        """Save detailed results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            'timestamp': timestamp,
            'model': 'Qwen/Qwen1.5-MoE-A2.7B',
            'moe_layers': self.moe_layers,
            'total_tokens_analyzed': len(self.gating_decisions),
            'x_to_x_plus_1': {
                str(k): v for k, v in accuracies_x1.items()
            },
            'x_to_x_plus_2': {
                str(k): v for k, v in accuracies_x2.items()
            }
        }

        output_path = os.path.join(output_dir, f'gating_prediction_results_{timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")
        return output_path

    def print_summary(self, accuracies_x1, accuracies_x2):
        """Print a summary of the results."""
        print("\n" + "="*80)
        print("GATING PREDICTION ACCURACY SUMMARY")
        print("="*80)

        print("\n📊 Layer X → X+1 Prediction Accuracy:")
        print("-" * 80)
        if accuracies_x1:
            for source_layer in sorted(accuracies_x1.keys()):
                acc = accuracies_x1[source_layer]
                target_layer = acc['target_layer']
                print(f"Layer {source_layer:2d} → {target_layer:2d}: "
                      f"Exact Match: {acc['exact_match_rate']:6.2%} | "
                      f"Avg Overlap: {acc['avg_intersection']:.2f}/4 | "
                      f"Jaccard: {acc['avg_jaccard']:.3f}")

            # Calculate overall statistics
            exact_matches = [acc['exact_match_rate'] for acc in accuracies_x1.values()]
            intersections = [acc['avg_intersection'] for acc in accuracies_x1.values()]
            jaccards = [acc['avg_jaccard'] for acc in accuracies_x1.values()]

            print(f"\n  Overall Averages (X→X+1):")
            print(f"    - Exact Match Rate: {np.mean(exact_matches):.2%}")
            print(f"    - Avg Intersection: {np.mean(intersections):.2f}/4")
            print(f"    - Avg Jaccard: {np.mean(jaccards):.3f}")
        else:
            print("  No data available")

        print("\n📊 Layer X → X+2 Prediction Accuracy:")
        print("-" * 80)
        if accuracies_x2:
            for source_layer in sorted(accuracies_x2.keys()):
                acc = accuracies_x2[source_layer]
                target_layer = acc['target_layer']
                print(f"Layer {source_layer:2d} → {target_layer:2d}: "
                      f"Exact Match: {acc['exact_match_rate']:6.2%} | "
                      f"Avg Overlap: {acc['avg_intersection']:.2f}/4 | "
                      f"Jaccard: {acc['avg_jaccard']:.3f}")

            # Calculate overall statistics
            exact_matches = [acc['exact_match_rate'] for acc in accuracies_x2.values()]
            intersections = [acc['avg_intersection'] for acc in accuracies_x2.values()]
            jaccards = [acc['avg_jaccard'] for acc in accuracies_x2.values()]

            print(f"\n  Overall Averages (X→X+2):")
            print(f"    - Exact Match Rate: {np.mean(exact_matches):.2%}")
            print(f"    - Avg Intersection: {np.mean(intersections):.2f}/4")
            print(f"    - Avg Jaccard: {np.mean(jaccards):.3f}")
        else:
            print("  No data available")

        print("\n" + "="*80)


def main():
    """Main analysis workflow."""
    print("="*80)
    print("GATING PREDICTION ACCURACY ANALYSIS")
    print("="*80)
    print("\nThis script analyzes how well layer X's gating can predict")
    print("the expert selections at layers X+1 and X+2 for Qwen1.5-MoE-A2.7B")
    print("="*80)

    # Initialize analyzer
    analyzer = GatingAnalyzer()

    # Generate and collect gating decisions
    # Use a longer generation to get more data points
    analyzer.generate_and_collect(
        text="The capital of France is Paris, and the capital of Germany is",
        num_tokens=50
    )

    # Calculate prediction accuracy for X -> X+1
    accuracies_x1 = analyzer.calculate_prediction_accuracy(lookahead=1)

    # Calculate prediction accuracy for X -> X+2
    accuracies_x2 = analyzer.calculate_prediction_accuracy(lookahead=2)

    # Print summary
    analyzer.print_summary(accuracies_x1, accuracies_x2)

    # Create visualizations
    output_dir = "gating_analysis"
    viz_path = analyzer.visualize_prediction_accuracy(accuracies_x1, accuracies_x2, output_dir)

    # Save detailed results
    results_path = analyzer.save_results(accuracies_x1, accuracies_x2, output_dir)

    print("\n✅ Analysis complete!")
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   - Visualization: {viz_path}")
    print(f"   - Detailed results: {results_path}")


if __name__ == "__main__":
    main()
