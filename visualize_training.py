#!/usr/bin/env python3
"""Visualize training curves from predictor training."""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_training_history():
    """Load and visualize training history."""
    history_path = Path("predictor_checkpoints/training_history.json")

    if not history_path.exists():
        print(f"❌ Training history not found at {history_path}")
        print("   Training may not be complete yet.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    if len(history) == 0:
        print("❌ Training history is empty")
        return

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('KL Divergence Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, train_acc, 'b-o', label='Train Acc', linewidth=2, markersize=6)
    ax2.plot(epochs, val_acc, 'r-s', label='Val Acc', linewidth=2, markersize=6)
    ax2.axhline(y=0.4, color='g', linestyle='--', label='Target (40%)', linewidth=2)
    ax2.axhline(y=0.6, color='orange', linestyle='--', label='Goal (60%)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Top-4 Overlap Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Save figure
    output_path = Path("predictor_checkpoints/training_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training curves saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nTotal epochs: {len(history)}")
    print(f"\nFinal results (Epoch {epochs[-1]}):")
    print(f"  Train Loss: {train_loss[-1]:.4f}")
    print(f"  Val Loss:   {val_loss[-1]:.4f}")
    print(f"  Train Acc:  {train_acc[-1]:.4f} ({train_acc[-1]*100:.2f}%)")
    print(f"  Val Acc:    {val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)")

    # Best results
    best_val_acc_idx = val_acc.index(max(val_acc))
    best_val_loss_idx = val_loss.index(min(val_loss))

    print(f"\nBest validation accuracy: {val_acc[best_val_acc_idx]:.4f} ({val_acc[best_val_acc_idx]*100:.2f}%) at epoch {epochs[best_val_acc_idx]}")
    print(f"Best validation loss: {val_loss[best_val_loss_idx]:.4f} at epoch {epochs[best_val_loss_idx]}")

    # Check success criteria
    print("\n" + "="*80)
    print("PHASE 2 SUCCESS CRITERIA")
    print("="*80)
    success_40 = val_acc[-1] >= 0.4
    success_60 = val_acc[-1] >= 0.6
    loss_decreased = val_loss[-1] < val_loss[0]

    print(f"✅ Top-4 accuracy > 40%: {success_40} ({'PASS' if success_40 else 'FAIL'})")
    if success_60:
        print(f"🎯 Top-4 accuracy > 60%: {success_60} (EXCELLENT - exceeds goal!)")
    print(f"✅ Validation loss decreased: {loss_decreased} ({'PASS' if loss_decreased else 'FAIL'})")

    gap = abs(train_loss[-1] - val_loss[-1])
    no_overfitting = gap < 0.2
    print(f"✅ No overfitting (loss gap < 0.2): {no_overfitting} ({'PASS' if no_overfitting else 'FAIL'}, gap={gap:.4f})")

    if success_40 and loss_decreased and no_overfitting:
        print("\n🎉 Phase 2 training SUCCESSFUL! Ready for Phase 3.")
    else:
        print("\n⚠️  Some success criteria not met. Consider retraining with adjusted parameters.")

    print("="*80)

if __name__ == "__main__":
    try:
        visualize_training_history()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
