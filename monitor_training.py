#!/usr/bin/env python3
"""Monitor training progress for the expert predictor."""

import json
import os
import time
from pathlib import Path

def get_training_status():
    """Get current training status from logs and checkpoints."""
    checkpoint_dir = Path("predictor_checkpoints")

    print("="*80)
    print("TRAINING PROGRESS MONITOR")
    print("="*80)

    # Check if training has started
    if not checkpoint_dir.exists():
        print("\n❌ Checkpoint directory not found. Training has not started.")
        return

    # Check for config
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\n📋 Configuration:")
        print(f"   - Samples: {config.get('max_samples', 'all')}")
        print(f"   - Batch size: {config.get('batch_size')}")
        print(f"   - Epochs: {config.get('num_epochs')}")
        print(f"   - Learning rate: {config.get('learning_rate')}")
        print(f"   - Loss function: {config.get('loss_function')}")

    # Check for training history
    history_path = checkpoint_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

        print(f"\n📈 Training History ({len(history)} epochs completed):")
        print(f"\n   {'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
        print(f"   {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for entry in history:
            print(f"   {entry['epoch']:<6} "
                  f"{entry['train_loss']:<12.4f} "
                  f"{entry['train_acc']:<12.4f} "
                  f"{entry['val_loss']:<12.4f} "
                  f"{entry['val_acc']:<12.4f}")

        # Show best results
        best_val_acc = max(history, key=lambda x: x['val_acc'])
        best_val_loss = min(history, key=lambda x: x['val_loss'])

        print(f"\n   🏆 Best Validation Accuracy: {best_val_acc['val_acc']:.4f} (Epoch {best_val_acc['epoch']})")
        print(f"   🏆 Best Validation Loss: {best_val_loss['val_loss']:.4f} (Epoch {best_val_loss['epoch']})")
    else:
        print("\n⏳ Training in progress, no history available yet...")

    # Check for checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        print(f"\n💾 Checkpoints:")
        for cp in checkpoints[-5:]:  # Show last 5
            size_mb = cp.stat().st_size / 1024 / 1024
            print(f"   - {cp.name} ({size_mb:.1f} MB)")

    best_model = checkpoint_dir / "best_model.pt"
    if best_model.exists():
        size_mb = best_model.stat().st_size / 1024 / 1024
        print(f"   ✅ best_model.pt ({size_mb:.1f} MB)")

    # Check training log for current progress
    if Path("training.log").exists():
        with open("training.log", 'r') as f:
            lines = f.readlines()

        # Find most recent progress line
        recent_progress = []
        for line in reversed(lines[-50:]):
            if "Epoch" in line and "%" in line:
                recent_progress.append(line.strip())
                if len(recent_progress) >= 3:
                    break

        if recent_progress:
            print(f"\n📊 Recent Progress:")
            for line in reversed(recent_progress):
                # Clean up the line
                if "loss=" in line and "acc=" in line:
                    print(f"   {line}")

    print("\n" + "="*80)

if __name__ == "__main__":
    while True:
        try:
            get_training_status()
            print("\nRefreshing in 60 seconds... (Ctrl+C to stop)")
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
