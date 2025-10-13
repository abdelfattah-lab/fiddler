#!/usr/bin/env python3
"""Wait for training to complete and validate Phase 2 completion."""

import json
import time
from pathlib import Path
import subprocess
import sys

def check_training_complete():
    """Check if training has completed all epochs."""
    config_path = Path("predictor_checkpoints/config.json")
    history_path = Path("predictor_checkpoints/training_history.json")

    if not config_path.exists():
        return False, "Config not found"

    with open(config_path, 'r') as f:
        config = json.load(f)

    target_epochs = config.get('num_epochs', 10)

    if not history_path.exists():
        return False, f"No history yet (target: {target_epochs} epochs)"

    with open(history_path, 'r') as f:
        history = json.load(f)

    completed_epochs = len(history)

    if completed_epochs >= target_epochs:
        return True, f"Training complete ({completed_epochs}/{target_epochs} epochs)"
    else:
        return False, f"Training in progress ({completed_epochs}/{target_epochs} epochs)"

def validate_phase2_completion():
    """Validate that Phase 2 requirements are met."""
    print("\n" + "="*80)
    print("PHASE 2 VALIDATION")
    print("="*80)

    checkpoint_dir = Path("predictor_checkpoints")

    # Check 1: best_model.pt exists
    best_model = checkpoint_dir / "best_model.pt"
    check1 = best_model.exists()
    print(f"\n✓ Check 1: best_model.pt exists: {check1}")

    # Check 2: All epoch checkpoints exist
    config_path = checkpoint_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    num_epochs = config['num_epochs']
    all_checkpoints_exist = all((checkpoint_dir / f"checkpoint_epoch_{i}.pt").exists()
                                 for i in range(1, num_epochs + 1))
    check2 = all_checkpoints_exist
    print(f"✓ Check 2: All {num_epochs} epoch checkpoints exist: {check2}")

    # Check 3: config.json and training_history.json exist
    check3 = (checkpoint_dir / "config.json").exists() and (checkpoint_dir / "training_history.json").exists()
    print(f"✓ Check 3: config.json and training_history.json exist: {check3}")

    # Check 4: Training metrics
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'r') as f:
        history = json.load(f)

    final_epoch = history[-1]
    val_acc = final_epoch['val_acc']
    val_loss = final_epoch['val_loss']
    train_loss = final_epoch['train_loss']

    check4_acc = val_acc > 0.4  # Minimum requirement
    check4_loss_decrease = val_loss < history[0]['val_loss']
    check4_no_overfit = abs(val_loss - train_loss) < 0.2

    print(f"\n✓ Check 4a: Validation top-4 accuracy > 40%: {check4_acc} ({val_acc:.4f} = {val_acc*100:.2f}%)")
    print(f"✓ Check 4b: Validation loss decreased: {check4_loss_decrease} ({history[0]['val_loss']:.4f} → {val_loss:.4f})")
    print(f"✓ Check 4c: No overfitting (gap < 0.2): {check4_no_overfit} (gap = {abs(val_loss - train_loss):.4f})")

    # Overall success
    all_checks = check1 and check2 and check3 and check4_acc and check4_loss_decrease and check4_no_overfit

    print("\n" + "="*80)
    if all_checks:
        print("✅ PHASE 2 COMPLETE - All validation checks passed!")
        print("   Ready to proceed to Phase 3: Evaluation")
    else:
        print("⚠️  Some validation checks failed. Review results above.")
    print("="*80)

    return all_checks

def main():
    """Main function to wait for completion and validate."""
    print("="*80)
    print("PHASE 2 TRAINING MONITOR")
    print("="*80)
    print("\nWaiting for training to complete...")
    print("This script will check every 2 minutes and validate when done.\n")

    check_interval = 120  # 2 minutes
    last_status = ""

    while True:
        complete, status = check_training_complete()

        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] {status}")
            last_status = status

        if complete:
            print(f"\n[{time.strftime('%H:%M:%S')}] 🎉 Training completed!")
            print("\nRunning visualization...")

            # Run visualization
            result = subprocess.run(
                ["python3", "visualize_training.py"],
                capture_output=True,
                text=True
            )

            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)

            # Validate Phase 2
            success = validate_phase2_completion()

            if success:
                print("\n✅ Phase 2 successfully completed and validated!")
                return 0
            else:
                print("\n⚠️  Phase 2 validation had issues. Check output above.")
                return 1

        time.sleep(check_interval)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(130)
