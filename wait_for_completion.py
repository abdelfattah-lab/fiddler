#!/usr/bin/env python3
"""
wait_for_completion.py - Wait for data collection to complete and automatically verify
"""

import h5py
import os
import sys
import time
from pathlib import Path

def get_total_samples():
    """Get current total sample count."""
    data_dir = Path("predictor_training_data")
    if not data_dir.exists():
        return 0

    h5_files = sorted(data_dir.glob('training_data_*.h5'))
    if len(h5_files) == 0:
        return 0

    total = 0
    for file_path in h5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                total += f.attrs['n_samples']
        except:
            pass
    return total

def is_process_running():
    """Check if collection process is still running."""
    result = os.popen("ps aux | grep 'python.*collect_training_data.py' | grep -v grep").read()
    return bool(result.strip())

def wait_for_completion(target_samples=50000, check_interval=300):
    """
    Wait for data collection to complete.

    Args:
        target_samples: Target number of samples
        check_interval: Seconds between progress checks
    """
    print("Waiting for data collection to complete...")
    print(f"Target: {target_samples:,} samples")
    print(f"Checking progress every {check_interval} seconds ({check_interval//60} minutes)")
    print("")

    last_count = 0
    stall_checks = 0

    while True:
        current_count = get_total_samples()
        process_running = is_process_running()

        # Show progress
        if current_count >= target_samples:
            print(f"\n✅ Target reached: {current_count:,} / {target_samples:,} samples")
            break

        progress_pct = (current_count / target_samples) * 100
        samples_since_last = current_count - last_count

        print(f"Progress: {current_count:,} / {target_samples:,} ({progress_pct:.1f}%) "
              f"[+{samples_since_last} samples] "
              f"[Process: {'running' if process_running else 'NOT RUNNING'}]")

        # Check for stalled collection
        if current_count == last_count:
            stall_checks += 1
            if stall_checks >= 3 and not process_running:
                print(f"\n⚠ WARNING: Collection appears to have stalled!")
                print(f"   Samples collected: {current_count:,}")
                print(f"   Process status: NOT RUNNING")
                print(f"   Check log: tail -100 data_collection.log")
                return False
        else:
            stall_checks = 0

        last_count = current_count

        # Wait before next check
        time.sleep(check_interval)

    print("\n✅ Data collection complete!")
    return True

if __name__ == "__main__":
    success = wait_for_completion(target_samples=50000, check_interval=300)

    if success:
        print("\nRunning verification...")
        os.system("python verify_training_data.py")
    else:
        print("\nCollection did not complete successfully.")
        sys.exit(1)
