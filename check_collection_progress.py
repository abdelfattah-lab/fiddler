#!/usr/bin/env python3
"""
check_collection_progress.py - Monitor data collection progress
"""

import h5py
import os
from pathlib import Path
import time

def check_progress():
    """Check current progress of data collection."""
    data_dir = Path("predictor_training_data")

    if not data_dir.exists():
        print("❌ Data directory does not exist yet. Collection may not have started.")
        return

    # Find all HDF5 files
    h5_files = sorted(data_dir.glob('training_data_*.h5'))

    if len(h5_files) == 0:
        print("❌ No data files found yet. Collection is starting...")
        return

    print(f"\n{'='*80}")
    print(f"DATA COLLECTION PROGRESS")
    print(f"{'='*80}\n")

    total_samples = 0
    total_size = 0

    for file_path in h5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                n_samples = f.attrs['n_samples']
                total_samples += n_samples

            file_size = file_path.stat().st_size
            total_size += file_size
            print(f"✓ {file_path.name}: {n_samples} samples, {file_size / (1024**2):.2f} MB")
        except Exception as e:
            print(f"⚠ {file_path.name}: Error reading file - {e}")

    target_samples = 50000
    progress_pct = (total_samples / target_samples) * 100

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total files:   {len(h5_files)}")
    print(f"Total samples: {total_samples:,} / {target_samples:,} ({progress_pct:.1f}%)")
    print(f"Total size:    {total_size / (1024**2):.2f} MB")

    if total_samples >= target_samples:
        print(f"\n✅ DATA COLLECTION COMPLETE!")
        print(f"\nNext steps:")
        print(f"  1. Run: python verify_training_data.py")
        print(f"  2. Check that all files pass verification")
        print(f"  3. Proceed to Phase 2")
    else:
        remaining = target_samples - total_samples
        print(f"\n⏳ Collection in progress...")
        print(f"   Remaining: {remaining:,} samples")
        print(f"\nTo monitor collection:")
        print(f"  - Check log: tail -f data_collection.log")
        print(f"  - Run this script again: python check_collection_progress.py")

    # Check if process is still running
    try:
        result = os.popen("ps aux | grep 'python.*collect_training_data.py' | grep -v grep").read()
        if result.strip():
            print(f"\n✓ Collection process is running")
        else:
            print(f"\n⚠ Collection process not found - may have finished or crashed")
            print(f"  Check: tail -100 data_collection.log")
    except Exception as e:
        print(f"\n⚠ Could not check process status: {e}")

if __name__ == "__main__":
    check_progress()
