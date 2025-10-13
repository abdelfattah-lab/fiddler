#!/usr/bin/env python3
"""
verify_training_data.py - Verify collected training data format and integrity
"""

import h5py
import numpy as np
from pathlib import Path
import sys

def verify_single_file(file_path):
    """Verify a single HDF5 file."""
    print(f"\n{'='*80}")
    print(f"Verifying: {file_path}")
    print(f"{'='*80}")

    try:
        with h5py.File(file_path, 'r') as f:
            # Check datasets exist
            expected_datasets = ['attention_outputs', 'gating_scores']
            actual_datasets = list(f.keys())

            print(f"✓ Datasets: {actual_datasets}")

            if set(expected_datasets) != set(actual_datasets):
                print(f"✗ ERROR: Expected datasets {expected_datasets}, got {actual_datasets}")
                return False

            # Check shapes
            attn_shape = f['attention_outputs'].shape
            gating_shape = f['gating_scores'].shape

            print(f"✓ Attention outputs shape: {attn_shape}")
            print(f"✓ Gating scores shape: {gating_shape}")

            # Validate shapes
            if len(attn_shape) != 2 or attn_shape[1] != 2048:
                print(f"✗ ERROR: Attention outputs should be [n_samples, 2048], got {attn_shape}")
                return False

            if len(gating_shape) != 3 or gating_shape[1] != 22 or gating_shape[2] != 60:
                print(f"✗ ERROR: Gating scores should be [n_samples, 22, 60], got {gating_shape}")
                return False

            if attn_shape[0] != gating_shape[0]:
                print(f"✗ ERROR: Mismatched sample counts: attn={attn_shape[0]}, gating={gating_shape[0]}")
                return False

            # Check attributes
            print(f"\nAttributes:")
            required_attrs = ['n_samples', 'hidden_dim', 'n_moe_layers_to_predict',
                            'moe_layers_to_predict', 'n_experts', 'dataset', 'split']

            for attr in required_attrs:
                if attr in f.attrs:
                    value = f.attrs[attr]
                    print(f"  ✓ {attr}: {value}")
                else:
                    print(f"  ✗ Missing attribute: {attr}")
                    return False

            # Validate attribute values
            if f.attrs['n_samples'] != attn_shape[0]:
                print(f"✗ ERROR: n_samples attribute ({f.attrs['n_samples']}) != actual samples ({attn_shape[0]})")
                return False

            if f.attrs['hidden_dim'] != 2048:
                print(f"✗ ERROR: hidden_dim should be 2048, got {f.attrs['hidden_dim']}")
                return False

            if f.attrs['n_moe_layers_to_predict'] != 22:
                print(f"✗ ERROR: n_moe_layers_to_predict should be 22, got {f.attrs['n_moe_layers_to_predict']}")
                return False

            if f.attrs['n_experts'] != 60:
                print(f"✗ ERROR: n_experts should be 60, got {f.attrs['n_experts']}")
                return False

            if f.attrs['dataset'] != 'wikitext-103-raw-v1':
                print(f"✗ ERROR: dataset should be 'wikitext-103-raw-v1', got {f.attrs['dataset']}")
                return False

            if f.attrs['split'] != 'train':
                print(f"✗ ERROR: split should be 'train', got {f.attrs['split']}")
                return False

            # Check data integrity
            print(f"\nData integrity checks:")

            # Load small sample
            attn_data = f['attention_outputs'][:10]
            gating_data = f['gating_scores'][:10]

            # Check for NaN or Inf
            if np.any(np.isnan(attn_data)) or np.any(np.isinf(attn_data)):
                print(f"✗ ERROR: Attention outputs contain NaN or Inf")
                return False
            else:
                print(f"  ✓ Attention outputs: No NaN or Inf")

            if np.any(np.isnan(gating_data)) or np.any(np.isinf(gating_data)):
                print(f"✗ ERROR: Gating scores contain NaN or Inf")
                return False
            else:
                print(f"  ✓ Gating scores: No NaN or Inf")

            # Check that gating scores are valid probabilities (sum to ~1 for each layer)
            gating_sums = np.sum(gating_data, axis=2)  # Sum over experts
            expected_sum = 1.0
            if not np.allclose(gating_sums, expected_sum, atol=1e-5):
                print(f"✗ ERROR: Gating scores don't sum to 1 (should be probability distribution)")
                print(f"  Sample sums: {gating_sums[0, :5]}")
                return False
            else:
                print(f"  ✓ Gating scores sum to 1 (valid probability distribution)")

            # Check that gating scores are non-negative
            if np.any(gating_data < 0):
                print(f"✗ ERROR: Gating scores contain negative values")
                return False
            else:
                print(f"  ✓ Gating scores are non-negative")

            print(f"\n✓ File verification PASSED")
            return True

    except Exception as e:
        print(f"✗ ERROR: Exception while verifying file: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_all_data(data_dir="predictor_training_data"):
    """Verify all collected data."""
    print(f"\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*80}")

    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"✗ ERROR: Directory {data_dir} does not exist")
        return False

    # Find all HDF5 files
    h5_files = sorted(data_path.glob('training_data_*.h5'))

    if len(h5_files) == 0:
        print(f"✗ ERROR: No training data files found in {data_dir}")
        return False

    print(f"Found {len(h5_files)} HDF5 files")

    # Verify each file
    all_passed = True
    total_samples = 0
    total_size = 0

    for file_path in h5_files:
        passed = verify_single_file(file_path)
        if not passed:
            all_passed = False
            print(f"✗ FAILED: {file_path}")

        # Count samples and size
        with h5py.File(file_path, 'r') as f:
            total_samples += f.attrs['n_samples']

        total_size += file_path.stat().st_size

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {len(h5_files)}")
    print(f"Total samples: {total_samples}")
    print(f"Total size: {total_size / (1024**2):.2f} MB")

    if all_passed:
        print(f"\n✓✓✓ ALL FILES PASSED VERIFICATION ✓✓✓")
        return True
    else:
        print(f"\n✗✗✗ SOME FILES FAILED VERIFICATION ✗✗✗")
        return False

def main():
    if len(sys.argv) > 1:
        # Verify specific file
        file_path = sys.argv[1]
        passed = verify_single_file(file_path)
        sys.exit(0 if passed else 1)
    else:
        # Verify all data
        passed = verify_all_data()
        sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
