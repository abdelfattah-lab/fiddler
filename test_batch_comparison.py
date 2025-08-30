#!/usr/bin/env python3

import argparse
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_batch_comparison():
    """Test the updated comparison script with batch sizes"""
    
    print("🔧 Testing batch size comparison...")
    
    # Test with small parameters for quick testing
    cmd = [
        sys.executable, "compare_approaches.py",
        "--input", "Hello world", 
        "--n-token", "3",
        "--runs", "1",
        "--batch-sizes", "1", "2",  # Test just 2 batch sizes
        "--max-experts-gpu", "16"  # Limit GPU experts
    ]
    
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✅ Batch comparison test PASSED!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ Batch comparison test FAILED!")
            print("Error:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running batch comparison: {e}")


def test_expert_limitation():
    """Test that expert limitation works"""
    
    print("\n🔧 Testing expert limitation...")
    
    try:
        from fiddler.mixtral import FiddlerMixtral
        
        class Args:
            def __init__(self):
                self.model = 'mistralai/Mixtral-8x7B-v0.1'
                self.beam_width = 1
                self.cpu_offload = 1
                self.max_experts_gpu = 16  # Limit to 16 experts
                
        args = Args()
        model = FiddlerMixtral(args)
        
        # Check that it respected the limitation
        n_experts_on_gpu = model.calc_n_expert_on_gpu()
        print(f"  Experts on GPU with max_experts_gpu=16: {n_experts_on_gpu}")
        
        if n_experts_on_gpu <= 16:
            print("✅ Expert limitation works!")
        else:
            print(f"❌ Expert limitation failed - got {n_experts_on_gpu} > 16")
            
        # Test without limitation
        args.max_experts_gpu = None
        model2 = FiddlerMixtral(args)
        n_experts_auto = model2.calc_n_expert_on_gpu()
        print(f"  Experts on GPU without limitation: {n_experts_auto}")
        
        if n_experts_auto >= n_experts_on_gpu:
            print("✅ Auto-detection allows more experts as expected!")
        
    except Exception as e:
        print(f"❌ Expert limitation test failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-test-only", action="store_true", help="Only test expert limitation")
    
    args = parser.parse_args()
    
    # Set environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if args.expert_test_only:
        test_expert_limitation()
    else:
        test_expert_limitation()
        test_batch_comparison()