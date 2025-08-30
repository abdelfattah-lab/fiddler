#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


def run_test_comparison():
    """Run a simple test comparison with a small model or reduced parameters"""
    
    print("🎯 Running Fiddler Approach Comparison...")
    print()
    
    # Test inputs
    test_inputs = [
        "University of Washington is",
        "The quick brown fox",
        "Machine learning is"
    ]
    
    # Check if we have matplotlib installed
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        has_matplotlib = True
    except ImportError:
        print("⚠️  matplotlib not found. Installing it for visualization...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
            import matplotlib
            matplotlib.use('Agg')
            has_matplotlib = True
        except Exception as e:
            print(f"❌ Failed to install matplotlib: {e}")
            print("Proceeding without visualization...")
            has_matplotlib = False
    
    results_summary = []
    
    for i, test_input in enumerate(test_inputs):
        print(f"=== Test {i+1}/3: '{test_input}' ===")
        
        # Run comparison for this input
        cmd = [
            sys.executable, "compare_approaches.py",
            "--input", test_input,
            "--n-token", "10",  # Small number for quick testing
            "--runs", "2"       # Few runs for quick testing
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
            if result.returncode == 0:
                print("✅ Comparison completed successfully!")
                print("Output:")
                print(result.stdout)
                
                # Extract speedup from output
                lines = result.stdout.split('\n')
                speedup_line = [line for line in lines if 'SPEEDUP:' in line]
                if speedup_line:
                    speedup = speedup_line[0].split('SPEEDUP:')[1].strip()
                    results_summary.append(f"'{test_input}': {speedup}")
            else:
                print("❌ Comparison failed!")
                print("Error:")
                print(result.stderr)
                results_summary.append(f"'{test_input}': FAILED")
                
        except Exception as e:
            print(f"❌ Error running comparison: {e}")
            results_summary.append(f"'{test_input}': ERROR")
        
        print()
    
    # Print summary
    print("=" * 50)
    print("🎉 FINAL SUMMARY")
    print("=" * 50)
    for result in results_summary:
        print(f"  {result}")
    print()
    print("📊 Check the generated visualization files for detailed results!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test comparison between approaches")
    parser.add_argument("--full-test", action="store_true", help="Run with more tokens and runs")
    
    args = parser.parse_args()
    
    if args.full_test:
        print("Running full test with more tokens and runs...")
        # You can modify the test parameters here for more thorough testing
        # This would take longer but provide more accurate results
    
    run_test_comparison()