#!/usr/bin/env python
"""
Experiment runner script for Nostalgia experiments.

This script provides a convenient way to run experiments with different methods.

Usage:
    python run_experiment.py --method nostalgia_global --batch_size 64 --max_epochs 100
    python run_experiment.py --method full_ft
    python run_experiment.py --method lora --use_lora
"""
import os
import sys
import subprocess
from pathlib import Path

# Ensure we're in the project root
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

def run_experiment(method="full_ft", **kwargs):
    """
    Run an experiment with specified method and parameters.
    
    Args:
        method: Training method (full_ft, lora, nostalgia_global, nostalgia_layerwise)
        **kwargs: Additional parameters to pass to train.py
    """
    # Build command
    cmd = [sys.executable, "src/train.py", f"train.method={method}"]
    
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"{key}=true")
            else:
                cmd.append(f"{key}=false")
        elif isinstance(value, list):
            # Handle list values (e.g., for checkpoint paths)
            cmd.append(f"{key}={','.join(map(str, value))}")
        else:
            cmd.append(f"{key}={value}")
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print("-" * 80)
    
    # Run the command
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\nError: Experiment failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Nostalgia experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full fine-tuning
  python run_experiment.py --method full_ft
  
  # LoRA fine-tuning
  python run_experiment.py --method lora --use_lora
  
  # Nostalgia (Global)
  python run_experiment.py --method nostalgia_global --num_eigenthings 100
  
  # Nostalgia (Layer-wise)
  python run_experiment.py --method nostalgia_layerwise --num_eigenthings_per_layer 50
  
  # Custom parameters
  python run_experiment.py --method nostalgia_global --batch_size 64 --max_epochs 50 --lr 5e-5
        """
    )
    parser.add_argument("--method", type=str, default="full_ft",
                       choices=["full_ft", "lora", "nostalgia_global", "nostalgia_layerwise"],
                       help="Training method (default: full_ft)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA (for lora method)")
    parser.add_argument("--num_eigenthings", type=int, default=100,
                       help="Number of eigencomponents for nostalgia_global (default: 100)")
    parser.add_argument("--num_eigenthings_per_layer", type=int, default=50,
                       help="Number of eigencomponents per layer for nostalgia_layerwise (default: 50)")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory (overrides config)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset name (overrides config)")
    
    args = parser.parse_args()
    
    kwargs = {
        "train.batch_size": args.batch_size,
        "train.max_epochs": args.max_epochs,
        "train.lr": args.lr,
    }
    
    if args.data_dir:
        kwargs["dataset.data_dir"] = args.data_dir
    
    if args.dataset:
        kwargs["dataset.name"] = args.dataset
    
    if args.method == "lora" or args.use_lora:
        kwargs["model.use_lora"] = True
    
    if args.method == "nostalgia_global":
        kwargs["train.num_eigenthings"] = args.num_eigenthings
    
    if args.method == "nostalgia_layerwise":
        kwargs["train.num_eigenthings_per_layer"] = args.num_eigenthings_per_layer
    
    print("=" * 80)
    print("Nostalgia Experiments - Experiment Runner")
    print("=" * 80)
    print(f"Method: {args.method}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.lr}")
    if args.method == "nostalgia_global":
        print(f"Num eigencomponents: {args.num_eigenthings}")
    if args.method == "nostalgia_layerwise":
        print(f"Num eigencomponents per layer: {args.num_eigenthings_per_layer}")
    print("=" * 80)
    
    run_experiment(args.method, **kwargs)

