"""
Script to compare multiple methods and generate comparison plots.

Aggregates results from multiple training runs and creates comparison visualizations.
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from src.visualize import plot_method_comparison_bar, plot_id_ood_scatter
from src.metrics import ContinualLearningMetrics


def load_metrics_from_file(metrics_file: str) -> ContinualLearningMetrics:
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    return ContinualLearningMetrics.from_dict(data)


def aggregate_results(metrics_files: Dict[str, str]) -> Dict[str, Dict]:
    """
    Aggregate results from multiple metrics files.
    
    Args:
        metrics_files: Dictionary mapping method names to metrics file paths
        
    Returns:
        Dictionary with aggregated results
    """
    results = {}
    
    for method_name, metrics_file in metrics_files.items():
        if not os.path.exists(metrics_file):
            print(f"Warning: Metrics file not found: {metrics_file}")
            continue
        
        metrics = load_metrics_from_file(metrics_file)
        
        # Compute average forgetting
        if len(metrics.avg_forgetting_over_time) > 0:
            mean_forgetting = np.mean(metrics.avg_forgetting_over_time)
            std_forgetting = np.std(metrics.avg_forgetting_over_time)
        else:
            # Compute from forgetting matrix
            forgetting_matrix = metrics.get_forgetting_matrix()
            if forgetting_matrix.size > 0:
                # Get final forgetting (last row, excluding diagonal)
                final_forgetting = forgetting_matrix[-1, :]
                mean_forgetting = np.mean(final_forgetting[final_forgetting > 0]) if np.any(final_forgetting > 0) else 0.0
                std_forgetting = np.std(final_forgetting[final_forgetting > 0]) if np.any(final_forgetting > 0) else 0.0
            else:
                mean_forgetting = 0.0
                std_forgetting = 0.0
        
        # Get eval metrics
        id_acc = metrics.eval_metrics.get(method_name, {}).get("id_accuracy", 0.0)
        ood_acc = metrics.eval_metrics.get(method_name, {}).get("ood_accuracy", 0.0)
        
        results[method_name] = {
            "mean_forgetting": mean_forgetting,
            "std": std_forgetting,
            "id_accuracy": id_acc,
            "ood_accuracy": ood_acc,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare multiple methods and generate plots")
    parser.add_argument("--metrics_dir", type=str, default="logs/plots",
                       help="Directory containing metrics JSON files")
    parser.add_argument("--methods", type=str, nargs="+",
                       help="List of method names to compare")
    parser.add_argument("--metrics_files", type=str, nargs="+",
                       help="List of metrics JSON file paths (same order as methods)")
    parser.add_argument("--save_dir", type=str, default="logs/plots",
                       help="Directory to save comparison plots")
    parser.add_argument("--plot_comparison", action="store_true",
                       help="Generate method comparison bar chart")
    parser.add_argument("--plot_id_ood", action="store_true",
                       help="Generate ID vs OOD scatter plot")
    
    args = parser.parse_args()
    
    # Determine metrics files
    if args.metrics_files:
        if args.methods and len(args.methods) == len(args.metrics_files):
            metrics_files = dict(zip(args.methods, args.metrics_files))
        else:
            # Use file names as method names
            metrics_files = {
                Path(f).stem: f for f in args.metrics_files
            }
    else:
        # Look for metrics.json files in metrics_dir
        metrics_dir = Path(args.metrics_dir)
        if args.methods:
            metrics_files = {
                method: metrics_dir / f"{method}_metrics.json"
                for method in args.methods
            }
        else:
            # Find all metrics.json files
            metrics_files = {
                f.parent.name: str(f)
                for f in metrics_dir.rglob("metrics.json")
            }
    
    if not metrics_files:
        print("No metrics files found. Please specify --metrics_files or --methods.")
        return
    
    print(f"Comparing {len(metrics_files)} methods:")
    for method, file in metrics_files.items():
        print(f"  {method}: {file}")
    
    # Aggregate results
    results = aggregate_results(metrics_files)
    
    if not results:
        print("No valid results found.")
        return
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate comparison bar chart
    if args.plot_comparison or not args.plot_id_ood:
        print("\nGenerating method comparison bar chart...")
        plot_method_comparison_bar(results, args.save_dir)
    
    # Generate ID vs OOD scatter
    if args.plot_id_ood or not args.plot_comparison:
        print("\nGenerating ID vs OOD scatter plot...")
        eval_metrics = {
            method: (results[method]["id_accuracy"], results[method]["ood_accuracy"])
            for method in results.keys()
        }
        plot_id_ood_scatter(eval_metrics, args.save_dir)
    
    print(f"\nComparison plots saved to {args.save_dir}")


if __name__ == "__main__":
    main()

