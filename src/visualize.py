"""
Visualization module for continual learning and robustness analysis.

Produces paper-quality plots for forgetting, accuracy, and method comparisons.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for paper-quality figures
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for paper quality
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


__all__ = [
    "plot_forgetting_vs_steps",
    "plot_forgetting_heatmap",
    "plot_task_retention_curves",
    "plot_method_comparison_bar",
    "plot_id_ood_scatter",
    "plot_hessian_spectra",
    "plot_ablation_curves",
    "save_and_log_plot",
]


def save_and_log_plot(fig, filename: str, save_dir: str, wandb_logger=None, 
                      tensorboard_writer=None, global_step: int = 0):
    """
    Save plot and log to wandb/TensorBoard.
    
    Args:
        fig: Matplotlib figure
        filename: Filename (without extension)
        save_dir: Directory to save plots
        wandb_logger: Optional wandb logger
        tensorboard_writer: Optional TensorBoard writer
        global_step: Global step for logging
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as PNG and PDF
    png_path = os.path.join(save_dir, f"{filename}.png")
    pdf_path = os.path.join(save_dir, f"{filename}.pdf")
    
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Log to wandb
    if wandb_logger is not None:
        try:
            import wandb
            wandb.log({f"plots/{filename}": wandb.Image(png_path)}, step=global_step)
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")
    
    # Log to TensorBoard
    if tensorboard_writer is not None:
        try:
            from PIL import Image
            import io
            
            # Save figure to bytes
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            # Load as PIL Image and convert to numpy
            img = Image.open(buf)
            img_array = np.array(img)
            
            # Convert to CHW format
            if len(img_array.shape) == 3:
                img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
            else:
                img_array = np.expand_dims(img_array, 0)  # Add channel dimension
            
            tensorboard_writer.add_image(f"plots/{filename}", img_array, global_step)
            buf.close()
        except Exception as e:
            print(f"Warning: Failed to log to TensorBoard: {e}")
    
    plt.close(fig)


def plot_forgetting_vs_steps(
    avg_forgetting: List[float],
    train_loss: List[float],
    task_boundaries: List[Tuple[int, int]],
    task_names: List[str],
    save_dir: str,
    wandb_logger=None,
    tensorboard_writer=None,
    global_step: int = 0,
) -> plt.Figure:
    """
    Plot average forgetting vs training steps with training loss overlay.
    
    Args:
        avg_forgetting: List of average forgetting values
        train_loss: List of training losses
        task_boundaries: List of (start_step, end_step) tuples
        task_names: List of task names
        save_dir: Directory to save plots
        wandb_logger: Optional wandb logger
        tensorboard_writer: Optional TensorBoard writer
        global_step: Global step for logging
    """
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    # Use task boundaries to determine step ranges if available
    if task_boundaries and len(task_boundaries) > 0:
        # Map forgetting to task boundaries
        max_step = max(end for _, end in task_boundaries) if task_boundaries else len(avg_forgetting)
        steps = list(range(max_step))
        
        # Interpolate forgetting values to steps
        if len(avg_forgetting) < len(steps):
            # Repeat last value or interpolate
            forgetting_interp = avg_forgetting + [avg_forgetting[-1] if avg_forgetting else 0.0] * (len(steps) - len(avg_forgetting))
        else:
            forgetting_interp = avg_forgetting[:len(steps)]
        
        # Interpolate loss values
        if len(train_loss) < len(steps):
            loss_interp = train_loss + [train_loss[-1] if train_loss else 0.5] * (len(steps) - len(train_loss))
        else:
            loss_interp = train_loss[:len(steps)]
    else:
        # No task boundaries, use array lengths
        max_len = max(len(avg_forgetting), len(train_loss), 1)
        steps = list(range(max_len))
        
        # Pad if necessary
        if len(avg_forgetting) < max_len:
            avg_forgetting = avg_forgetting + [avg_forgetting[-1] if avg_forgetting else 0.0] * (max_len - len(avg_forgetting))
        if len(train_loss) < max_len:
            train_loss = train_loss + [train_loss[-1] if train_loss else 0.5] * (max_len - len(train_loss))
        
        forgetting_interp = avg_forgetting
        loss_interp = train_loss
    
    # Plot forgetting on primary axis
    color1 = 'tab:red'
    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Average Forgetting', color=color1, fontsize=11)
    line1 = ax1.plot(steps, forgetting_interp, color=color1, label='Average Forgetting', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss on secondary axis
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Training Loss', color=color2, fontsize=11)
    line2 = ax2.plot(steps, loss_interp, color=color2, label='Training Loss', linewidth=2, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add task boundaries as vertical shaded regions
    if task_boundaries and len(task_boundaries) > 0:
        colors = sns.color_palette("husl", len(task_names))
        for i, (start, end) in enumerate(task_boundaries):
            if start < len(steps) and end <= len(steps):
                label = task_names[i] if i < len(task_names) else f"Task {i+1}"
                ax1.axvspan(start, end, alpha=0.2, color=colors[i % len(colors)], label=label)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # Add task boundary labels if any
    if task_boundaries and len(task_boundaries) > 0:
        # Create custom legend with lines and task regions
        from matplotlib.patches import Patch
        legend_elements = lines + [
            Patch(facecolor=sns.color_palette("husl", len(task_names))[i % len(task_names)], 
                  alpha=0.2, label=task_names[i] if i < len(task_names) else f"Task {i+1}")
            for i in range(min(len(task_boundaries), len(task_names)))
        ]
        ax1.legend(legend_elements, [l.get_label() for l in lines] + 
                  [task_names[i] if i < len(task_names) else f"Task {i+1}" 
                   for i in range(min(len(task_boundaries), len(task_names)))],
                  loc='upper left', fontsize=8, ncol=2)
    else:
        ax1.legend(lines, labels, loc='upper left', fontsize=9)
    
    plt.title('Average Forgetting vs Training Steps', fontsize=12)
    plt.tight_layout()
    
    save_and_log_plot(fig, "forgetting_vs_steps", save_dir, wandb_logger, 
                     tensorboard_writer, global_step)
    return fig


def plot_forgetting_heatmap(
    forgetting_matrix: np.ndarray,
    task_names: List[str],
    save_dir: str,
    wandb_logger=None,
    tensorboard_writer=None,
    global_step: int = 0,
) -> plt.Figure:
    """
    Plot task-wise forgetting heatmap.
    
    Args:
        forgetting_matrix: Matrix of shape (num_tasks, num_tasks)
        task_names: List of task names
        save_dir: Directory to save plots
        wandb_logger: Optional wandb logger
        tensorboard_writer: Optional TensorBoard writer
        global_step: Global step for logging
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Create heatmap
    sns.heatmap(
        forgetting_matrix,
        annot=True,
        fmt='.2f',
        cmap='Reds',
        cbar_kws={'label': 'Forgetting'},
        xticklabels=task_names,
        yticklabels=task_names,
        ax=ax,
        linewidths=0.5,
    )
    
    ax.set_xlabel('Past Tasks', fontsize=11)
    ax.set_ylabel('Current Task', fontsize=11)
    ax.set_title('Task-wise Forgetting Heatmap', fontsize=12)
    
    plt.tight_layout()
    
    save_and_log_plot(fig, "forgetting_heatmap", save_dir, wandb_logger,
                     tensorboard_writer, global_step)
    return fig


def plot_task_retention_curves(
    acc_history: Dict[int, List[float]],
    task_names: List[str],
    save_dir: str,
    wandb_logger=None,
    tensorboard_writer=None,
    global_step: int = 0,
) -> plt.Figure:
    """
    Plot task retention curves showing how each task's accuracy evolves.
    
    Args:
        acc_history: Dictionary mapping task_id to list of accuracies
        task_names: List of task names
        save_dir: Directory to save plots
        wandb_logger: Optional wandb logger
        tensorboard_writer: Optional TensorBoard writer
        global_step: Global step for logging
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    if not acc_history:
        print("Warning: No accuracy history data for retention curves")
        return fig
    
    max_task_id = max(acc_history.keys()) if acc_history else 0
    colors = sns.color_palette("husl", max(max_task_id + 1, len(task_names)))
    
    for task_id, accuracies in sorted(acc_history.items()):
        if len(accuracies) > 0:
            # Convert to percentages if needed
            accuracies_pct = [acc * 100 if acc <= 1.0 else acc for acc in accuracies]
            
            # Task indices: task_id, task_id+1, ..., task_id+len(accuracies)-1
            task_indices = list(range(task_id, task_id + len(accuracies)))
            label = task_names[task_id] if task_id < len(task_names) else f"Task {task_id + 1}"
            ax.plot(task_indices, accuracies_pct, marker='o', label=label, 
                   color=colors[task_id % len(colors)], linewidth=2, markersize=4)
    
    ax.set_xlabel('Task Index', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Task Retention Curves', fontsize=12)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_and_log_plot(fig, "task_retention_curves", save_dir, wandb_logger,
                     tensorboard_writer, global_step)
    return fig


def plot_method_comparison_bar(
    results_dict: Dict[str, Dict[str, float]],
    save_dir: str,
    wandb_logger=None,
    tensorboard_writer=None,
    global_step: int = 0,
) -> plt.Figure:
    """
    Plot method comparison bar chart.
    
    Args:
        results_dict: Dictionary mapping method names to metrics
        save_dir: Directory to save plots
        wandb_logger: Optional wandb logger
        tensorboard_writer: Optional TensorBoard writer
        global_step: Global step for logging
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    methods = list(results_dict.keys())
    mean_forgetting = [results_dict[m].get("mean_forgetting", 0.0) for m in methods]
    std_forgetting = [results_dict[m].get("std", 0.0) for m in methods]
    ood_acc = [results_dict[m].get("ood_accuracy", 0.0) for m in methods]
    
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, mean_forgetting, yerr=std_forgetting, capsize=5, 
                 alpha=0.7, color=sns.color_palette("husl", len(methods)))
    
    # Annotate OOD accuracy above bars
    for i, (bar, acc) in enumerate(zip(bars, ood_acc)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_forgetting[i] + 0.01,
               f'OOD: {acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Final Average Forgetting', fontsize=11)
    ax.set_title('Method Comparison', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_and_log_plot(fig, "method_comparison_bar", save_dir, wandb_logger,
                     tensorboard_writer, global_step)
    return fig


def plot_id_ood_scatter(
    eval_metrics: Dict[str, Tuple[float, float]],
    save_dir: str,
    wandb_logger=None,
    tensorboard_writer=None,
    global_step: int = 0,
) -> plt.Figure:
    """
    Plot ID vs OOD scatter plot (Pareto plot).
    
    Args:
        eval_metrics: Dictionary mapping method names to (ID_accuracy, OOD_accuracy)
        save_dir: Directory to save plots
        wandb_logger: Optional wandb logger
        tensorboard_writer: Optional TensorBoard writer
        global_step: Global step for logging
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    methods = list(eval_metrics.keys())
    id_accs = [eval_metrics[m][0] for m in methods]
    ood_accs = [eval_metrics[m][1] for m in methods]
    
    colors = sns.color_palette("husl", len(methods))
    scatter = ax.scatter(id_accs, ood_accs, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add labels
    for i, method in enumerate(methods):
        ax.annotate(method, (id_accs[i], ood_accs[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Draw Pareto frontier (upper envelope)
    if len(id_accs) > 1:
        # Sort by ID accuracy
        sorted_indices = np.argsort(id_accs)
        sorted_id = np.array(id_accs)[sorted_indices]
        sorted_ood = np.array(ood_accs)[sorted_indices]
        
        # Find Pareto-optimal points
        pareto_mask = np.ones(len(sorted_id), dtype=bool)
        for i in range(len(sorted_id)):
            for j in range(len(sorted_id)):
                if i != j and sorted_id[j] >= sorted_id[i] and sorted_ood[j] >= sorted_ood[i]:
                    if sorted_id[j] > sorted_id[i] or sorted_ood[j] > sorted_ood[i]:
                        pareto_mask[i] = False
                        break
        
        pareto_id = sorted_id[pareto_mask]
        pareto_ood = sorted_ood[pareto_mask]
        
        # Sort for plotting
        pareto_sort = np.argsort(pareto_id)
        pareto_id = pareto_id[pareto_sort]
        pareto_ood = pareto_ood[pareto_sort]
        
        # Plot Pareto frontier
        ax.plot(pareto_id, pareto_ood, 'r--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    ax.set_xlabel('ID Accuracy (%)', fontsize=11)
    ax.set_ylabel('Avg OOD Accuracy (%)', fontsize=11)
    ax.set_title('ID vs OOD Performance', fontsize=12)
    ax.grid(True, alpha=0.3)
    if len(id_accs) > 1:
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    save_and_log_plot(fig, "id_ood_scatter", save_dir, wandb_logger,
                     tensorboard_writer, global_step)
    return fig


def plot_hessian_spectra(
    hessian_eigs: Dict[str, np.ndarray],
    save_dir: str,
    wandb_logger=None,
    tensorboard_writer=None,
    global_step: int = 0,
) -> plt.Figure:
    """
    Plot Hessian eigenvalue spectra.
    
    Args:
        hessian_eigs: Dictionary mapping task names to eigenvalue arrays
        save_dir: Directory to save plots
        wandb_logger: Optional wandb logger
        tensorboard_writer: Optional TensorBoard writer
        global_step: Global step for logging
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    colors = sns.color_palette("husl", len(hessian_eigs))
    
    for i, (task_name, eigenvalues) in enumerate(hessian_eigs.items()):
        if len(eigenvalues) > 0:
            # Sort eigenvalues in descending order
            sorted_eigs = np.sort(eigenvalues)[::-1]
            indices = np.arange(1, len(sorted_eigs) + 1)
            
            ax.plot(indices, sorted_eigs, marker='o', label=task_name, 
                   color=colors[i % len(colors)], linewidth=2, markersize=3)
    
    ax.set_xlabel('Eigenvalue Index', fontsize=11)
    ax.set_ylabel('Eigenvalue Magnitude (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Hessian Eigenvalue Spectra', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    save_and_log_plot(fig, "hessian_spectra", save_dir, wandb_logger,
                     tensorboard_writer, global_step)
    return fig


def plot_ablation_curves(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_dir: str,
    wandb_logger=None,
    tensorboard_writer=None,
    global_step: int = 0,
) -> plt.Figure:
    """
    Plot ablation curves for hyperparameter sensitivity.
    
    Args:
        ablation_results: Dictionary mapping hyperparameter names to {values, metric}
        save_dir: Directory to save plots
        wandb_logger: Optional wandb logger
        tensorboard_writer: Optional TensorBoard writer
        global_step: Global step for logging
    """
    num_params = len(ablation_results)
    if num_params == 0:
        return None
    
    fig, axes = plt.subplots(1, num_params, figsize=(7 * num_params, 4))
    if num_params == 1:
        axes = [axes]
    
    colors = sns.color_palette("husl", num_params)
    
    for idx, (param_name, data) in enumerate(ablation_results.items()):
        ax = axes[idx]
        values = data.get("values", [])
        metrics = data.get("metric", [])
        
        if len(values) > 0 and len(metrics) > 0:
            ax.plot(values, metrics, marker='o', color=colors[idx], linewidth=2, markersize=6)
            
            # Mark minimum
            min_idx = np.argmin(metrics)
            ax.plot(values[min_idx], metrics[min_idx], 'ro', markersize=10, label='Best')
            ax.annotate(f'Best: {values[min_idx]}', 
                       xy=(values[min_idx], metrics[min_idx]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('Metric (Forgetting)', fontsize=11)
        ax.set_title(f'Ablation: {param_name}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if len(values) > 0 and len(metrics) > 0:
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    save_and_log_plot(fig, "ablation_curves", save_dir, wandb_logger,
                     tensorboard_writer, global_step)
    return fig


if __name__ == "__main__":
    # Example usage with mock data
    import tempfile
    
    save_dir = tempfile.mkdtemp()
    print(f"Creating example plots in {save_dir}")
    
    # Mock data
    num_tasks = 5
    task_names = [f"Task{i+1}" for i in range(num_tasks)]
    
    # Example 1: Forgetting vs Steps
    avg_forgetting = [0.0, 0.05, 0.08, 0.10, 0.12]
    train_loss = [0.5, 0.3, 0.2, 0.15, 0.12]
    task_boundaries = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    plot_forgetting_vs_steps(avg_forgetting, train_loss, task_boundaries, task_names, save_dir)
    
    # Example 2: Forgetting heatmap
    forgetting_matrix = np.random.rand(num_tasks, num_tasks) * 0.2
    forgetting_matrix = np.triu(forgetting_matrix)  # Upper triangular
    plot_forgetting_heatmap(forgetting_matrix, task_names, save_dir)
    
    # Example 3: Task retention
    acc_history = {
        0: [0.85, 0.82, 0.80, 0.78, 0.75],
        1: [0.88, 0.85, 0.83, 0.80],
        2: [0.87, 0.84, 0.81],
        3: [0.86, 0.83],
        4: [0.85],
    }
    plot_task_retention_curves(acc_history, task_names, save_dir)
    
    # Example 4: Method comparison
    results_dict = {
        "Full FT": {"mean_forgetting": 0.12, "std": 0.03, "ood_accuracy": 51.4},
        "LoRA": {"mean_forgetting": 0.09, "std": 0.02, "ood_accuracy": 52.7},
        "Nostalgia-Global": {"mean_forgetting": 0.05, "std": 0.01, "ood_accuracy": 53.8},
        "Nostalgia-LWP": {"mean_forgetting": 0.04, "std": 0.01, "ood_accuracy": 53.6},
    }
    plot_method_comparison_bar(results_dict, save_dir)
    
    # Example 5: ID vs OOD
    eval_metrics = {
        "Full FT": (80.2, 51.4),
        "LoRA": (79.8, 52.7),
        "Nostalgia-Global": (80.1, 53.8),
        "Nostalgia-LWP": (79.9, 53.6),
    }
    plot_id_ood_scatter(eval_metrics, save_dir)
    
    # Example 6: Hessian spectra
    hessian_eigs = {
        "Task1": np.random.exponential(0.1, 50),
        "Task2": np.random.exponential(0.15, 50),
        "Task3": np.random.exponential(0.12, 50),
    }
    plot_hessian_spectra(hessian_eigs, save_dir)
    
    # Example 7: Ablation curves
    ablation_results = {
        "lanczos_k": {"values": [5, 10, 20, 50], "metric": [0.07, 0.05, 0.04, 0.035]},
        "lora_rank": {"values": [2, 4, 8, 16], "metric": [0.06, 0.05, 0.05, 0.055]},
    }
    plot_ablation_curves(ablation_results, save_dir)
    
    print(f"Example plots saved to {save_dir}")

