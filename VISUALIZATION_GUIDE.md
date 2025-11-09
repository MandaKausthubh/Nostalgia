# Visualization Guide

This guide explains how to use the visualization module for continual learning and robustness analysis.

## Overview

The visualization module (`src/visualize.py`) provides paper-quality plots for:
1. **Forgetting vs Steps**: Average forgetting over training steps with loss overlay
2. **Forgetting Heatmap**: Task-wise interference visualization
3. **Task Retention Curves**: How each task's accuracy evolves
4. **Method Comparison Bar Chart**: Compare final forgetting across methods
5. **ID vs OOD Scatter**: Pareto plot showing ID/OOD trade-offs
6. **Hessian Spectra**: Eigenvalue spectrum analysis
7. **Ablation Curves**: Hyperparameter sensitivity analysis

## Automatic Integration

Visualizations are automatically generated during training and evaluation when enabled in the configuration.

### Training

Visualizations are generated at the end of training if `visualize.enabled=True` in the config:

```yaml
# config/train/default.yaml
defaults:
  - visualize: default

visualize:
  enabled: true
  save_dir: logs/plots
  plots:
    forgetting_vs_steps: true
    forgetting_heatmap: true
    task_retention_curves: true
    hessian_spectra: true
```

### Evaluation

Visualizations are generated during evaluation:

```yaml
# config/eval/default.yaml
defaults:
  - visualize: default

visualize:
  enabled: true
  plots:
    id_ood_scatter: true
```

## Manual Usage

### Comparing Multiple Methods

After running multiple training experiments, you can compare methods:

```bash
python src/compare_methods.py \
    --methods Full-FT LoRA Nostalgia-Global Nostalgia-LWP \
    --metrics_files logs/plots/full_ft/metrics.json \
                   logs/plots/lora/metrics.json \
                   logs/plots/nostalgia_global/metrics.json \
                   logs/plots/nostalgia_lwp/metrics.json \
    --save_dir logs/plots/comparison \
    --plot_comparison \
    --plot_id_ood
```

### Standalone Visualization

You can also use the visualization functions directly:

```python
from src.visualize import plot_forgetting_vs_steps
from src.metrics import ContinualLearningMetrics

# Load metrics
metrics = ContinualLearningMetrics.from_dict(json.load(open("metrics.json")))

# Generate plot
plot_forgetting_vs_steps(
    avg_forgetting=metrics.avg_forgetting_over_time,
    train_loss=metrics.train_loss_over_time,
    task_boundaries=metrics.task_boundaries,
    task_names=metrics.task_names,
    save_dir="logs/plots",
)
```

## Configuration

### Visualization Config (`config/visualize/default.yaml`)

```yaml
visualize:
  enabled: true
  save_dir: logs/plots
  save_formats: ["png", "pdf"]
  log_to_wandb: true
  log_to_tensorboard: true
  
  plots:
    forgetting_vs_steps: true
    forgetting_heatmap: true
    task_retention_curves: true
    method_comparison_bar: true
    id_ood_scatter: true
    hessian_spectra: true
    ablation_curves: true
```

## Output Files

All plots are saved to `logs/plots/` (or specified `save_dir`) in both PNG and PDF formats:

- `forgetting_vs_steps.png` / `.pdf`
- `forgetting_heatmap.png` / `.pdf`
- `task_retention_curves.png` / `.pdf`
- `method_comparison_bar.png` / `.pdf`
- `id_ood_scatter.png` / `.pdf`
- `hessian_spectra.png` / `.pdf`
- `ablation_curves.png` / `.pdf`
- `metrics.json` (raw metrics data)

## Integration with Logging

### WandB

Plots are automatically logged to WandB if enabled:

```yaml
logger:
  wandb:
    enabled: true
    project: nostalgia-experiments
```

Plots appear in the WandB dashboard under the "plots" section.

### TensorBoard

Plots are automatically logged to TensorBoard if enabled:

```yaml
logger:
  tensorboard:
    enabled: true
    save_dir: logs
```

View plots in TensorBoard:

```bash
tensorboard --logdir logs
```

## Metrics Tracking

The `ContinualLearningMetrics` class tracks:

- **Accuracy History**: `acc_history[t][i]` = accuracy on task i after training task t
- **Forgetting**: `forgetting[t][i]` = forgetting of task i after task t
- **Training Loss**: Loss values over training steps
- **Task Boundaries**: Start and end steps for each task
- **Hessian Eigenvalues**: Eigenvalue spectra for curvature analysis

## Example Workflow

1. **Train with visualizations enabled**:
   ```bash
   python src/train.py train.method=nostalgia_global \
                       train.continual_learning=true \
                       train.num_tasks=5 \
                       visualize.enabled=true
   ```

2. **Check generated plots**:
   ```bash
   ls logs/plots/
   ```

3. **Compare methods** (after running multiple experiments):
   ```bash
   python src/compare_methods.py \
       --metrics_dir logs/plots \
       --plot_comparison \
       --plot_id_ood
   ```

## Customization

### Plot Aesthetics

Modify `src/visualize.py` to customize:
- Colors: Change `sns.color_palette("husl", ...)`
- Font sizes: Modify `matplotlib.rcParams`
- Figure sizes: Adjust `figsize` parameters
- Styles: Change `plt.style.use(...)`

### Adding New Plots

1. Add function to `src/visualize.py`
2. Export in `__all__`
3. Add config flag in `config/visualize/default.yaml`
4. Call in `src/train.py` or `src/eval.py`

## Troubleshooting

### No plots generated

- Check that `visualize.enabled=true` in config
- Verify metrics are being tracked (check `logs/plots/metrics.json`)
- Ensure data is available (e.g., continual learning for heatmap)

### Empty plots

- Verify metrics are being recorded during training
- Check that task boundaries are set correctly
- Ensure validation accuracy is being logged

### Missing Hessian spectra

- Only available when using Nostalgia methods
- Check that `nostalgia_obj.eigenvals` is populated
- Verify Hessian computation succeeded

## References

See `PROMPT_VISUALISATIONS.md` for detailed specifications of each visualization.

