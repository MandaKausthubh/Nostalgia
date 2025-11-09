# Epochs Per Task Configuration

## Current Implementation

The current implementation supports both **single-task training** and **continual learning** with multiple tasks.

### Single-Task Training (Default)

- **Configuration**: `max_epochs: 100` (default)
- **Usage**: Trains on a single dataset for the specified number of epochs
- **Example**:
  ```bash
  python src/train.py train.max_epochs=50
  ```

### Continual Learning (Multi-Task)

- **Configuration**: 
  - `continual_learning: true`
  - `epochs_per_task: 20` (epochs to train on each task)
  - `num_tasks: 5` (number of tasks)
- **Total epochs**: `num_tasks × epochs_per_task` (e.g., 5 × 20 = 100 epochs)
- **Usage**:
  ```bash
  python src/train.py train.continual_learning=true train.epochs_per_task=20 train.num_tasks=5
  ```

## Configuration Options

### In `config/train/default.yaml`:

```yaml
train:
  max_epochs: 100  # Total epochs for single-task training
  epochs_per_task: null  # Epochs per task (null = use max_epochs for single task)
  
  # Continual learning parameters
  continual_learning: false  # Enable continual learning
  num_tasks: 1  # Number of tasks
  epochs_per_task: null  # Epochs per task (required if continual_learning=true)
  save_task_checkpoints: true  # Save checkpoint at end of each task
```

### In `config/train/continual_learning.yaml`:

```yaml
train:
  continual_learning: true
  num_tasks: 5
  epochs_per_task: 20  # Each task trains for 20 epochs
  save_task_checkpoints: true
```

## Examples

### Example 1: Single Task (100 epochs)

```bash
python src/train.py train.max_epochs=100
```

This trains on one task for 100 epochs.

### Example 2: Continual Learning (5 tasks, 20 epochs each)

```bash
python src/train.py \
    train.continual_learning=true \
    train.num_tasks=5 \
    train.epochs_per_task=20
```

This trains on 5 tasks sequentially, 20 epochs per task (100 total epochs).

### Example 3: Using Continual Learning Config

```bash
python src/train.py --config-name train/continual_learning
```

### Example 4: With Argparse

```bash
# Note: argparse version doesn't currently support continual learning
# Use Hydra config for continual learning experiments
python src/train_argparse.py --max_epochs 100
```

## Implementation Notes

### Current Limitations

1. **Task Data Loading**: The current implementation assumes the same dataset for all tasks. For true continual learning with different datasets per task, you would need to:
   - Implement task-specific data loaders
   - Load different datasets for each task
   - Handle task boundaries properly

2. **Checkpoint Loading**: Between tasks, the implementation should:
   - Load the checkpoint from the previous task
   - Continue training from that checkpoint
   - Reset optimizer/scheduler if needed

3. **Evaluation**: For continual learning, you typically want to:
   - Evaluate on all previous tasks after each new task
   - Compute forgetting metrics
   - Track performance per task

### Recommended Settings

For **continual learning experiments** (as mentioned in PROMPT.md):
- **epochs_per_task**: 20-50 epochs (depending on dataset size)
- **num_tasks**: 3-10 tasks (depending on experiment design)
- **save_task_checkpoints**: `true` (to build model soups from task checkpoints)

For **single-task robustness experiments**:
- **max_epochs**: 50-100 epochs (standard fine-tuning)
- **continual_learning**: `false`

## Model Soups with Task Checkpoints

As mentioned in PROMPT.md, you can build model soups from task checkpoints:

```bash
# After training with continual_learning=true, collect task checkpoints
python src/eval.py \
    eval.soup.enabled=true \
    eval.soup.type=uniform \
    eval.soup.checkpoint_paths=[
        checkpoints/task-1-nostalgia_global-epoch=20.ckpt,
        checkpoints/task-2-nostalgia_global-epoch=40.ckpt,
        checkpoints/task-3-nostalgia_global-epoch=60.ckpt
    ]
```

## Future Improvements

For a complete continual learning implementation, consider:

1. **Task-specific data modules**: Load different datasets for each task
2. **Task boundaries**: Properly handle task transitions
3. **Forgetting metrics**: Compute and log forgetting after each task
4. **Task-aware checkpoints**: Save and load task-specific checkpoints
5. **Evaluation after each task**: Evaluate on all previous tasks

