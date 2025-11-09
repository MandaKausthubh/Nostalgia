# Task Sequence in Continual Learning

## Current Default Behavior

**By default, there is NO explicit task sequence defined.** The implementation uses the same dataset for all tasks.

### Default Configuration

- **Single Task Mode** (default): Uses `dataset: imagenet1k` for training
- **Continual Learning Mode**: Uses the same dataset (`cfg.dataset.name`, default: `imagenet1k`) for all tasks

### Example: Default Continual Learning

If you run:
```bash
python src/train.py train.continual_learning=true train.num_tasks=5 train.epochs_per_task=20
```

The task sequence will be:
1. **Task 1**: imagenet1k (20 epochs)
2. **Task 2**: imagenet1k (20 epochs)
3. **Task 3**: imagenet1k (20 epochs)
4. **Task 4**: imagenet1k (20 epochs)
5. **Task 5**: imagenet1k (20 epochs)

**All tasks use the same dataset** (imagenet1k).

## Defining Custom Task Sequences

### Option 1: Using Config File

Create or modify `config/train/continual_learning.yaml`:

```yaml
train:
  continual_learning: true
  num_tasks: 5
  epochs_per_task: 20

# Define task sequence with different datasets
tasks:
  - name: task1
    dataset: imagenet1k
  - name: task2
    dataset: imagenetv2
  - name: task3
    dataset: imagenet-a
  - name: task4
    dataset: imagenet-r
  - name: task5
    dataset: imagenet-sketch
```

Then run:
```bash
python src/train.py --config-name train/continual_learning
```

### Option 2: Using Command Line (Hydra)

You can override the dataset per task, but this requires modifying the config structure. The current implementation reads from `cfg.tasks` if it exists.

### Option 3: Same Dataset, Different Splits

For true continual learning with the same dataset but different data splits, you would need to:

1. Modify the data module to support task-specific data splits
2. Create task-specific samplers or data subsets
3. Define how to split the data across tasks

## Typical Continual Learning Scenarios

### Scenario 1: Different Datasets (Domain Incremental)

Train on different datasets sequentially:

```yaml
tasks:
  - name: task1
    dataset: imagenet1k
  - name: task2
    dataset: imagenetv2
  - name: task3
    dataset: imagenet-a
```

### Scenario 2: Class Incremental

Train on different class subsets of the same dataset. This requires:
- Custom data loading with class filtering
- Task-specific class masks
- Modified data module

### Scenario 3: Same Dataset, Multiple Tasks

Use the same dataset for all tasks (current default):

```yaml
tasks:
  - name: task1
    dataset: imagenet1k
  - name: task2
    dataset: imagenet1k
  # ... etc
```

## Implementation Details

### How Tasks Are Processed

1. **Task Sequence**: If `cfg.tasks` is defined, it uses those datasets. Otherwise, uses `cfg.dataset.name` for all tasks.

2. **Checkpoint Loading**: After each task, the model checkpoint is saved and loaded for the next task.

3. **Data Module**: If tasks have different datasets, the data module is recreated for each task.

4. **Training**: Each task trains for `epochs_per_task` epochs.

### Current Limitations

1. **No Class Incremental Support**: The current implementation doesn't support splitting classes across tasks.

2. **No Data Splitting**: Can't split a single dataset into multiple tasks (e.g., 20% per task).

3. **Same Dataset Default**: If no task sequence is defined, all tasks use the same dataset.

## Recommended Task Sequences

### For Robustness Experiments

Use OOD datasets as tasks to test robustness across domains:

```yaml
tasks:
  - name: task1
    dataset: imagenet1k  # ID dataset
  - name: task2
    dataset: imagenetv2
  - name: task3
    dataset: imagenet-a
  - name: task4
    dataset: imagenet-r
  - name: task5
    dataset: imagenet-sketch
```

### For Standard Continual Learning

Use the same dataset with different splits (requires custom implementation):

```yaml
tasks:
  - name: task1
    dataset: imagenet1k
    split: "0-200"  # Classes 0-200 (requires custom implementation)
  - name: task2
    dataset: imagenet1k
    split: "200-400"  # Classes 200-400
  # etc.
```

## Example Usage

### Default (Same Dataset for All Tasks)

```bash
python src/train.py \
    train.continual_learning=true \
    train.num_tasks=5 \
    train.epochs_per_task=20
```

Result: 5 tasks, all using `imagenet1k`, 20 epochs each.

### Custom Task Sequence

```bash
# Use the continual_learning config which defines tasks
python src/train.py --config-name train/continual_learning
```

Or modify the config to define your task sequence.

## Summary

- **Default**: No explicit task sequence; all tasks use the same dataset (`imagenet1k`)
- **Custom**: Define `tasks` in config file with different datasets per task
- **Current**: Implementation supports different datasets per task, but not class splitting or data splitting
- **Future**: Could extend to support class-incremental or split-based continual learning

