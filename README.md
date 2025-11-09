# Nostalgia Experiments - ICML Submission

Implementation of Nostalgia experiments for continual learning and robustness evaluation, including:

- **Nostalgia (Global)**: Lanczos Null-Space Projection + LoRA
- **Layer-wise Nostalgia (LWP)**: Per-layer null space projection
- **Model-Soups baselines**: Uniform Soup and Greedy Soup (Wortsman et al., 2022)

## Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Install `pytorch-hessian-eigenthings` for Hessian computation:

```bash
pip install pytorch-hessian-eigenthings
```

3. (Optional) Clone Model-Soups repository for reference:

```bash
mkdir -p third_party
git clone https://github.com/mlfoundations/model-soups.git third_party/model-soups
```

## Dataset Setup

Ensure your datasets are organized as follows:

```
~/data/
├── imagenet/
│   ├── train/
│   └── val/
├── imagenetv2/
│   └── val/
├── imagenet-a/
│   └── val/
├── imagenet-r/
│   └── val/
├── imagenet-sketch/
│   └── val/
└── objectnet/
    └── val/
```

Update the `data_dir` in config files to point to your data location.

## Usage

### Training

**Note:** Run all commands from the project root directory. Hydra will change the working directory to `outputs/` by default.

Train a model using Hydra configuration:

```bash
# Full fine-tuning
python src/train.py train.method=full_ft

# LoRA fine-tuning
python src/train.py train.method=lora model.use_lora=true

# Nostalgia (Global)
python src/train.py train.method=nostalgia_global train.num_eigenthings=100

# Nostalgia (Layer-wise)
python src/train.py train.method=nostalgia_layerwise train.num_eigenthings_per_layer=50
```

Alternatively, use the experiment runner:

```bash
python run_experiment.py --method nostalgia_global --batch_size 64 --max_epochs 100
```

### Configuration

All configurations are managed via Hydra. Key config files:

- `config/train/default.yaml`: Training configuration
- `config/models/resnet50.yaml`: Model configuration
- `config/datasets/imagenet1k.yaml`: Dataset configuration
- `config/logger/tensorboard.yaml`: Logging configuration

Override any parameter from command line:

```bash
python src/train.py train.batch_size=64 train.max_epochs=50 train.lr=5e-5
```

### Evaluation

**Note:** When specifying checkpoint paths, use paths relative to the Hydra output directory, or absolute paths.

Evaluate a single checkpoint:

```bash
python src/eval.py eval.checkpoint_path=checkpoints/best_model.ckpt
```

Or with absolute path:

```bash
python src/eval.py eval.checkpoint_path=/absolute/path/to/checkpoint.ckpt
```

Evaluate a model soup:

```bash
python src/eval.py \
    eval.soup.enabled=true \
    eval.soup.type=uniform \
    eval.soup.checkpoint_paths=[checkpoints/ckpt1.ckpt,checkpoints/ckpt2.ckpt,checkpoints/ckpt3.ckpt]
```

For greedy soup:

```bash
python src/eval.py \
    eval.soup.enabled=true \
    eval.soup.type=greedy \
    eval.soup.checkpoint_paths=[checkpoints/ckpt1.ckpt,checkpoints/ckpt2.ckpt,checkpoints/ckpt3.ckpt]
```

### Model Soups

The implementation includes two soup methods:

1. **Uniform Soup**: Simple average of checkpoint weights
2. **Greedy Soup**: Iteratively adds checkpoints that improve validation performance

To build soups from training checkpoints:

1. Train models and save checkpoints (automatically done during training)
2. Collect checkpoint paths from `checkpoints/` directory
3. Use `eval.py` with `soup.enabled=true` to create and evaluate soups

**Recommended checkpoint collection policy:**
- Save best validation checkpoints per epoch (default: top-5)
- For continual learning: save end-of-task checkpoints
- Order checkpoints by validation performance for greedy soup

## Project Structure

```
.
├── config/              # Hydra configuration files
│   ├── train/          # Training configs
│   ├── eval/            # Evaluation configs
│   ├── models/          # Model configs
│   ├── datasets/        # Dataset configs
│   └── logger/          # Logging configs
├── src/                 # Source code
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   ├── data_module.py   # PyTorch Lightning DataModule
│   ├── nostalgia_global.py      # Global Nostalgia implementation
│   ├── nostalgia_layerwise.py   # Layer-wise Nostalgia implementation
│   └── model_soups_utils.py      # Model soups utilities
├── models/              # Model definitions
├── datasets/            # Dataset loaders
└── requirements.txt     # Python dependencies
```

## Logging

The framework supports logging to:

- **TensorBoard**: Enabled by default, logs to `logs/`
- **Weights & Biases**: Enable in config or via command line:

```bash
python src/train.py logger.wandb.enabled=true logger.wandb.project=my-project
```

## Experiments

### Baseline Methods

1. **Full Fine-tuning**: Standard end-to-end fine-tuning
2. **LoRA**: Low-Rank Adaptation fine-tuning
3. **Nostalgia (Global)**: Global null-space projection
4. **Nostalgia (Layer-wise)**: Per-layer null-space projection

### Model Soups

Build soups from checkpoints of any method:
- Uniform Soup from top-K checkpoints
- Greedy Soup using validation metric

### Evaluation Metrics

- **ID Accuracy**: ImageNet-1K validation accuracy
- **OOD Accuracy**: Average across OOD datasets
- **ID→OOD Gap**: Difference between ID and OOD performance
- **Forgetting Matrix**: For continual learning experiments

## Notes

- The Hessian computation uses `pytorch-hessian-eigenthings` library
- For large models, Hessian computation can be memory-intensive
- Adjust `num_eigenthings` and `num_eigenthings_per_layer` based on available resources
- Model soups require multiple checkpoints - ensure sufficient checkpoint saving during training

## Citation

If you use this code, please cite the original paper.

