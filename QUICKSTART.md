# Quick Start Guide - ViT and Argparse Implementation

## ViT Model Implementation

The ViT (Vision Transformer) model is now available at `models/vit.py`. It supports:
- Full fine-tuning
- LoRA fine-tuning (targets attention and MLP layers)
- Nostalgia methods (Global and Layer-wise)
- Model soups

## Usage Examples

### Training with argparse (ViT)

```bash
# ViT with full fine-tuning
python src/train_argparse.py \
    --model_type vit \
    --method full_ft \
    --batch_size 32 \
    --max_epochs 100 \
    --lr 1e-4

# ViT with LoRA
python src/train_argparse.py \
    --model_type vit \
    --method lora \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32

# ViT with Nostalgia (Global)
python src/train_argparse.py \
    --model_type vit \
    --method nostalgia_global \
    --num_eigenthings 100 \
    --batch_size 32 \
    --max_epochs 100

# ViT with Nostalgia (Layer-wise)
python src/train_argparse.py \
    --model_type vit \
    --method nostalgia_layerwise \
    --num_eigenthings_per_layer 50 \
    --batch_size 32 \
    --max_epochs 100

# With W&B logging
python src/train_argparse.py \
    --model_type vit \
    --method nostalgia_global \
    --wandb_project my-project \
    --wandb_name vit-nostalgia-exp1
```

### Evaluation with argparse (ViT)

```bash
# Evaluate single checkpoint
python src/eval_argparse.py \
    --model_type vit \
    --checkpoint_path checkpoints/vit-best_model.ckpt \
    --batch_size 32

# Evaluate uniform soup
python src/eval_argparse.py \
    --model_type vit \
    --soup \
    --soup_type uniform \
    --soup_checkpoints \
        checkpoints/vit-ckpt1.ckpt \
        checkpoints/vit-ckpt2.ckpt \
        checkpoints/vit-ckpt3.ckpt

# Evaluate greedy soup
python src/eval_argparse.py \
    --model_type vit \
    --soup \
    --soup_type greedy \
    --soup_checkpoints \
        checkpoints/vit-ckpt1.ckpt \
        checkpoints/vit-ckpt2.ckpt \
        checkpoints/vit-ckpt3.ckpt \
    --dataset imagenet1k
```

### Training with Hydra (ViT)

```bash
# ViT with full fine-tuning
python src/train.py train.method=full_ft model=vit

# ViT with Nostalgia
python src/train.py train.method=nostalgia_global model=vit train.num_eigenthings=100
```

### Evaluation with Hydra (ViT)

```bash
# Evaluate single checkpoint
python src/eval.py eval.checkpoint_path=checkpoints/vit-best.ckpt model=vit

# Evaluate soup
python src/eval.py \
    eval.soup.enabled=true \
    eval.soup.type=uniform \
    eval.soup.checkpoint_paths=[checkpoints/vit-ckpt1.ckpt,checkpoints/vit-ckpt2.ckpt] \
    model=vit
```

## ResNet vs ViT

Both models work the same way, just specify `--model_type resnet` or `--model_type vit`:

```bash
# ResNet
python src/train_argparse.py --model_type resnet --method full_ft

# ViT
python src/train_argparse.py --model_type vit --method full_ft
```

## All Available Arguments

### Training (train_argparse.py)

```bash
python src/train_argparse.py --help
```

Key arguments:
- `--model_type`: resnet or vit
- `--method`: full_ft, lora, nostalgia_global, nostalgia_layerwise
- `--batch_size`: Batch size (default: 32)
- `--max_epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--use_lora`: Enable LoRA
- `--num_eigenthings`: For nostalgia_global (default: 100)
- `--num_eigenthings_per_layer`: For nostalgia_layerwise (default: 50)
- `--wandb_project`: W&B project name
- `--wandb_name`: W&B run name
- `--checkpoint_dir`: Checkpoint directory (default: checkpoints)
- `--data_dir`: Data directory (default: ~/data)
- `--dataset`: Dataset name (default: imagenet1k)

### Evaluation (eval_argparse.py)

```bash
python src/eval_argparse.py --help
```

Key arguments:
- `--model_type`: resnet or vit
- `--checkpoint_path`: Path to checkpoint
- `--soup`: Enable model soup evaluation
- `--soup_type`: uniform or greedy
- `--soup_checkpoints`: List of checkpoint paths
- `--batch_size`: Batch size (default: 32)
- `--ood_datasets`: OOD dataset names (default: imagenetv2, imagenet-a, imagenet-r, imagenet-sketch, objectnet)

## Notes

1. **ViT LoRA**: Targets `["query", "key", "value", "dense"]` modules (attention and MLP layers)
2. **ResNet LoRA**: Targets `["layer4", "layer3"]` modules
3. **Model Soups**: Work with both ResNet and ViT
4. **Nostalgia**: Works with both architectures

## Example Workflow

```bash
# 1. Train ViT with Nostalgia
python src/train_argparse.py \
    --model_type vit \
    --method nostalgia_global \
    --num_eigenthings 100 \
    --batch_size 32 \
    --max_epochs 100 \
    --wandb_project vit-nostalgia \
    --checkpoint_dir checkpoints/vit_nostalgia

# 2. Evaluate best checkpoint
python src/eval_argparse.py \
    --model_type vit \
    --checkpoint_path checkpoints/vit_nostalgia/vit-nostalgia_global-epoch=99-val_acc=0.8500.ckpt \
    --batch_size 32

# 3. Create and evaluate uniform soup from top-3 checkpoints
python src/eval_argparse.py \
    --model_type vit \
    --soup \
    --soup_type uniform \
    --soup_checkpoints \
        checkpoints/vit_nostalgia/vit-nostalgia_global-epoch=99-val_acc=0.8500.ckpt \
        checkpoints/vit_nostalgia/vit-nostalgia_global-epoch=98-val_acc=0.8480.ckpt \
        checkpoints/vit_nostalgia/vit-nostalgia_global-epoch=97-val_acc=0.8470.ckpt
```

