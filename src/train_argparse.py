"""
Training script with argparse (alternative to Hydra).

This script provides an argparse-based interface for training models.
"""
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import torch.nn as nn
from peft import LoraConfig

# Import models and utilities
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models.resnet import ResNetModel, ResNetLoraConfig
from models.vit import ViTModel, ViTLoraConfig
from models.baseModel import BaseModel
from src.data_module import ImageNetDataModule
from src.nostalgia_global import NostalgiaGlobal
from src.nostalgia_layerwise import NostalgiaLayerwise


class NostalgiaCallback(pl.Callback):
    """Callback to apply Nostalgia projection during training."""
    
    def __init__(self, nostalgia_obj, compute_null_space_every_n_epochs=1):
        self.nostalgia_obj = nostalgia_obj
        self.compute_null_space_every_n_epochs = compute_null_space_every_n_epochs
        self.null_space_computed = False
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Compute null space at the start of training epochs."""
        if not self.null_space_computed or trainer.current_epoch % self.compute_null_space_every_n_epochs == 0:
            if self.null_space_computed:
                print("Recomputing null space...")
            
            # Get dataloader from trainer
            dataloader = trainer.train_dataloader.loaders if hasattr(trainer.train_dataloader, 'loaders') else trainer.train_dataloader
            if callable(dataloader):
                dataloader = dataloader()
            
            device = next(pl_module.parameters()).device
            
            if hasattr(self.nostalgia_obj, 'compute_null_space'):
                self.nostalgia_obj.compute_null_space(
                    dataloader, pl_module.criterion, device
                )
            elif hasattr(self.nostalgia_obj, 'compute_layer_null_spaces'):
                self.nostalgia_obj.compute_layer_null_spaces(
                    dataloader, pl_module.criterion, device
                )
            
            self.null_space_computed = True
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Apply gradient projection before optimizer step."""
        if self.null_space_computed:
            self.nostalgia_obj.apply_projection_to_gradients()


def get_model(
    model_type: str,
    model_name: str,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> BaseModel:
    """Create model based on arguments."""
    lora_config = None
    
    if use_lora:
        if model_type == "resnet":
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["layer4", "layer3"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="IMAGE_CLASSIFICATION",
            )
        elif model_type == "vit":
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "key", "value", "dense"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="IMAGE_CLASSIFICATION",
            )
    
    if model_type == "resnet":
        return ResNetModel(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
        )
    elif model_type == "vit":
        return ViTModel(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Train models with Nostalgia experiments (argparse version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="resnet",
                       choices=["resnet", "vit"],
                       help="Model type")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (overrides default for model_type)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Training method
    parser.add_argument("--method", type=str, default="full_ft",
                       choices=["full_ft", "lora", "nostalgia_global", "nostalgia_layerwise"],
                       help="Training method")
    parser.add_argument("--num_eigenthings", type=int, default=100,
                       help="Number of eigencomponents for nostalgia_global")
    parser.add_argument("--num_eigenthings_per_layer", type=int, default=50,
                       help="Number of eigencomponents per layer for nostalgia_layerwise")
    parser.add_argument("--power_iter_steps", type=int, default=20,
                       help="Power iteration steps for Lanczos")
    parser.add_argument("--recompute_null_space_every", type=int, default=1,
                       help="Recompute null space every N epochs")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="imagenet1k",
                       help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="~/data",
                       help="Data directory")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                       help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                       help="Accumulate gradients over N batches")
    
    # Scheduler arguments
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "step", "none"],
                       help="Learning rate scheduler type")
    parser.add_argument("--scheduler_eta_min", type=float, default=0.0,
                       help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--scheduler_step_size", type=int, default=30,
                       help="Step size for step scheduler")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1,
                       help="Gamma for step scheduler")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--save_top_k", type=int, default=5,
                       help="Save top K checkpoints")
    parser.add_argument("--run_test", action="store_true",
                       help="Run test after training")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="TensorBoard log directory")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="Weights & Biases project name (None = disabled)")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="Weights & Biases run name")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--accelerator", type=str, default="auto",
                       help="Accelerator (auto, gpu, cpu)")
    parser.add_argument("--devices", type=str, default="auto",
                       help="Devices (auto, or specific device IDs)")
    parser.add_argument("--precision", type=str, default="32",
                       choices=["16", "32", "bf16"],
                       help="Training precision")
    
    args = parser.parse_args()
    
    # Set default model names
    if args.model_name is None:
        if args.model_type == "resnet":
            args.model_name = "microsoft/resnet-50"
        elif args.model_type == "vit":
            args.model_name = "google/vit-base-patch16-224"
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Create data module
    data_module = ImageNetDataModule(
        dataset_name=args.dataset,
        data_dir=os.path.expanduser(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        distributed=False,
    )
    
    # Create model
    use_lora = args.use_lora or args.method == "lora"
    model = get_model(
        model_type=args.model_type,
        model_name=args.model_name,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Setup Nostalgia if enabled
    nostalgia_obj = None
    nostalgia_callback = None
    
    if args.method == "nostalgia_global":
        print("Using Nostalgia (Global)")
        nostalgia_obj = NostalgiaGlobal(
            model=model,
            num_eigenthings=args.num_eigenthings,
            power_iter_steps=args.power_iter_steps,
            use_gpu=torch.cuda.is_available(),
        )
        nostalgia_callback = NostalgiaCallback(
            nostalgia_obj,
            compute_null_space_every_n_epochs=args.recompute_null_space_every
        )
    
    elif args.method == "nostalgia_layerwise":
        print("Using Nostalgia (Layer-wise)")
        nostalgia_obj = NostalgiaLayerwise(
            model=model,
            num_eigenthings_per_layer=args.num_eigenthings_per_layer,
            power_iter_steps=args.power_iter_steps,
            use_gpu=torch.cuda.is_available(),
            layer_names=None,
        )
        nostalgia_callback = NostalgiaCallback(
            nostalgia_obj,
            compute_null_space_every_n_epochs=args.recompute_null_space_every
        )
    
    # Setup optimizers
    def configure_optimizers_fn():
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.max_epochs,
                eta_min=args.scheduler_eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        elif args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.scheduler_step_size,
                gamma=args.scheduler_gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        
        return optimizer
    
    model.configure_optimizers = configure_optimizers_fn
    
    # Setup loggers
    loggers = []
    
    if args.wandb_project:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
        )
        loggers.append(wandb_logger)
    
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f"{args.model_type}_{args.method}",
    )
    loggers.append(tb_logger)
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{args.model_type}-{args.method}-{{epoch:02d}}-{{val_acc:.4f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Add Nostalgia callback if enabled
    if nostalgia_callback:
        callbacks.append(nostalgia_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
    )
    
    # Train
    print(f"Training {args.model_type} model with method: {args.method}")
    trainer.fit(model, data_module)
    
    # Test
    if args.run_test:
        trainer.test(model, data_module)
    
    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()

