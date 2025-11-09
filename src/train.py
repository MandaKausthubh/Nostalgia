"""
Training script with Hydra configuration and PyTorch Lightning.

Supports:
- Full fine-tuning
- LoRA fine-tuning
- Nostalgia (Global)
- Nostalgia (Layer-wise)
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import torch.nn as nn

# Import models and utilities
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models.resnet import ResNetModel, ResNetLoraConfig
from models.baseModel import BaseModel
from src.data_module import ImageNetDataModule
from src.nostalgia_global import NostalgiaGlobal
from src.nostalgia_layerwise import NostalgiaLayerwise
from peft import LoraConfig


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


def get_model(cfg: DictConfig) -> BaseModel:
    """Create model based on configuration."""
    model_name = cfg.model.name
    use_lora = cfg.model.get("use_lora", False)
    
    lora_config = None
    if use_lora:
        lora_cfg = cfg.model.get("lora", {})
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["layer4", "layer3"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.1),
            bias=lora_cfg.get("bias", "none"),
            task_type="IMAGE_CLASSIFICATION",
        )
    
    # For now, default to ResNetModel
    # Can be extended to support other models
    model = ResNetModel(
        model_name=model_name,
        use_lora=use_lora,
        lora_config=lora_config,
    )
    
    return model


@hydra.main(version_base=None, config_path="../config", config_name="train/default")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seeds
    pl.seed_everything(cfg.get("seed", 42))
    
    # Create data module
    data_module = ImageNetDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.get("data_dir", "~/data"),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.get("num_workers", 4),
        image_size=cfg.dataset.get("image_size", 224),
        distributed=cfg.train.get("distributed", False),
    )
    
    # Create model
    model = get_model(cfg)
    
    # Setup Nostalgia if enabled
    nostalgia_obj = None
    nostalgia_callback = None
    
    method = cfg.train.get("method", "full_ft")
    
    if method == "nostalgia_global":
        print("Using Nostalgia (Global)")
        nostalgia_obj = NostalgiaGlobal(
            model=model,
            num_eigenthings=cfg.train.get("num_eigenthings", 100),
            power_iter_steps=cfg.train.get("power_iter_steps", 20),
            use_gpu=torch.cuda.is_available(),
        )
        nostalgia_callback = NostalgiaCallback(
            nostalgia_obj,
            compute_null_space_every_n_epochs=cfg.train.get("recompute_null_space_every", 1)
        )
    
    elif method == "nostalgia_layerwise":
        print("Using Nostalgia (Layer-wise)")
        nostalgia_obj = NostalgiaLayerwise(
            model=model,
            num_eigenthings_per_layer=cfg.train.get("num_eigenthings_per_layer", 50),
            power_iter_steps=cfg.train.get("power_iter_steps", 20),
            use_gpu=torch.cuda.is_available(),
            layer_names=cfg.train.get("layer_names", None),
        )
        nostalgia_callback = NostalgiaCallback(
            nostalgia_obj,
            compute_null_space_every_n_epochs=cfg.train.get("recompute_null_space_every", 1)
        )
    
    # Setup optimizers - override configure_optimizers method
    def configure_optimizers_fn():
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.get("lr", 1e-4),
            weight_decay=cfg.train.get("weight_decay", 0.01),
        )
        
        # Setup learning rate scheduler
        if cfg.train.get("scheduler", None) and cfg.train.scheduler.get("type") != "null":
            scheduler_cfg = cfg.train.scheduler
            if scheduler_cfg.type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.train.max_epochs,
                    eta_min=scheduler_cfg.get("eta_min", 0),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    }
                }
            elif scheduler_cfg.type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_cfg.get("step_size", 30),
                    gamma=scheduler_cfg.get("gamma", 0.1),
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
    
    if cfg.logger.get("wandb", {}).get("enabled", False):
        wandb_cfg = cfg.logger.wandb
        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "nostalgia-experiments"),
            name=wandb_cfg.get("name", None),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)
    
    if cfg.logger.get("tensorboard", {}).get("enabled", True):
        tb_logger = TensorBoardLogger(
            save_dir=cfg.logger.tensorboard.get("save_dir", "logs"),
            name=cfg.logger.tensorboard.get("name", "default"),
        )
        loggers.append(tb_logger)
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_dir = cfg.train.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{method}-{{epoch:02d}}-{{val_acc:.4f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=cfg.train.get("save_top_k", 5),
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
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.get("accelerator", "auto"),
        devices=cfg.train.get("devices", "auto"),
        precision=cfg.train.get("precision", "32"),
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.get("gradient_clip_val", None),
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1),
        log_every_n_steps=cfg.train.get("log_every_n_steps", 10),
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    if cfg.train.get("run_test", True):
        trainer.test(model, data_module)
    
    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()

