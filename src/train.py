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
from models.vit import ViTModel, ViTLoraConfig
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
    model_type = cfg.model.get("type", "resnet")  # Default to resnet for backwards compatibility
    
    lora_config = None
    if use_lora:
        lora_cfg = cfg.model.get("lora", {})
        if model_type == "resnet":
            lora_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                target_modules=lora_cfg.get("target_modules", ["layer4", "layer3"]),
                lora_dropout=lora_cfg.get("lora_dropout", 0.1),
                bias=lora_cfg.get("bias", "none"),
                task_type="IMAGE_CLASSIFICATION",
            )
        elif model_type == "vit":
            lora_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                target_modules=lora_cfg.get("target_modules", ["query", "key", "value", "dense"]),
                lora_dropout=lora_cfg.get("lora_dropout", 0.1),
                bias=lora_cfg.get("bias", "none"),
                task_type="IMAGE_CLASSIFICATION",
            )
    
    # Create model based on type
    if model_type == "resnet":
        model = ResNetModel(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
        )
    elif model_type == "vit":
        model = ViTModel(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


@hydra.main(version_base=None, config_path="../config", config_name="train/default")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seeds
    pl.seed_everything(cfg.get("seed", 42))
    
    # Determine epochs per task
    continual_learning = cfg.train.get("continual_learning", False)
    epochs_per_task = cfg.train.get("epochs_per_task", None)
    max_epochs = cfg.train.max_epochs
    
    if continual_learning:
        if epochs_per_task is not None:
            num_tasks = cfg.train.get("num_tasks", 1)
            max_epochs = num_tasks * epochs_per_task
            print(f"Continual learning mode: {num_tasks} tasks × {epochs_per_task} epochs = {max_epochs} total epochs")
        else:
            print("Warning: continual_learning=True but epochs_per_task not specified. Using max_epochs per task.")
            epochs_per_task = max_epochs
    else:
        # Single task training
        if epochs_per_task is None:
            epochs_per_task = max_epochs
        print(f"Single task training: {max_epochs} epochs")
    
    # Get task sequence if defined
    tasks_config = cfg.get("tasks", None)
    if tasks_config is not None and len(tasks_config) > 0:
        task_datasets = [task.get("dataset", cfg.dataset.name) for task in tasks_config]
        print(f"Task sequence defined: {len(task_datasets)} tasks")
        for i, task_dataset in enumerate(task_datasets):
            print(f"  Task {i+1}: {task_dataset}")
    else:
        # Use same dataset for all tasks
        default_dataset = cfg.dataset.name
        num_tasks = cfg.train.get("num_tasks", 1) if continual_learning else 1
        task_datasets = [default_dataset] * num_tasks
        print(f"No task sequence defined. Using dataset '{default_dataset}' for all {num_tasks} tasks.")
    
    # Create data module (will be recreated per task if tasks have different datasets)
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
        max_epochs=max_epochs,
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
    if continual_learning:
        # Continual learning: train on multiple tasks
        num_tasks = len(task_datasets) if task_datasets else cfg.train.get("num_tasks", 1)
        save_task_checkpoints = cfg.train.get("save_task_checkpoints", True)
        task_checkpoint_prefix = cfg.train.get("task_checkpoint_prefix", "task")
        
        print(f"\nStarting continual learning with {num_tasks} tasks...")
        print(f"Task sequence: {task_datasets}")
        
        # Store initial model state for loading between tasks
        previous_ckpt_path = None
        
        for task_id in range(num_tasks):
            print(f"\n{'='*80}")
            print(f"Training Task {task_id + 1}/{num_tasks}: {task_datasets[task_id]}")
            print(f"{'='*80}")
            
            # Create data module for current task if dataset is different
            current_dataset = task_datasets[task_id]
            if current_dataset != data_module.dataset_name:
                print(f"Switching to dataset: {current_dataset}")
                data_module = ImageNetDataModule(
                    dataset_name=current_dataset,
                    data_dir=cfg.dataset.get("data_dir", "~/data"),
                    batch_size=cfg.train.batch_size,
                    num_workers=cfg.train.get("num_workers", 4),
                    image_size=cfg.dataset.get("image_size", 224),
                    distributed=cfg.train.get("distributed", False),
                )
                data_module.setup("fit")
            
            # Load model state from previous task if available
            if previous_ckpt_path and os.path.exists(previous_ckpt_path):
                print(f"Loading model state from previous task: {previous_ckpt_path}")
                ckpt = torch.load(previous_ckpt_path, map_location="cpu")
                if isinstance(ckpt, dict):
                    if "state_dict" in ckpt:
                        model.load_state_dict(ckpt["state_dict"], strict=False)
                    elif "model_state_dict" in ckpt:
                        model.load_state_dict(ckpt["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
                print("Model state loaded successfully.")
            
            # Create a trainer for this task (trains for epochs_per_task epochs)
            task_trainer = pl.Trainer(
                max_epochs=epochs_per_task,
                accelerator=cfg.train.get("accelerator", "auto"),
                devices=cfg.train.get("devices", "auto"),
                precision=cfg.train.get("precision", "32"),
                logger=loggers,
                callbacks=callbacks,
                gradient_clip_val=cfg.train.get("gradient_clip_val", None),
                accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1),
                log_every_n_steps=cfg.train.get("log_every_n_steps", 10),
            )
            
            # Train on current task (always starts from epoch 0 for this task)
            task_trainer.fit(model, data_module)
            
            # Save task checkpoint
            if save_task_checkpoints:
                task_ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"{task_checkpoint_prefix}-{task_id + 1}-{method}-{current_dataset}-epoch={epochs_per_task}.ckpt"
                )
                task_trainer.save_checkpoint(task_ckpt_path)
                previous_ckpt_path = task_ckpt_path
                print(f"Saved task {task_id + 1} checkpoint: {task_ckpt_path}")
        
        print(f"\nContinual learning completed. Trained on {num_tasks} tasks.")
    else:
        # Single task training
        trainer.fit(model, data_module)
    
    # Test
    if cfg.train.get("run_test", True):
        trainer.test(model, data_module)
    
    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()

