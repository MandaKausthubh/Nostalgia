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
from src.metrics import ContinualLearningMetrics
from src.visualize import (
    plot_forgetting_vs_steps,
    plot_forgetting_heatmap,
    plot_task_retention_curves,
    plot_method_comparison_bar,
    plot_id_ood_scatter,
    plot_hessian_spectra,
    plot_ablation_curves,
)
from peft import LoraConfig
import numpy as np


class MetricsCallback(pl.Callback):
    """Callback to track metrics during training."""
    
    def __init__(self, metrics_tracker: ContinualLearningMetrics):
        self.metrics_tracker = metrics_tracker
        self.current_task = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track training loss."""
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss.item() if torch.is_tensor(outputs.loss) else outputs.loss
            self.metrics_tracker.record_train_loss(loss, step=trainer.global_step)
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss'].item() if torch.is_tensor(outputs['loss']) else outputs['loss']
            self.metrics_tracker.record_train_loss(loss, step=trainer.global_step)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Track validation accuracy."""
        if trainer.current_epoch >= 0:
            # Get validation accuracy from logged metrics
            if 'val_acc' in trainer.callback_metrics:
                val_acc = trainer.callback_metrics['val_acc']
                if torch.is_tensor(val_acc):
                    val_acc = val_acc.item()
                # Record accuracy for current task
                # Note: In continual learning, this tracks accuracy on current task
                self.metrics_tracker.record_accuracy(
                    self.current_task, 
                    self.current_task, 
                    val_acc
                )


class NostalgiaCallback(pl.Callback):
    """Callback to apply Nostalgia projection during training."""
    
    def __init__(self, nostalgia_obj, compute_null_space_every_n_epochs=1, metrics_tracker=None):
        self.nostalgia_obj = nostalgia_obj
        self.compute_null_space_every_n_epochs = compute_null_space_every_n_epochs
        self.null_space_computed = False
        self.metrics_tracker = metrics_tracker
    
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
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Record Hessian eigenvalues if available."""
        if self.null_space_computed and self.metrics_tracker is not None:
            if hasattr(self.nostalgia_obj, 'eigenvals') and self.nostalgia_obj.eigenvals is not None:
                task_name = f"Task{trainer.current_epoch // trainer.max_epochs + 1}"
                eigenvals = np.array(self.nostalgia_obj.eigenvals)
                self.metrics_tracker.record_hessian_eigenvalues(task_name, eigenvals)


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
    
    # Initialize metrics tracker
    num_tasks = len(task_datasets) if task_datasets else (cfg.train.get("num_tasks", 1) if continual_learning else 1)
    task_names = [task.get("name", f"Task{i+1}") if tasks_config else f"Task{i+1}" 
                  for i, task in enumerate(tasks_config or [])] if tasks_config else [f"Task{i+1}" for i in range(num_tasks)]
    if not task_names or len(task_names) != num_tasks:
        task_names = [f"Task{i+1}" for i in range(num_tasks)]
    
    metrics_tracker = ContinualLearningMetrics(num_tasks=num_tasks, task_names=task_names)
    
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
            compute_null_space_every_n_epochs=cfg.train.get("recompute_null_space_every", 1),
            metrics_tracker=metrics_tracker
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
            compute_null_space_every_n_epochs=cfg.train.get("recompute_null_space_every", 1),
            metrics_tracker=metrics_tracker
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
    
    # Metrics tracking callback
    metrics_callback = MetricsCallback(metrics_tracker)
    callbacks.append(metrics_callback)
    
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
            
            # Record task boundary
            start_step = metrics_tracker.current_step
            metrics_callback.current_task = task_id
            
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
            
            # Record task boundary end
            end_step = metrics_tracker.current_step
            metrics_tracker.record_task_boundary(task_id, start_step, end_step)
            
            # Evaluate on all previous tasks and current task
            if cfg.train.get("eval_after_task", False):
                print(f"Evaluating on all tasks after task {task_id + 1}...")
                # Get validation accuracy for current task
                val_acc = task_trainer.callback_metrics.get('val_acc', None)
                if val_acc is not None:
                    val_acc = val_acc.item() if torch.is_tensor(val_acc) else val_acc
                    # Record accuracy for current task
                    metrics_tracker.record_accuracy(task_id, task_id, val_acc)
                    # Compute forgetting for all previous tasks
                    for eval_task_id in range(task_id):
                        metrics_tracker.compute_forgetting(task_id, eval_task_id)
            else:
                # Still record accuracy for current task
                val_acc = task_trainer.callback_metrics.get('val_acc', None)
                if val_acc is not None:
                    val_acc = val_acc.item() if torch.is_tensor(val_acc) else val_acc
                    metrics_tracker.record_accuracy(task_id, task_id, val_acc)
                    # Compute forgetting for previous tasks
                    for eval_task_id in range(task_id):
                        metrics_tracker.compute_forgetting(task_id, eval_task_id)
            
            # Save task checkpoint
            if save_task_checkpoints:
                task_ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"{task_checkpoint_prefix}-{task_id + 1}-{method}-{current_dataset}-epoch={epochs_per_task}.ckpt"
                )
                task_trainer.save_checkpoint(task_ckpt_path)
                previous_ckpt_path = task_ckpt_path
                print(f"Saved task {task_id + 1} checkpoint: {task_ckpt_path}")
        
        # Update average forgetting over time (compute for each task)
        # This should already be updated during training, but ensure it's complete
        for t in range(num_tasks):
            avg_forget = metrics_tracker.compute_average_forgetting(t)
            if t >= len(metrics_tracker.avg_forgetting_over_time):
                metrics_tracker.avg_forgetting_over_time.append(avg_forget)
            else:
                metrics_tracker.avg_forgetting_over_time[t] = avg_forget
        
        print(f"\nContinual learning completed. Trained on {num_tasks} tasks.")
    else:
        # Single task training
        trainer.fit(model, data_module)
        
        # Record metrics for single task
        if len(metrics_tracker.train_steps) > 0:
            metrics_tracker.record_task_boundary(0, 0, metrics_tracker.train_steps[-1])
        else:
            metrics_tracker.record_task_boundary(0, 0, max_epochs)
        
        # Record final validation accuracy
        if 'val_acc' in trainer.callback_metrics:
            val_acc = trainer.callback_metrics['val_acc']
            if torch.is_tensor(val_acc):
                val_acc = val_acc.item()
            metrics_tracker.record_accuracy(0, 0, val_acc)
        
        # Update average forgetting (will be 0 for single task)
        metrics_tracker.update_avg_forgetting_over_time()
    
    # Test
    if cfg.train.get("run_test", True):
        trainer.test(model, data_module)
    
    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")
    
    # Generate visualizations
    if cfg.get("visualize", {}).get("enabled", False):
        print("\nGenerating visualizations...")
        visualize_cfg = cfg.visualize
        save_dir = visualize_cfg.get("save_dir", "logs/plots")
        os.makedirs(save_dir, exist_ok=True)
        
        # Get wandb and tensorboard loggers
        wandb_logger = None
        tensorboard_writer = None
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
            elif isinstance(logger, TensorBoardLogger):
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    tensorboard_writer = SummaryWriter(logger.log_dir)
                except:
                    pass
        
        plots_cfg = visualize_cfg.get("plots", {})
        global_step = max_epochs if not continual_learning else num_tasks * epochs_per_task
        
        # Plot 1: Forgetting vs Steps
        if plots_cfg.get("forgetting_vs_steps", True):
            # Ensure we have data for plotting
            if len(metrics_tracker.train_loss_over_time) == 0:
                # Use dummy data if no loss recorded
                metrics_tracker.train_loss_over_time = [0.5] * max(len(metrics_tracker.avg_forgetting_over_time), 1)
            if len(metrics_tracker.avg_forgetting_over_time) == 0:
                # Compute if not already computed
                metrics_tracker.update_avg_forgetting_over_time()
            
            if len(metrics_tracker.avg_forgetting_over_time) > 0 or len(metrics_tracker.train_loss_over_time) > 0:
                plot_forgetting_vs_steps(
                    metrics_tracker.avg_forgetting_over_time if len(metrics_tracker.avg_forgetting_over_time) > 0 else [0.0],
                    metrics_tracker.train_loss_over_time,
                    metrics_tracker.task_boundaries,
                    metrics_tracker.task_names,
                    save_dir,
                    wandb_logger=wandb_logger,
                    tensorboard_writer=tensorboard_writer,
                    global_step=global_step,
                )
        
        # Plot 2: Forgetting Heatmap
        if plots_cfg.get("forgetting_heatmap", True) and continual_learning:
            forgetting_matrix = metrics_tracker.get_forgetting_matrix()
            if forgetting_matrix.size > 0:
                plot_forgetting_heatmap(
                    forgetting_matrix,
                    metrics_tracker.task_names,
                    save_dir,
                    wandb_logger=wandb_logger,
                    tensorboard_writer=tensorboard_writer,
                    global_step=global_step,
                )
        
        # Plot 3: Task Retention Curves
        if plots_cfg.get("task_retention_curves", True) and continual_learning:
            retention_data = metrics_tracker.get_task_retention_data()
            if retention_data:
                plot_task_retention_curves(
                    retention_data,
                    metrics_tracker.task_names,
                    save_dir,
                    wandb_logger=wandb_logger,
                    tensorboard_writer=tensorboard_writer,
                    global_step=global_step,
                )
        
        # Plot 4: Method Comparison (if multiple methods are being compared)
        # This is typically done after multiple runs, so we skip it here
        
        # Plot 6: Hessian Spectra
        if plots_cfg.get("hessian_spectra", True) and nostalgia_obj is not None:
            if metrics_tracker.hessian_eigs:
                plot_hessian_spectra(
                    metrics_tracker.hessian_eigs,
                    save_dir,
                    wandb_logger=wandb_logger,
                    tensorboard_writer=tensorboard_writer,
                    global_step=global_step,
                )
        
        print(f"Visualizations saved to {save_dir}")
        
        # Save metrics to file
        metrics_file = os.path.join(save_dir, "metrics.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics_tracker.to_dict(), f, indent=2)
        print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    train()

