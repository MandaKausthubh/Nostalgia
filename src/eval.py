"""
Evaluation script for ID and OOD datasets.

Evaluates models on ImageNet-1K (ID) and various OOD datasets:
- ImageNet-V2
- ImageNet-A
- ImageNet-R
- ImageNet-Sketch
- ObjectNet
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Import models and utilities
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models.resnet import ResNetModel
from models.vit import ViTModel
from models.baseModel import BaseModel
from src.data_module import ImageNetDataModule
from src.model_soups_utils import uniform_soup, greedy_soup, load_checkpoint_state_dict
from src.visualize import plot_id_ood_scatter, plot_method_comparison_bar


def evaluate_model(
    model: BaseModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset_name: str = "unknown",
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary of metrics (accuracy, loss, etc.)
    """
    model.eval()
    model = model.to(device)
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            pixel_values = batch.get("pixel_values") or batch.get("images") or batch.get("image")
            labels = batch.get("labels") or batch.get("label") or batch.get("targets")
            
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model.forward(pixel_values, labels=labels, no_grad=True)
            logits = outputs["logits"]
            
            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item() * pixel_values.size(0)
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += pixel_values.size(0)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "total_samples": total_samples,
    }


def evaluate_soup(
    checkpoint_paths: list,
    model_class: type,
    model_kwargs: dict,
    dataloaders: dict,
    device: torch.device,
    soup_type: str = "uniform",
    dev_loader: torch.utils.data.DataLoader = None,
    metric_fn: callable = None,
) -> dict:
    """
    Evaluate a model soup on multiple datasets.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        model_class: Model class
        model_kwargs: Model initialization kwargs
        dataloaders: Dict of {dataset_name: dataloader}
        device: Device to run on
        soup_type: "uniform" or "greedy"
        dev_loader: Validation loader for greedy soup
        metric_fn: Metric function for greedy soup
        
    Returns:
        Dictionary of results per dataset
    """
    print(f"Creating {soup_type} soup from {len(checkpoint_paths)} checkpoints...")
    
    if soup_type == "uniform":
        soup_model = uniform_soup(
            checkpoint_paths,
            model_class,
            model_kwargs,
            map_location=str(device),
            device=device,
        )
    elif soup_type == "greedy":
        soup_model = greedy_soup(
            checkpoint_paths,
            model_class,
            model_kwargs,
            dev_loader=dev_loader,
            metric_fn=metric_fn,
            map_location=str(device),
            device=device,
        )
    else:
        raise ValueError(f"Unknown soup type: {soup_type}")
    
    # Evaluate on all datasets
    results = {}
    for dataset_name, dataloader in dataloaders.items():
        print(f"\nEvaluating on {dataset_name}...")
        metrics = evaluate_model(soup_model, dataloader, device, dataset_name)
        results[dataset_name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Loss: {metrics['loss']:.4f}")
    
    return results


@hydra.main(version_base=None, config_path="../config", config_name="eval/default")
def eval(cfg: DictConfig) -> None:
    """Main evaluation function."""
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint_path = cfg.eval.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Determine if evaluating a soup
    if cfg.eval.get("soup", {}).get("enabled", False):
        soup_cfg = cfg.eval.soup
        checkpoint_paths = soup_cfg.checkpoint_paths
        
        # Determine model class
        model_type = cfg.model.get("type", "resnet")
        if model_type == "resnet":
            model_class = ResNetModel
        elif model_type == "vit":
            model_class = ViTModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create model
        model_kwargs = {
            "model_name": cfg.model.name,
            "use_lora": cfg.model.get("use_lora", False),
        }
        
        # Load dataloaders for evaluation
        dataloaders = {}
        ood_datasets = cfg.eval.get("ood_datasets", [
            "imagenetv2", "imagenet-a", "imagenet-r", 
            "imagenet-sketch", "objectnet"
        ])
        
        # ID dataset
        id_data_module = ImageNetDataModule(
            dataset_name="imagenet1k",
            data_dir=cfg.dataset.get("data_dir", "~/data"),
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.get("num_workers", 4),
        )
        id_data_module.setup("test")
        dataloaders["imagenet1k"] = id_data_module.test_dataloader()
        
        # OOD datasets
        for ood_name in ood_datasets:
            try:
                ood_data_module = ImageNetDataModule(
                    dataset_name=ood_name,
                    data_dir=cfg.dataset.get("data_dir", "~/data"),
                    batch_size=cfg.eval.batch_size,
                    num_workers=cfg.eval.get("num_workers", 4),
                )
                ood_data_module.setup("test")
                dataloaders[ood_name] = ood_data_module.test_dataloader()
            except Exception as e:
                print(f"Warning: Could not load {ood_name}: {e}")
        
        # Metric function for greedy soup
        def metric_fn(model, loader):
            metrics = evaluate_model(model, loader, device, "dev")
            return metrics["accuracy"]
        
        # Evaluate soup
        dev_loader = dataloaders.get("imagenet1k")  # Use ID as dev for greedy
        results = evaluate_soup(
            checkpoint_paths,
            model_class,
            model_kwargs,
            dataloaders,
            device,
            soup_type=soup_cfg.get("type", "uniform"),
            dev_loader=dev_loader,
            metric_fn=metric_fn,
        )
    
    else:
        # Evaluate single checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Determine model class
        model_type = cfg.model.get("type", "resnet")
        if model_type == "resnet":
            model = ResNetModel(
                model_name=cfg.model.name,
                use_lora=cfg.model.get("use_lora", False),
            )
        elif model_type == "vit":
            model = ViTModel(
                model_name=cfg.model.name,
                use_lora=cfg.model.get("use_lora", False),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"], strict=False)
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        
        model = model.to(device)
        
        # Load dataloaders
        dataloaders = {}
        
        # ID dataset
        id_data_module = ImageNetDataModule(
            dataset_name="imagenet1k",
            data_dir=cfg.dataset.get("data_dir", "~/data"),
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.get("num_workers", 4),
        )
        id_data_module.setup("test")
        dataloaders["imagenet1k"] = id_data_module.test_dataloader()
        
        # OOD datasets
        ood_datasets = cfg.eval.get("ood_datasets", [
            "imagenetv2", "imagenet-a", "imagenet-r", 
            "imagenet-sketch", "objectnet"
        ])
        
        for ood_name in ood_datasets:
            try:
                ood_data_module = ImageNetDataModule(
                    dataset_name=ood_name,
                    data_dir=cfg.dataset.get("data_dir", "~/data"),
                    batch_size=cfg.eval.batch_size,
                    num_workers=cfg.eval.get("num_workers", 4),
                )
                ood_data_module.setup("test")
                dataloaders[ood_name] = ood_data_module.test_dataloader()
            except Exception as e:
                print(f"Warning: Could not load {ood_name}: {e}")
        
        # Evaluate on all datasets
        results = {}
        for dataset_name, dataloader in dataloaders.items():
            print(f"\nEvaluating on {dataset_name}...")
            metrics = evaluate_model(model, dataloader, device, dataset_name)
            results[dataset_name] = metrics
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Loss: {metrics['loss']:.4f}")
    
    # Print summary
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    for dataset_name, metrics in results.items():
        print(f"{dataset_name:20s} | Acc: {metrics['accuracy']:.4f} | Loss: {metrics['loss']:.4f}")
    
    # Compute average OOD accuracy
    ood_results = {k: v for k, v in results.items() if k != "imagenet1k"}
    if ood_results:
        avg_ood_acc = np.mean([m["accuracy"] for m in ood_results.values()])
        id_acc = results.get("imagenet1k", {}).get("accuracy", 0.0)
        gap = id_acc - avg_ood_acc
        print(f"\nAverage OOD Accuracy: {avg_ood_acc:.4f}")
        print(f"ID Accuracy: {id_acc:.4f}")
        print(f"ID→OOD Gap: {gap:.4f}")
    
    # Generate visualizations if enabled
    if cfg.get("visualize", {}).get("enabled", False):
        print("\nGenerating evaluation visualizations...")
        visualize_cfg = cfg.visualize
        save_dir = visualize_cfg.get("save_dir", "logs/plots")
        os.makedirs(save_dir, exist_ok=True)
        
        plots_cfg = visualize_cfg.get("plots", {})
        
        # Plot ID vs OOD scatter
        if plots_cfg.get("id_ood_scatter", True):
            # Prepare eval_metrics dict
            eval_metrics = {}
            id_acc = results.get("imagenet1k", {}).get("accuracy", 0.0) * 100  # Convert to percentage
            avg_ood_acc = np.mean([m["accuracy"] for m in ood_results.values()]) * 100 if ood_results else 0.0
            
            # Determine method name from checkpoint or config
            method_name = cfg.eval.get("method_name", "Current Method")
            if cfg.eval.get("soup", {}).get("enabled", False):
                soup_type = cfg.eval.soup.get("type", "uniform")
                method_name = f"ModelSoup-{soup_type.capitalize()}"
            else:
                method_name = cfg.model.get("type", "resnet").upper()
            
            eval_metrics[method_name] = (id_acc, avg_ood_acc)
            
            plot_id_ood_scatter(
                eval_metrics,
                save_dir,
                wandb_logger=None,  # Can be added if wandb is used in eval
                tensorboard_writer=None,
                global_step=0,
            )
        
        print(f"Evaluation visualizations saved to {save_dir}")


if __name__ == "__main__":
    eval()

