"""
Evaluation script with argparse (alternative to Hydra).

This script provides an argparse-based interface for evaluating models.
"""
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from peft import LoraConfig

# Import models and utilities
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models.resnet import ResNetModel
from models.vit import ViTModel
from models.baseModel import BaseModel
from src.data_module import ImageNetDataModule
from src.model_soups_utils import uniform_soup, greedy_soup


def evaluate_model(
    model: BaseModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset_name: str = "unknown",
) -> dict:
    """Evaluate model on a dataset."""
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


def get_model(
    model_type: str,
    model_name: str,
    use_lora: bool = False,
    checkpoint_path: str = None,
    device: torch.device = None,
) -> BaseModel:
    """Load model from checkpoint or create new model."""
    if model_type == "resnet":
        model = ResNetModel(
            model_name=model_name,
            use_lora=use_lora,
        )
    elif model_type == "vit":
        model = ViTModel(
            model_name=model_name,
            use_lora=use_lora,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"], strict=False)
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models (argparse version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="resnet",
                       choices=["resnet", "vit"],
                       help="Model type")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (overrides default for model_type)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA (for loading LoRA models)")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--soup", action="store_true",
                       help="Evaluate model soup")
    parser.add_argument("--soup_type", type=str, default="uniform",
                       choices=["uniform", "greedy"],
                       help="Soup type (if --soup is enabled)")
    parser.add_argument("--soup_checkpoints", type=str, nargs="+",
                       help="List of checkpoint paths for soup (if --soup is enabled)")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="imagenet1k",
                       help="ID dataset name")
    parser.add_argument("--data_dir", type=str, default="~/data",
                       help="Data directory")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size")
    parser.add_argument("--ood_datasets", type=str, nargs="+",
                       default=["imagenetv2", "imagenet-a", "imagenet-r", "imagenet-sketch", "objectnet"],
                       help="OOD dataset names")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Set default model names
    if args.model_name is None:
        if args.model_type == "resnet":
            args.model_name = "microsoft/resnet-50"
        elif args.model_type == "vit":
            args.model_name = "google/vit-base-patch16-224"
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load dataloaders
    dataloaders = {}
    
    # ID dataset
    id_data_module = ImageNetDataModule(
        dataset_name=args.dataset,
        data_dir=os.path.expanduser(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    id_data_module.setup("test")
    dataloaders[args.dataset] = id_data_module.test_dataloader()
    
    # OOD datasets
    for ood_name in args.ood_datasets:
        try:
            ood_data_module = ImageNetDataModule(
                dataset_name=ood_name,
                data_dir=os.path.expanduser(args.data_dir),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                image_size=args.image_size,
            )
            ood_data_module.setup("test")
            dataloaders[ood_name] = ood_data_module.test_dataloader()
            print(f"Loaded OOD dataset: {ood_name}")
        except Exception as e:
            print(f"Warning: Could not load {ood_name}: {e}")
    
    # Evaluate soup or single checkpoint
    if args.soup and args.soup_checkpoints:
        print(f"Creating {args.soup_type} soup from {len(args.soup_checkpoints)} checkpoints...")
        
        def metric_fn(model, loader):
            metrics = evaluate_model(model, loader, device, "dev")
            return metrics["accuracy"]
        
        # Determine model class
        if args.model_type == "resnet":
            from models.resnet import ResNetModel
            model_class = ResNetModel
        elif args.model_type == "vit":
            from models.vit import ViTModel
            model_class = ViTModel
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # Model kwargs
        model_kwargs = {
            "model_name": args.model_name,
            "use_lora": args.use_lora,
        }
        
        # Create soup
        if args.soup_type == "uniform":
            soup_model = uniform_soup(
                args.soup_checkpoints,
                model_class,
                model_kwargs=model_kwargs,
                map_location=str(device),
                device=device,
            )
        else:  # greedy
            dev_loader = dataloaders.get(args.dataset)
            soup_model = greedy_soup(
                args.soup_checkpoints,
                model_class,
                model_kwargs=model_kwargs,
                dev_loader=dev_loader,
                metric_fn=metric_fn,
                map_location=str(device),
                device=device,
            )
        
        model = soup_model
    else:
        # Load single checkpoint
        model = get_model(
            args.model_type,
            args.model_name,
            args.use_lora,
            checkpoint_path=args.checkpoint_path,
            device=device,
        )
    
    # Evaluate on all datasets
    results = {}
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    for dataset_name, dataloader in dataloaders.items():
        print(f"\nEvaluating on {dataset_name}...")
        metrics = evaluate_model(model, dataloader, device, dataset_name)
        results[dataset_name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Loss: {metrics['loss']:.4f}")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for dataset_name, metrics in results.items():
        print(f"{dataset_name:20s} | Acc: {metrics['accuracy']:.4f} | Loss: {metrics['loss']:.4f}")
    
    # Compute average OOD accuracy
    ood_results = {k: v for k, v in results.items() if k != args.dataset}
    if ood_results:
        avg_ood_acc = np.mean([m["accuracy"] for m in ood_results.values()])
        id_acc = results.get(args.dataset, {}).get("accuracy", 0.0)
        gap = id_acc - avg_ood_acc
        print(f"\nAverage OOD Accuracy: {avg_ood_acc:.4f}")
        print(f"ID Accuracy: {id_acc:.4f}")
        print(f"ID→OOD Gap: {gap:.4f}")


if __name__ == "__main__":
    main()

