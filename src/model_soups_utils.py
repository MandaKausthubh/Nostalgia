"""
Model Soups utilities: Uniform Soup and Greedy Soup implementations.

Based on Wortsman et al., 2022: "Model soups: averaging weights of multiple fine-tuned models"
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path
import copy
from tqdm import tqdm


def uniform_soup(
    checkpoint_paths: List[str],
    model_class: type,
    model_kwargs: Optional[Dict] = None,
    map_location: str = 'cpu',
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create a uniform soup by averaging weights from multiple checkpoints.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        map_location: Device to load checkpoints on
        device: Device to place final model on
        
    Returns:
        Model with averaged weights
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    if device is None:
        device = torch.device(map_location)
    
    print(f"Creating uniform soup from {len(checkpoint_paths)} checkpoints...")
    
    # Load all state dicts
    state_dicts = []
    for ckpt_path in tqdm(checkpoint_paths, desc="Loading checkpoints"):
        if not Path(ckpt_path).exists():
            print(f"Warning: Checkpoint {ckpt_path} not found, skipping...")
            continue
        
        ckpt = torch.load(ckpt_path, map_location=map_location)
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        
        state_dicts.append(state_dict)
    
    if len(state_dicts) == 0:
        raise ValueError("No valid checkpoints found")
    
    # Create model instance
    model = model_class(**model_kwargs)
    model_state_dict = model.state_dict()
    
    # Average state dicts
    averaged_state_dict = {}
    for key in model_state_dict.keys():
        if key not in state_dicts[0]:
            print(f"Warning: Key {key} not found in checkpoints, using model default")
            averaged_state_dict[key] = model_state_dict[key]
            continue
        
        # Check if shapes match
        shapes = [sd[key].shape for sd in state_dicts if key in sd]
        if not all(s == shapes[0] for s in shapes):
            print(f"Warning: Shape mismatch for {key}, using model default")
            averaged_state_dict[key] = model_state_dict[key]
            continue
        
        # Average tensors
        tensors = [sd[key].float() for sd in state_dicts if key in sd]
        averaged_state_dict[key] = torch.stack(tensors).mean(dim=0).to(
            model_state_dict[key].dtype
        )
    
    # Load averaged state dict
    model.load_state_dict(averaged_state_dict, strict=False)
    model = model.to(device)
    
    print(f"Uniform soup created successfully with {len(state_dicts)} checkpoints")
    return model


def greedy_soup(
    checkpoint_paths: List[str],
    model_class: type,
    model_kwargs: Optional[Dict] = None,
    dev_loader: Optional[torch.utils.data.DataLoader] = None,
    metric_fn: Optional[Callable] = None,
    map_location: str = 'cpu',
    device: Optional[torch.device] = None,
    initial_checkpoint_idx: Optional[int] = None,
) -> nn.Module:
    """
    Create a greedy soup by iteratively adding checkpoints that improve dev performance.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files (should be ordered by performance)
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        dev_loader: DataLoader for validation set
        metric_fn: Function that takes (model, dataloader) and returns metric (higher is better)
        map_location: Device to load checkpoints on
        device: Device to place final model on
        initial_checkpoint_idx: Index of checkpoint to start with (None = best performing)
        
    Returns:
        Model with greedily averaged weights
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    if device is None:
        device = torch.device(map_location)
    
    if dev_loader is None or metric_fn is None:
        print("Warning: No dev_loader or metric_fn provided, falling back to uniform soup")
        return uniform_soup(checkpoint_paths, model_class, model_kwargs, map_location, device)
    
    print(f"Creating greedy soup from {len(checkpoint_paths)} checkpoints...")
    
    # Load all state dicts
    state_dicts = []
    for ckpt_path in checkpoint_paths:
        if not Path(ckpt_path).exists():
            continue
        
        ckpt = torch.load(ckpt_path, map_location=map_location)
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        
        state_dicts.append(state_dict)
    
    if len(state_dicts) == 0:
        raise ValueError("No valid checkpoints found")
    
    # Initialize with best single checkpoint
    if initial_checkpoint_idx is None:
        print("Evaluating individual checkpoints to find best starting point...")
        best_metric = float('-inf')
        best_idx = 0
        
        for i, state_dict in enumerate(tqdm(state_dicts, desc="Evaluating checkpoints")):
            model = model_class(**model_kwargs)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            model.eval()
            
            metric = metric_fn(model, dev_loader)
            if metric > best_metric:
                best_metric = metric
                best_idx = i
        
        print(f"Best single checkpoint: {best_idx} with metric {best_metric:.4f}")
        initial_checkpoint_idx = best_idx
    
    # Start with best checkpoint
    soup_state_dict = copy.deepcopy(state_dicts[initial_checkpoint_idx])
    soup_model = model_class(**model_kwargs)
    soup_model.load_state_dict(soup_state_dict, strict=False)
    soup_model = soup_model.to(device)
    soup_model.eval()
    
    best_metric = metric_fn(soup_model, dev_loader)
    soup_checkpoints = [initial_checkpoint_idx]
    
    print(f"Initial soup metric: {best_metric:.4f}")
    
    # Greedily add checkpoints
    for i, state_dict in enumerate(tqdm(state_dicts, desc="Greedy selection")):
        if i == initial_checkpoint_idx:
            continue
        
        # Try adding this checkpoint
        candidate_state_dict = {}
        for key in soup_state_dict.keys():
            if key in state_dict:
                # Average with existing soup (equal weight)
                candidate_state_dict[key] = (
                    soup_state_dict[key].float() + state_dict[key].float()
                ) / 2.0
        
        # Evaluate candidate
        candidate_model = model_class(**model_kwargs)
        candidate_model.load_state_dict(candidate_state_dict, strict=False)
        candidate_model = candidate_model.to(device)
        candidate_model.eval()
        
        candidate_metric = metric_fn(candidate_model, dev_loader)
        
        # If improves, add to soup
        if candidate_metric > best_metric:
            print(f"  Adding checkpoint {i}: metric {candidate_metric:.4f} > {best_metric:.4f}")
            soup_state_dict = candidate_state_dict
            soup_model = candidate_model
            best_metric = candidate_metric
            soup_checkpoints.append(i)
        else:
            print(f"  Skipping checkpoint {i}: metric {candidate_metric:.4f} <= {best_metric:.4f}")
    
    print(f"Greedy soup created with {len(soup_checkpoints)} checkpoints")
    print(f"Final metric: {best_metric:.4f}")
    
    return soup_model


def load_checkpoint_state_dict(checkpoint_path: str, map_location: str = 'cpu') -> Dict:
    """
    Load state dict from checkpoint file, handling different formats.
    
    Args:
        checkpoint_path: Path to checkpoint
        map_location: Device to load on
        
    Returns:
        State dict
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
        else:
            return ckpt
    else:
        return ckpt

