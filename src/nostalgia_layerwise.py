"""
Layer-wise Nostalgia (LWP) - Per-layer null space projection

Implements layer-wise Nostalgia that computes null spaces independently for each layer.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pytorch_lightning import LightningModule
import numpy as np

try:
    from hessian_eigenthings import compute_hessian_eigenthings
    HESSIAN_AVAILABLE = True
except ImportError:
    print("Warning: hessian_eigenthings not available. Install with: pip install pytorch-hessian-eigenthings")
    HESSIAN_AVAILABLE = False


class NostalgiaLayerwise:
    """
    Layer-wise Nostalgia: Computes null space projection independently for each layer.
    """
    
    def __init__(
        self,
        model: LightningModule,
        num_eigenthings_per_layer: int = 50,
        power_iter_steps: int = 20,
        use_gpu: bool = True,
        layer_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: PyTorch Lightning model
            num_eigenthings_per_layer: Number of eigencomponents per layer
            power_iter_steps: Number of power iteration steps
            use_gpu: Whether to use GPU
            layer_names: List of layer names to apply projection to (None = all layers)
        """
        self.model = model
        self.num_eigenthings_per_layer = num_eigenthings_per_layer
        self.power_iter_steps = power_iter_steps
        self.use_gpu = use_gpu
        self.layer_names = layer_names
        
        self.layer_null_spaces = {}  # Dict[layer_name, projection_matrix]
        
    def compute_layer_null_spaces(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        """
        Compute null space for each layer independently.
        
        Args:
            dataloader: DataLoader for computing Hessians
            criterion: Loss function
            device: Device to compute on
        """
        param_dict = dict(self.model.named_parameters())
        
        # Determine which layers to process
        if self.layer_names is None:
            layers_to_process = [name for name, param in param_dict.items() 
                               if param.requires_grad]
        else:
            layers_to_process = [name for name in self.layer_names 
                               if name in param_dict and param_dict[name].requires_grad]
        
        print(f"Computing layer-wise null spaces for {len(layers_to_process)} layers...")
        
        for layer_name in layers_to_process:
            print(f"Processing layer: {layer_name}")
            
            # Create a model wrapper that only exposes this layer's parameters
            layer_model = self._create_layer_model(layer_name)
            
            # Compute Hessian for this layer only
            if not HESSIAN_AVAILABLE:
                print(f"  Skipping {layer_name}: hessian_eigenthings not available")
                continue
                
            try:
                eigenvals, eigenvecs = compute_hessian_eigenthings(
                    layer_model,
                    dataloader,
                    criterion,
                    num_eigenthings=self.num_eigenthings_per_layer,
                    power_iter_steps=self.power_iter_steps,
                    use_gpu=self.use_gpu,
                )
                
                # Identify null space components
                null_threshold = 1e-6
                null_indices = np.where(eigenvals < null_threshold)[0]
                
                if len(null_indices) == 0:
                    null_indices = np.argsort(eigenvals)[:max(1, len(eigenvals) // 10)]
                
                # Build projection matrix for this layer
                param = param_dict[layer_name]
                projection = self._build_layer_projection(
                    layer_name, param, eigenvecs, null_indices
                )
                
                self.layer_null_spaces[layer_name] = projection
                print(f"  Found {len(null_indices)} null space components")
                
            except Exception as e:
                print(f"  Warning: Failed to compute null space for {layer_name}: {e}")
                continue
        
        print(f"Computed null spaces for {len(self.layer_null_spaces)} layers")
    
    def _create_layer_model(self, layer_name: str) -> nn.Module:
        """
        Create a model wrapper that only exposes parameters for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Wrapper model with only the specified layer's parameters
        """
        class LayerModel(nn.Module):
            def __init__(self, base_model, layer_name):
                super().__init__()
                self.base_model = base_model
                self.layer_name = layer_name
                # Register only the target layer's parameters
                param_dict = dict(base_model.named_parameters())
                if layer_name in param_dict:
                    self.register_parameter(layer_name.replace('.', '_'), 
                                          param_dict[layer_name])
            
            def forward(self, *args, **kwargs):
                return self.base_model(*args, **kwargs)
        
        return LayerModel(self.model, layer_name)
    
    def _build_layer_projection(
        self,
        layer_name: str,
        param: torch.nn.Parameter,
        eigenvecs: List[torch.Tensor],
        null_indices: np.ndarray,
    ) -> torch.Tensor:
        """
        Build projection matrix for a single layer.
        
        Args:
            layer_name: Name of the layer
            param: Parameter tensor
            eigenvecs: List of eigenvectors
            null_indices: Indices of null space components
            
        Returns:
            Projection matrix for this layer
        """
        param_size = param.numel()
        projection = torch.zeros(param_size, len(null_indices), device=param.device)
        
        for i, idx in enumerate(null_indices):
            if idx < len(eigenvecs):
                eigenvec_flat = eigenvecs[idx].flatten()
                if eigenvec_flat.numel() >= param_size:
                    projection[:, i] = eigenvec_flat[:param_size]
                else:
                    proj_flat = torch.zeros(param_size, device=param.device)
                    proj_flat[:eigenvec_flat.numel()] = eigenvec_flat
                    projection[:, i] = proj_flat
        
        return projection
    
    def project_layer_gradient(self, grad: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Project gradient for a specific layer onto its null space.
        
        Args:
            grad: Gradient tensor
            layer_name: Name of the layer
            
        Returns:
            Projected gradient
        """
        if layer_name not in self.layer_null_spaces:
            return grad
        
        proj_matrix = self.layer_null_spaces[layer_name]
        grad_flat = grad.flatten()
        grad_proj_flat = grad_flat.clone()
        
        # Project out null space components
        for i in range(proj_matrix.shape[1]):
            v = proj_matrix[:, i]
            grad_proj_flat = grad_proj_flat - (grad_proj_flat @ v) * v
        
        return grad_proj_flat.reshape(grad.shape)
    
    def apply_projection_to_gradients(self) -> None:
        """
        Apply layer-wise null space projection to all gradients.
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                param.grad.data = self.project_layer_gradient(param.grad.data, name)

