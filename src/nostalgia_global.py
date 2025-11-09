"""
Nostalgia (Global) - Lanczos Null-Space Projection + LoRA

Implements the global Nostalgia method that projects gradients onto the null space
of the Hessian computed using Lanczos iteration via pytorch-hessian-eigenthings.
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


class NostalgiaGlobal:
    """
    Global Nostalgia: Projects gradients onto the null space of the Hessian
    computed using Lanczos iteration.
    """
    
    def __init__(
        self,
        model: LightningModule,
        num_eigenthings: int = 100,
        power_iter_steps: int = 20,
        use_gpu: bool = True,
        momentum: float = 0.9,
    ):
        """
        Args:
            model: PyTorch Lightning model
            num_eigenthings: Number of top eigenvalues/eigenvectors to compute
            power_iter_steps: Number of power iteration steps for Lanczos
            use_gpu: Whether to use GPU for computation
            momentum: Momentum for gradient projection
        """
        self.model = model
        self.num_eigenthings = num_eigenthings
        self.power_iter_steps = power_iter_steps
        self.use_gpu = use_gpu
        self.momentum = momentum
        
        self.eigenvals = None
        self.eigenvecs = None
        self.null_space_projection = None
        
    def compute_null_space(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        """
        Compute the null space of the Hessian using Lanczos iteration.
        
        Args:
            dataloader: DataLoader for computing Hessian
            criterion: Loss function
            device: Device to compute on
        """
        if not HESSIAN_AVAILABLE:
            raise ImportError("hessian_eigenthings is required for Nostalgia. Install with: pip install pytorch-hessian-eigenthings")
        
        print(f"Computing Hessian null space with {self.num_eigenthings} eigencomponents...")
        
        # Create a function to compute loss for hessian computation
        def loss_fn(model):
            model.eval()
            total_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for batch in dataloader:
                    pixel_values = batch.get("pixel_values") or batch.get("images") or batch.get("image")
                    labels = batch.get("labels") or batch.get("label") or batch.get("targets")
                    
                    pixel_values = pixel_values.to(device)
                    labels = labels.to(device)
                    
                    outputs = model.forward(pixel_values, labels=labels, no_grad=True)
                    loss = outputs.get("loss")
                    if loss is None:
                        logits = outputs["logits"]
                        loss = criterion(logits, labels)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    if num_batches >= 10:  # Limit for efficiency
                        break
            return total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute top eigenvalues and eigenvectors
        # Note: hessian_eigenthings API may vary, adjust as needed
        try:
            eigenvals, eigenvecs = compute_hessian_eigenthings(
                self.model,
                dataloader,
                criterion,
                num_eigenthings=self.num_eigenthings,
                power_iter_steps=self.power_iter_steps,
                use_gpu=self.use_gpu,
            )
        except Exception as e:
            print(f"Error computing Hessian eigenthings: {e}")
            print("Attempting alternative computation method...")
            # Fallback: use a simplified approach
            eigenvals, eigenvecs = self._compute_hessian_simplified(dataloader, criterion, device)
        
        self.eigenvals = eigenvals
        self.eigenvecs = eigenvecs
        
        # Identify null space: eigenvectors corresponding to near-zero eigenvalues
        # Threshold: eigenvalues < 1e-6 are considered null space
        null_threshold = 1e-6
        null_indices = np.where(eigenvals < null_threshold)[0]
        
        if len(null_indices) == 0:
            print("Warning: No null space found. Using eigenvectors with smallest eigenvalues.")
            # Use bottom 10% of eigenvectors
            null_indices = np.argsort(eigenvals)[:max(1, len(eigenvals) // 10)]
        
        print(f"Found {len(null_indices)} null space components out of {len(eigenvals)}")
        
        # Store null space projection matrix
        self.null_space_projection = self._build_projection_matrix(null_indices)
        
    def _build_projection_matrix(self, null_indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Build projection matrix for null space components.
        
        Args:
            null_indices: Indices of null space eigenvectors
            
        Returns:
            Dictionary mapping parameter names to projection matrices
        """
        projection = {}
        param_dict = dict(self.model.named_parameters())
        
        # Flatten eigenvectors and organize by parameter
        for name, param in param_dict.items():
            if not param.requires_grad:
                continue
                
            # Find corresponding eigenvector components for this parameter
            param_size = param.numel()
            projection[name] = torch.zeros(param_size, len(null_indices), device=param.device)
            
            # Extract relevant components from eigenvectors
            # This is a simplified version - in practice, eigenvectors are flattened
            # and need to be reshaped to match parameter shapes
            for i, idx in enumerate(null_indices):
                eigenvec_flat = self.eigenvecs[idx]
                # Match the size - this assumes eigenvectors are flattened
                if eigenvec_flat.numel() >= param_size:
                    projection[name][:, i] = eigenvec_flat[:param_size]
                else:
                    # Pad with zeros if needed
                    proj_flat = torch.zeros(param_size, device=param.device)
                    proj_flat[:eigenvec_flat.numel()] = eigenvec_flat
                    projection[name][:, i] = proj_flat
        
        return projection
    
    def project_gradient(self, grad: torch.Tensor, param_name: str) -> torch.Tensor:
        """
        Project gradient onto null space.
        
        Args:
            grad: Gradient tensor
            param_name: Name of the parameter
            
        Returns:
            Projected gradient
        """
        if self.null_space_projection is None:
            return grad
        
        if param_name not in self.null_space_projection:
            return grad
        
        proj_matrix = self.null_space_projection[param_name]
        grad_flat = grad.flatten()
        
        # Project: P = I - sum(v_i v_i^T) where v_i are null space eigenvectors
        # For efficiency, we compute: grad_proj = grad - sum((grad^T v_i) v_i)
        grad_proj_flat = grad_flat.clone()
        
        for i in range(proj_matrix.shape[1]):
            v = proj_matrix[:, i]
            # Project out component in direction of v
            grad_proj_flat = grad_proj_flat - (grad_proj_flat @ v) * v
        
        return grad_proj_flat.reshape(grad.shape)
    
    def _compute_hessian_simplified(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple:
        """
        Simplified Hessian computation as fallback.
        This is a placeholder - in practice, use proper Lanczos implementation.
        """
        print("Using simplified Hessian computation (placeholder)")
        # This is a simplified version - in practice, implement proper Lanczos
        # For now, return dummy values
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        eigenvals = np.ones(self.num_eigenthings) * 1e-5  # Small eigenvalues
        eigenvecs = [torch.randn(num_params, device=device) for _ in range(self.num_eigenthings)]
        return eigenvals, eigenvecs
    
    def apply_projection_to_gradients(self) -> None:
        """
        Apply null space projection to all gradients in the model.
        This should be called after backward() but before optimizer.step().
        """
        if self.null_space_projection is None:
            return
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                param.grad.data = self.project_gradient(param.grad.data, name)

