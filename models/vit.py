import torch
from torch import Tensor
from typing import Dict, Optional, Callable, List, Tuple, cast
from .baseModel import BaseModel 

try:
    from hessian_eigenthings import compute_hessian_eigenthings
except Exception as e:
    compute_hessian_eigenthings = None
    _hessian_import_error = e


# -----------------------
# Utility helpers
# -----------------------
def flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors], dim=0)

def unflatten_to(flat: torch.Tensor, templates: List[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    idx = 0
    for t in templates:
        n = t.numel()
        out.append(flat[idx: idx + n].view_as(t))
        idx += n
    return out

def build_projection_from_U(U: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    U: (n_params, r) orthonormal columns (torch.Tensor)
    returns proj_fn(g_flat) -> g_proj_flat
    """
    # ensure U is float32/float64 same as grads
    def proj_fn(g_flat: torch.Tensor) -> torch.Tensor:
        # coeff = U^T g
        coeff = U.T @ g_flat
        return g_flat - (U @ coeff)
    return proj_fn


# -----------------------
# ViTModel
# -----------------------
class ViTModel(BaseModel):
    """
    ViT model subclass that supports Hessian-based null-space projection
    for continual fine-tuning (Algorithm 2) and a layer-wise variant (Algorithm 3).
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        use_lora: bool = False,
        lora_config = None,
        freeze_backbone: bool = False,
        dataset_name: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__(
            model_name=model_name,
            use_lora=use_lora,
            lora_config=lora_config,
            freeze_backbone=freeze_backbone,
        )

        # manual optimization to allow custom grad projection before optimizer.step()
        self.automatic_optimization = False

        # register/activate head if provided
        if dataset_name and num_classes:
            self.register_head(dataset_name, num_classes)
            self.set_active_head(dataset_name)

        # store last computed bases
        self.global_U: Optional[torch.Tensor] = None       # shape (n_params, r)
        self.layerwise_U: Dict[str, torch.Tensor] = {}     # prefix -> U

    # -----------------------
    # Trainable params helper
    # -----------------------
    def get_trainable_params(self) -> List[torch.Tensor]:
        """
        Return list of parameters that are trainable (requires_grad==True)
        Typically this will be LoRA adapter params if backbone frozen.
        """
        return [p for p in self.parameters() if p.requires_grad]

    # -----------------------
    # Hessian-based projections (global)
    # -----------------------
    def _compute_projection_global(
        self,
        batch: Dict[str, torch.Tensor],
        k: int = 16,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Build a projection function using top-k Hessian eigenvectors using hessian_eigenthings.
        Uses the single incoming batch to approximate the Hessian operator (you can change this).
        """
        if compute_hessian_eigenthings is None:
            raise ImportError(
                "hessian_eigenthings not available. Install it with "
                "'pip install hessian-eigenthings' (or check its API). "
                f"Original import error: {_hessian_import_error}"
            )

        params = self.get_trainable_params()
        if len(params) == 0:
            # nothing to project
            return lambda g: g

        device = next(self.parameters()).device

        # Wrap the single batch into a tiny dataloader-like list of (inputs, labels)
        # compute_hessian_eigenthings expects a dataloader; many implementations accept an iterable.
        single_loader = [(
            batch["images"].to(device),
            batch["labels"].to(device)
        )]

        # call the library function - signature may vary between versions; adapt if necessary
        # we assume it returns (eigenvals, eigenvecs) where eigenvecs is (n_params, k)
        _, eigenvecs = compute_hessian_eigenthings(
            model=self,
            dataloader=single_loader,
            loss=self.criterion,
            num_eigenthings=k,
            params=params
        )
        eigenvecs = torch.Tensor(eigenvecs)

        # ensure eigenvecs is a torch tensor on the right device
        U = eigenvecs.to(device)

        # store basis
        self.global_U = U
        return build_projection_from_U(U)

    # -----------------------
    # Layer-wise variant
    # -----------------------
    def _compute_projection_layerwise(
        self,
        batch: Dict[str, torch.Tensor],
        k: int = 8,
    ) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        """
        Build per-layer projection functions. We group trainable parameters by top-level prefix in their name.
        Returns dict: prefix -> projection_fn.
        Also caches U in self.layerwise_U[prefix].
        """
        if compute_hessian_eigenthings is None:
            raise ImportError(
                "hessian_eigenthings not available. Install it with "
                "'pip install hessian-eigenthings' (or check its API)."
            )

        device = next(self.parameters()).device
        named_trainable = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        if not named_trainable:
            return {}

        # group by prefix (customize grouping logic if desired)
        groups: Dict[str, List[Tuple[str, torch.Tensor]]] = {}
        for n, p in named_trainable:
            prefix = n.split('.')[0]
            groups.setdefault(prefix, []).append((n, p))

        single_loader = [(
            batch["images"].to(device),
            batch["labels"].to(device)
        )]

        projections: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for prefix, name_param_list in groups.items():
            params = [p for _, p in name_param_list]
            n_params = sum(p.numel() for p in params)
            if n_params == 0:
                continue

            num_eig = min(k, n_params)
            _, eigenvecs = compute_hessian_eigenthings(
                model=self,
                dataloader=single_loader,
                loss=self.criterion,
                num_eigenthings=num_eig,
                params=params
            )
            eigenvecs = torch.Tensor(eigenvecs)
            U = eigenvecs.to(device)
            self.layerwise_U[prefix] = U
            projections[prefix] = build_projection_from_U(U)

        return projections

    # -----------------------
    # Apply projection to gradients
    # -----------------------
    def _apply_projection_to_grads(
            self,
            proj_fn: Callable[[torch.Tensor], torch.Tensor],
            params: List[torch.Tensor]) -> None:
        """
        Flatten grads of `params`, project using proj_fn, and write back to each param.grad.
        """
        device = next(self.parameters()).device
        grads = []
        for p in params:
            if p.grad is None:
                grads.append(torch.zeros_like(p))
            else:
                grads.append(p.grad.detach())
        flat = flatten_tensors(grads).to(device)
        flat_proj = proj_fn(flat)
        new_grads = unflatten_to(flat_proj, grads)
        for p, ng in zip(params, new_grads):
            p.grad = ng

    # -----------------------
    # Training step (manual optimization)
    # -----------------------
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        **kwargs
    ):
        """
        Main training_step implementing Algorithm 2 / Algorithm 3 projection.

        Args:
            projection: optional user-supplied projection callable (flat -> flat).
            mode: 'global' or 'layerwise'
            k: top-k eigenvectors to compute
        """
        # manual optimization
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        opt.zero_grad()

        imgs, labels = batch["images"], batch["labels"]
        device = next(self.parameters()).device
        imgs, labels = imgs.to(device), labels.to(device)

        out = self.forward(pixel_values=imgs, labels=labels)
        loss = out["loss"]
        self.manual_backward(loss)

        trainable_params = self.get_trainable_params()

        # 🔹 apply projection if set
        if hasattr(self, "projection_fn") and self.projection_fn is not None:
            self._apply_projection_to_grads(self.projection_fn, trainable_params)

        opt.step()
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    # -----------------------
    # Validation step
    # -----------------------
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, **kwargs):
        imgs = batch["images"].to(next(self.parameters()).device)
        labels = batch["labels"].to(next(self.parameters()).device)
        out = self.forward(pixel_values=imgs, labels=labels)
        loss = out["loss"]
        preds = torch.argmax(out["logits"], dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}







