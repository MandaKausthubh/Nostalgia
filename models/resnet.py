import torch
from torch._prims_common import Tensor
import torch.nn as nn
import torchvision.models as tvm
from typing import Dict, Optional, Callable, List, Tuple, cast
from .baseModel import BaseModel

# Try to import the hessian-eigenthings helper
try:
    from hessian_eigenthings import compute_hessian_eigenthings
except Exception as e:
    compute_hessian_eigenthings = None
    _hessian_import_error = e


# -----------------------
# Utility helpers (same as vit_model; keep consistent)
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
    def proj_fn(g_flat: torch.Tensor) -> torch.Tensor:
        coeff = U.T @ g_flat
        return g_flat - (U @ coeff)
    return proj_fn


# -----------------------
# ResNetModel
# -----------------------
class ResNetModel(BaseModel):
    """
    ResNet-based model subclass with Hessian null-space projection support.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dataset_name: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        # call BaseModel but we will replace backbone with torchvision ResNet
        super().__init__(model_name=model_name, use_lora=False, lora_config=None, freeze_backbone=freeze_backbone)

        # manual optimization
        self.automatic_optimization = False

        # Replace backbone with torchvision model
        self.backbone = getattr(tvm, model_name)(pretrained=pretrained)
        embed_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embed_dim = embed_dim

        # the BaseModel's heads dict exists; ensure consistent state
        self.heads = nn.ModuleDict()
        self.active_head = None
        if dataset_name and num_classes:
            self.register_head(dataset_name, num_classes)
            self.set_active_head(dataset_name)

        self.global_U: Optional[torch.Tensor] = None
        self.layerwise_U: Dict[str, torch.Tensor] = {}

    # -----------------------
    # Trainable params helper
    # -----------------------
    def get_trainable_params(self) -> List[torch.Tensor]:
        return [p for p in self.parameters() if p.requires_grad]

    # -----------------------
    # Projection builders using hessian_eigenthings
    # -----------------------
    def _compute_projection_global(self, batch: Dict[str, torch.Tensor], k: int = 12) -> Callable[[torch.Tensor], torch.Tensor]:
        if compute_hessian_eigenthings is None:
            raise ImportError(
                "hessian_eigenthings not available. Install it with "
                "'pip install hessian-eigenthings' (or check its API). "
                f"Import error: {_hessian_import_error}"
            )

        params = self.get_trainable_params()
        if len(params) == 0:
            return lambda g: g

        device = next(self.parameters()).device
        single_loader = [(
            batch["images"].to(device),
            batch["labels"].to(device)
        )]

        _, eigenvecs = compute_hessian_eigenthings(
            model=self,
            dataloader=single_loader,
            loss=self.criterion,
            num_eigenthings=k,
            params=params
        )
        eigenvecs = torch.Tensor(eigenvecs)
        U = eigenvecs.to(device)
        self.global_U = U
        return build_projection_from_U(U)

    def _compute_projection_layerwise(self, batch: Dict[str, torch.Tensor], k:int = 8) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        if compute_hessian_eigenthings is None:
            raise ImportError(
                "hessian_eigenthings not available. Install it with "
                "'pip install hessian-eigenthings' (or check its API)."
            )

        device = next(self.parameters()).device
        named_trainable = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
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
            eigenvals, eigenvecs = compute_hessian_eigenthings(
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
    # apply projection to grads
    # -----------------------
    def _apply_projection_to_grads(self, proj_fn: Callable[[torch.Tensor], torch.Tensor], params: List[torch.Tensor]) -> None:
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
    # training_step
    # -----------------------
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        **kwargs
    ):
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
    # validation_step
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
