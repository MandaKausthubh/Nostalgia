import os
import json
from datetime import datetime
from abc import abstractmethod
from typing import Any, Dict, Optional, cast, Tuple, Union, Callable, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModel, AutoConfig, PreTrainedModel
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model

from typing import TypedDict
from torch.utils.data import DataLoader

class TaskLoaders(TypedDict):
    train: DataLoader
    val: DataLoader
    num_classes: int

class BaseModel(pl.LightningModule):

    def __init__(
        self,
        model_name: str,
        use_lora: bool = False,
        lora_config: Optional[LoraConfig] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lora_config"])
        self.model_name = model_name

        # -------------------------------
        # Backbone + Config
        # -------------------------------
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # -------------------------------
        # LoRA integration (optional)
        # -------------------------------
        self.lora_config = lora_config
        self.use_lora = use_lora
        self.peft_applied = False

        if self.use_lora and lora_config is not None:
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.peft_applied = True

        # -------------------------------
        # Classification Heads (multi-task)
        # -------------------------------
        if hasattr(self.backbone.config, "hidden_size"):
            embed_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone, "hidden_dim"):
            embed_dim = self.backbone.hidden_dim
        else:
            raise AttributeError("Cannot determine embedding dimension for classification head.")

        self.embed_dim = int(embed_dim)
        self.heads = nn.ModuleDict()
        self.active_head: Optional[str] = None

        # -------------------------------
        # Loss + freezing
        # -------------------------------
        self.criterion = nn.CrossEntropyLoss()
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # -------------------------------------------------------------------------
    # Head management
    # -------------------------------------------------------------------------
    def register_head(self, name: str, num_classes: int) -> None:
        """Register a new classification head for a dataset or task."""
        self.heads[name] = nn.Linear(self.embed_dim, num_classes) #pyright: ignore[reportGeneralTypeIssues]

    def set_active_head(self, name: str) -> None:
        """Switch which classification head to use."""
        if name not in self.heads:
            raise KeyError(f"No head registered under name '{name}'")
        self.active_head = name

    def get_active_head(self) -> nn.Module:
        if self.active_head is None:
            raise RuntimeError("No active head set. Use set_active_head(name).")
        return self.heads[self.active_head]

    @torch.no_grad()
    def extract_features(self, dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts frozen backbone features and labels from a dataloader.
        Returns (features, labels) as tensors on CPU.
        """
        self.eval()
        feats, labels = [], []
        device = next(self.parameters()).device
        for batch in dataloader:
            pixel_values = batch["images"].to(device)
            y = batch["labels"].to(device)
            z = self.forward_features(pixel_values)
            feats.append(z.detach().cpu())
            labels.append(y.detach().cpu())
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, labels

    def retrain_head_for_task(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_classes: int,
        lr: float = 1e-3,
        epochs: int = 5,
    ) -> float:
        """
        Retrains a simple linear classifier on frozen features for a given task.
        Returns top-1 validation accuracy.
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # Extract features for train and val sets
            X_train, y_train = self.extract_features(train_loader)
            X_val, y_val = self.extract_features(val_loader)

        clf = torch.nn.Linear(X_train.shape[1], num_classes).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Simple training loop
        for _ in range(epochs):
            clf.train()
            idx = torch.randperm(len(X_train))
            for i in range(0, len(X_train), 256):
                j = idx[i:i+256]
                xb, yb = X_train[j].to(device), y_train[j].to(device)
                opt.zero_grad()
                logits = clf(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        # Validation
        clf.eval()
        with torch.no_grad():
            logits = clf(X_val.to(device))
            preds = logits.argmax(dim=1)
            acc = (preds == y_val.to(device)).float().mean().item()

        return acc

    def evaluate_representation_retention(
        self,
        past_tasks : Dict[str, TaskLoaders],
        current_epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates representation retention for all past tasks.
        past_tasks: dict {task_name: {"train": train_loader, "val": val_loader, "num_classes": int}}
        Returns dict of task_name -> retrained-head accuracy.
        Logs results via JSON.
        """
        retention_results = {}

        for task_name, loaders in past_tasks.items():
            print(f"[Evaluating representation retention for {task_name}]")
            acc = self.retrain_head_for_task(
                loaders["train"], # pyright: ignore[reportGeneralTypeIssues]
                loaders["val"],   # pyright: ignore[reportGeneralTypeIssues]
                num_classes=loaders["num_classes"] # pyright: ignore[reportGeneralTypeIssues]
            )
            retention_results[f"repr_acc_{task_name}"] = acc

        # Log to JSON
        self.log_metrics_to_json(retention_results, epoch=current_epoch, prefix="repr_")
        return retention_results




    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError("Unexpected backbone output type.")

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        head = self.get_active_head()
        return head(features)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        feats = self.forward_features(pixel_values)
        logits = self.forward_head(feats)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {"logits": logits, "loss": loss}

    # -------------------------------------------------------------------------
    # Abstract training methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def projection_based_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, projection, **kwargs) -> torch.Tensor:
        pass

    # -------------------------------------------------------------------------
    # PEFT management
    # -------------------------------------------------------------------------
    def _apply_peft_model(self) -> None:
        if self.lora_config is None or not self.use_lora:
            raise ValueError("LoRA configuration must be provided to apply PEFT model.")
        if not self.peft_applied:
            self.backbone = get_peft_model(cast(PreTrainedModel, self.backbone), self.lora_config)
            self.peft_applied = True

    def combine_peft_model(self) -> None:
        if not self.use_lora:
            raise ValueError("LoRA is not enabled; cannot combine model.")
        if self.peft_applied and isinstance(self.backbone, (PeftMixedModel, PeftModel)):
            self.backbone = self.backbone.merge_and_unload()
            self.peft_applied = False
        else:
            raise TypeError("Backbone is not a PEFT model; cannot combine.")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def setup_logging(
            self,
            run_name: str = "Base_Class_Experiment",
            log_dir: str = "logs/experiments") -> None:
        """
        Initializes the JSON logger directory and metadata.
        Should be called once before training starts.
        """
        if run_name is None:
            run_name = f"{self.model_name.replace('/', '_')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.run_name = run_name
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.log_dir, "metrics.json")

        # Initialize file if not exists
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                json.dump({"run_name": run_name, "metrics": []}, f, indent=4)

        print(f"[Logger initialized] → {self.metrics_file}")

    def log_metrics_to_json(self, metrics: Dict[str, Any], step: Optional[int] = None, epoch: Optional[int] = None, prefix: str = "") -> None:
        """
        Logs metrics both through Lightning's self.log() and appends them to a JSON file for later analysis.

        Args:
            metrics: dict of metric_name -> value
            step: current step (optional)
            epoch: current epoch (optional)
            prefix: optional prefix for metric names (e.g., 'train_', 'val_')
        """
        # ---------- 1. Log to Lightning ----------
        for k, v in metrics.items():
            self.log(f"{prefix}{k}", v, prog_bar=False, on_step=True, on_epoch=True)

        # ---------- 2. Append to JSON file ----------
        record = {
            "step": int(step) if step is not None else int(self.global_step),
            "epoch": int(epoch) if epoch is not None else int(self.current_epoch),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        # Load existing logs
        if not hasattr(self, "metrics_file") or not os.path.exists(self.metrics_file):
            self.setup_logging()

        with open(self.metrics_file, "r") as f:
            data = json.load(f)

        data["metrics"].append(record)

        with open(self.metrics_file, "w") as f:
            json.dump(data, f, indent=4)


    def set_projection(self, proj_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> None:
            """
            Sets the active projection operator (e.g., Nostalgia null-space projection).
            If None, disables projection.
            """
            self.projection_fn = proj_fn

    def _apply_projection_to_grads(
            self,
            proj_fn: Callable[[torch.Tensor], torch.Tensor],
            params: List[TensorOrParam]
        ) -> None:
            """
            Flatten grads of `params`, project using proj_fn, and write back to param.grad.
            Works for both nn.Parameter and torch.Tensor.
            """
            device = next(self.parameters()).device
            grads = []
            for p in params:
                if p.grad is None:
                    grads.append(torch.zeros_like(p))
                else:
                    grads.append(p.grad.detach())
            flat = torch.cat([g.reshape(-1) for g in grads]).to(device)
            flat_proj = proj_fn(flat)
            # write back to each param.grad
            idx = 0
            for p, g in zip(params, grads):
                n = p.numel()
                p.grad = flat_proj[idx:idx+n].view_as(p)
                idx += n

