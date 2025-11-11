from abc import abstractmethod
from typing import Any, Dict, Optional, cast
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModel, AutoConfig, PreTrainedModel
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model


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

