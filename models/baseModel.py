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
    ):
        self.save_hyperparameters()
        self.model_name = model_name

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        self.lora_config = lora_config
        self.use_lora = use_lora
        self.peft_applied = False

        if self.use_lora and lora_config is not None:
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.peft_applied = True

        self.criterion = nn.CrossEntropyLoss()   # Defaulting to classification tasks

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, **kwargs) -> None:
        pass

    def _apply_peft_model(self) -> None:
        if self.lora_config is None or not self.use_lora:
            raise ValueError("LoRA configuration must be provided to apply PEFT model.")

        if self.use_lora and not self.peft_applied and self.lora_config is not None:
            self.backbone = get_peft_model(cast(PreTrainedModel, self.backbone), self.lora_config)
            self.peft_applied = True

    def combine_peft_model(self) -> None:
        if not self.use_lora:
            raise ValueError("LoRA is not enabled; cannot combine model.")

        if self.use_lora and self.peft_applied:
            if isinstance(self.backbone, (PeftMixedModel, PeftModel)):
                self.backbone = self.backbone.merge_and_unload()
                self.peft_applied = False
            else:
                raise TypeError("Backbone is not a PEFT model; cannot combine.")


    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-4)





















