from typing import Optional, Dict, Any
from .baseModel import BaseModel
from peft import LoraConfig

import torch


ResNetLoraConfig = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["layer4", "layer3"],
    lora_dropout=0.1,
    bias="none",
    task_type="IMAGE_CLASSIFICATION",
)


class ResNetModel(BaseModel):
    """ResNet wrapper that extends BaseModel and uses the backbone's built-in classifier.

    - Uses backbone outputs (logits/pooler_output) directly for classification.
    - No additional classifier head is added.
    """

    def __init__(
        self,
        model_name: str = "microsoft/resnet-50",
        use_lora: bool = False,
        lora_config: Optional[LoraConfig] = ResNetLoraConfig,
    ):
        super().__init__(model_name, use_lora, lora_config)

    def _get_logits_from_backbone(self, outputs: Any) -> torch.Tensor:
        # Prefer direct logits if present
        if hasattr(outputs, "logits") and outputs.logits is not None:
            return outputs.logits
        # Some models expose pooler_output for classification; use a linear layer from config if present
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        # Last hidden state -> mean-pool over sequence dim
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            feats = outputs.last_hidden_state
            if feats.ndim == 3:
                return feats.mean(dim=1)
            return feats
        # fallback: if backbone returned a tensor
        if isinstance(outputs, torch.Tensor):
            return outputs

        # fallback to tuple-like
        try:
            first = outputs[0]
            if isinstance(first, torch.Tensor):
                return first
        except Exception:
            pass

        raise RuntimeError("Unable to extract logits/features from backbone outputs")

    def forward(self, pixel_values, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        # Prepare inputs using the processor, then move tensors to model device
        inputs = self.processor(images=pixel_values, return_tensors="pt")
        device = next(self.backbone.parameters()).device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        no_grad = kwargs.get("no_grad", False)
        if no_grad:
            with torch.no_grad():
                outputs = self.backbone(**inputs)
        else:
            outputs = self.backbone(**inputs)

        logits = self._get_logits_from_backbone(outputs)

        result: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            labels = labels.to(device)
            loss = self.criterion(logits, labels)
            result["loss"] = loss

        return result

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, **kwargs) -> torch.Tensor:
        # support common batch key names
        pixel_values = batch.get("pixel_values") or batch.get("images") or batch.get("image")
        labels = batch.get("labels") or batch.get("label") or batch.get("targets")

        out = self.forward(pixel_values, labels=labels)
        loss = out.get("loss")
        if loss is None:
            # compute if forward didn't compute
            logits = out["logits"]
            loss = self.criterion(logits, labels.to(logits.device))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, **kwargs) -> None:
        pixel_values = batch.get("pixel_values") or batch.get("images") or batch.get("image")
        labels = batch.get("labels") or batch.get("label") or batch.get("targets")

        out = self.forward(pixel_values, labels=labels, no_grad=True)
        logits = out["logits"]
        loss = out.get("loss")
        if loss is None:
            loss = self.criterion(logits, labels.to(logits.device))

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels.to(logits.device)).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)




