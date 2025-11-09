from typing import Optional, Dict, Any
from .baseModel import BaseModel
from peft import LoraConfig
import torch


# ViT LoRA config - target attention and MLP layers
ViTLoraConfig = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],  # Attention and MLP layers
    lora_dropout=0.1,
    bias="none",
    task_type="IMAGE_CLASSIFICATION",
)


class ViTModel(BaseModel):
    """Vision Transformer (ViT) wrapper that extends BaseModel.
    
    Supports ViT models from Hugging Face transformers library.
    - Uses the [CLS] token output for classification
    - Supports LoRA fine-tuning via PEFT
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        use_lora: bool = False,
        lora_config: Optional[LoraConfig] = ViTLoraConfig,
        num_classes: Optional[int] = None,
    ):
        super().__init__(model_name, use_lora, lora_config)
        self.num_classes = num_classes
        
        # ViT models from transformers typically have a classifier head
        # If num_classes is specified and different from model's, replace it
        if num_classes is not None and hasattr(self.backbone, 'classifier'):
            if hasattr(self.backbone.classifier, 'out_features'):
                if self.backbone.classifier.out_features != num_classes:
                    in_features = self.backbone.classifier.in_features
                    self.backbone.classifier = torch.nn.Linear(
                        in_features, num_classes
                    )

    def _get_logits_from_backbone(self, outputs: Any) -> torch.Tensor:
        """Extract logits from ViT model outputs."""
        # ViT models typically return logits directly
        if hasattr(outputs, "logits") and outputs.logits is not None:
            return outputs.logits
        
        # Some ViT variants return last_hidden_state and we need to extract [CLS] token
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            last_hidden_state = outputs.last_hidden_state
            # [CLS] token is typically the first token (index 0)
            cls_token = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            
            # If classifier exists, use it
            if hasattr(self.backbone, 'classifier'):
                return self.backbone.classifier(cls_token)
            else:
                # Fallback: return CLS token (might need a classifier head)
                return cls_token
        
        # Pooler output if available
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooler_output = outputs.pooler_output
            if hasattr(self.backbone, 'classifier'):
                return self.backbone.classifier(pooler_output)
            return pooler_output
        
        # If outputs is a tensor
        if isinstance(outputs, torch.Tensor):
            return outputs
        
        # Try tuple/list
        try:
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                first = outputs[0]
                if isinstance(first, torch.Tensor):
                    # If it's a sequence, take [CLS] token
                    if first.ndim == 3:
                        cls_token = first[:, 0, :]
                        if hasattr(self.backbone, 'classifier'):
                            return self.backbone.classifier(cls_token)
                        return cls_token
                    return first
        except Exception:
            pass

        raise RuntimeError("Unable to extract logits from ViT model outputs")

    def forward(self, pixel_values, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Forward pass through ViT model."""
        # Prepare inputs using the processor
        inputs = self.processor(images=pixel_values, return_tensors="pt")
        device = next(self.backbone.parameters()).device
        
        # Move inputs to device
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
        """Training step for ViT model."""
        pixel_values = batch.get("pixel_values") or batch.get("images") or batch.get("image")
        labels = batch.get("labels") or batch.get("label") or batch.get("targets")

        out = self.forward(pixel_values, labels=labels)
        loss = out.get("loss")
        if loss is None:
            logits = out["logits"]
            loss = self.criterion(logits, labels.to(logits.device))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, **kwargs) -> None:
        """Validation step for ViT model."""
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

