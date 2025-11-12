import os
import yaml
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from datetime import datetime

# -------------------------------------------------------
# Imports from your repo
# -------------------------------------------------------
from models.vit import ViTModel
from models.resnet import ResNetModel
from datasets.ImageNetV2 import ImageNetV2Dataset
from datasets.ImageNetV2 import ImageNetV2Wrapper
from datasets.TinyImageNet import TinyImageNetDataset
from datasets.ImagenetODD import ImageNetODataset
from metrics.metrics import summarize_metrics

# -------------------------------------------------------
# Dataset registry
# -------------------------------------------------------
DATASET_REGISTRY = {
    "ImageNetV2Dataset": ImageNetV2Dataset,
    "TinyImageNetDataset": TinyImageNetDataset,
    "ImageNetODataset": ImageNetODataset,
}

# Update the dataset registry to use the wrapper class
DATASET_REGISTRY.update({
    "ImageNetV2Dataset": ImageNetV2Wrapper,
})

# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
def build_projection_from_U(U: torch.Tensor):
    def proj_fn(g_flat: torch.Tensor) -> torch.Tensor:
        coeff = U.T @ g_flat
        return g_flat - (U @ coeff)
    return proj_fn


@torch.no_grad()
def evaluate_accuracy(model, dataloader):
    """Computes top-1 accuracy on a given dataloader."""
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0
    for batch in dataloader:
        imgs, labels = batch["images"].to(device), batch["labels"].to(device)
        logits = model.forward(pixel_values=imgs)["logits"]
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / total if total > 0 else 0.0


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


# -------------------------------------------------------
# Config loader
# -------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Nostalgia Sequential Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override_epochs", type=int, default=None, help="Override epochs for all tasks")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------
# Main Experiment
# -------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["experiment"].get("seed", 42))

    # Setup run name and loggers
    run_name = f"{cfg['experiment']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_name = cfg["experiment"]["project_name"]

    loggers = []
    if cfg["experiment"].get("use_wandb", True):
        loggers.append(WandbLogger(project=project_name, name=run_name))
    if cfg["experiment"].get("use_tensorboard", True):
        loggers.append(TensorBoardLogger(save_dir="logs/tb", name=run_name))

    # ---------------------------------------------------
    # Model selection
    # ---------------------------------------------------
    model_type = cfg["model"]["type"].lower()
    if model_type == "vit":
        model = ViTModel(model_name=cfg["model"]["name"], freeze_backbone=cfg["model"]["freeze_backbone"])
    else:
        model = ResNetModel(model_name=cfg["model"]["name"], freeze_backbone=cfg["model"]["freeze_backbone"])

    model.setup_logging(run_name)

    # ---------------------------------------------------
    # Datasets
    # ---------------------------------------------------
    datasets = {}
    for task in cfg["tasks"]:
        DClass = DATASET_REGISTRY[task["dataset_class"]]
        dataset = DClass(**{k: v for k, v in task.items() if k in ["root", "list_file"]})
        datasets[task["name"]] = {
            "train": DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=4),
            "val": DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4),
            "num_classes": dataset.metadata["num_classes"],
        }

    # Register heads
    for name, d in datasets.items():
        model.register_head(name, d["num_classes"])

    # ---------------------------------------------------
    # Sequential Training
    # ---------------------------------------------------
    baseline_accs = {}
    hessian_bases = {}
    past_tasks = {}
    k_lanczos = cfg["training"]["k_lanczos"]

    for task in cfg["tasks"]:
        name = task["name"]
        nostalgia = task.get("nostalgia", False)
        hsrc = task.get("hessian_source", None)
        epochs = args.override_epochs or task["epochs"]

        print(f"\n🚀 Task {name} | nostalgia={nostalgia} | epochs={epochs}")
        model.set_active_head(name)

        # ------------------------------------------------
        # Projection setup (Nostalgia)
        # ------------------------------------------------
        proj_fn = None
        if nostalgia:
            if isinstance(hsrc, list):
                U_cat = torch.cat([hessian_bases[h] for h in hsrc], dim=1)
                U_avg, _ = torch.linalg.qr(U_cat)
                proj_fn = build_projection_from_U(U_avg)
            elif isinstance(hsrc, str):
                U_src = hessian_bases[hsrc]
                proj_fn = build_projection_from_U(U_src)
        model.set_projection(proj_fn)

        # ------------------------------------------------
        # Create per-task Trainer
        # ------------------------------------------------
        trainer_task = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=epochs,
            logger=loggers,
            log_every_n_steps=10,
            enable_checkpointing=False,
        )

        # ------------------------------------------------
        # Train
        # ------------------------------------------------
        trainer_task.fit(model, datasets[name]["train"], datasets[name]["val"])

        # ------------------------------------------------
        # Evaluate current task
        # ------------------------------------------------
        acc = evaluate_accuracy(model, datasets[name]["val"])
        baseline_accs[name] = acc
        model.log_metrics_to_json({f"id_acc_{name}": acc}, epoch=trainer_task.current_epoch)
        print(f"✅ Validation accuracy ({name}): {acc:.4f}")

        # ------------------------------------------------
        # Compute Hessian (for next Nostalgia projection)
        # ------------------------------------------------
        print(f"📊 Computing Hessian for {name} ...")
        model._compute_projection_global(next(iter(datasets[name]["train"])), k=k_lanczos)
        hessian_bases[name] = model.global_U.clone().detach()

        # ------------------------------------------------
        # Evaluate representation retention on previous tasks
        # ------------------------------------------------
        if past_tasks:
            repr_accs = model.evaluate_retention(past_tasks)
            repr_forgets = model.compute_representation_forgetting(
                baseline_accs={k: baseline_accs[k] for k in past_tasks},
                current_accs={
                    k: repr_accs[f"repr_acc_{k}"]
                    for k in past_tasks
                    if f"repr_acc_{k}" in repr_accs
                },
            )
            print("🧠 Representation Forgetting:", repr_forgets)

        # Mark task as completed
        past_tasks[name] = datasets[name]

    # ---------------------------------------------------
    # Final Summary
    # ---------------------------------------------------
    id_acc_final = baseline_accs[list(baseline_accs.keys())[-1]]
    summary = summarize_metrics(
        id_acc=id_acc_final,
        ood_accs=list(baseline_accs.values())[:-1],
        forgetting_values=[],
        model=model,
        epoch_time=trainer_task.estimated_stepping_batches, # pyright: ignore[reporrUnboundVariable]
    )

    model.log_metrics_to_json(summary)
    print("\n✅ Final Summary:", summary)
    print("Logs saved at:", model.metrics_file)


if __name__ == "__main__":
    main()
