import torch
from typing import Dict, List, Tuple

# -------------------------
# Basic accuracy metrics
# -------------------------
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int] = (1,)) -> Dict[int, float]:
    """Compute top-k accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[k] = (correct_k / batch_size).item()
        return res


# -------------------------
# Forgetting metrics
# -------------------------
def task_forgetting(prev_loss: float, current_loss: float) -> float:
    """
    Computes forgetting for a given task:
    E_t = L_t(current) - L_t(previous)
    """
    return current_loss - prev_loss

def average_forgetting(forgetting_values: List[float]) -> float:
    """Mean forgetting across tasks."""
    if len(forgetting_values) == 0:
        return 0.0
    return sum(forgetting_values) / len(forgetting_values)


# -------------------------
# OOD robustness metrics
# -------------------------
def compute_ood_metrics(id_acc: float, ood_accs: List[float]) -> Dict[str, float]:
    """Compute OOD averages and generalization gap."""
    avg_ood = sum(ood_accs) / len(ood_accs)
    std_ood = torch.std(torch.tensor(ood_accs)).item()
    gap = id_acc - avg_ood
    return {
        "avg_ood_acc": avg_ood,
        "std_ood_acc": std_ood,
        "id_ood_gap": gap,
    }


# -------------------------
# Efficiency metrics
# -------------------------
def count_trainable_params(model: torch.nn.Module) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Aggregate summary
# -------------------------
def summarize_metrics(
    id_acc: float,
    ood_accs: List[float],
    forgetting_values: List[float],
    model: torch.nn.Module,
    epoch_time: float
) -> Dict[str, float]:
    """Returns a dictionary of summary metrics for logging."""
    ood_summary = compute_ood_metrics(id_acc, ood_accs)
    forget_summary = {
        "avg_forgetting": average_forgetting(forgetting_values)
    }

    return {
        "id_acc": id_acc,
        **ood_summary,
        **forget_summary,
        "trainable_params": count_trainable_params(model),
        "epoch_time": epoch_time
    }
