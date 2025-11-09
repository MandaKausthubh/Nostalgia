"""
Metrics tracking for continual learning experiments.

Tracks accuracy history, forgetting, and other metrics needed for visualizations.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import torch


class ContinualLearningMetrics:
    """Track metrics for continual learning experiments."""
    
    def __init__(self, num_tasks: int, task_names: Optional[List[str]] = None):
        """
        Initialize metrics tracker.
        
        Args:
            num_tasks: Number of tasks
            task_names: Optional list of task names
        """
        self.num_tasks = num_tasks
        self.task_names = task_names or [f"Task{i+1}" for i in range(num_tasks)]
        
        # Accuracy history: acc_history[t][i] = accuracy on task i after training task t
        self.acc_history: Dict[int, Dict[int, float]] = defaultdict(dict)
        
        # Forgetting: forgetting[t][i] = forgetting of task i after task t
        self.forgetting: Dict[int, Dict[int, float]] = defaultdict(dict)
        
        # Training metrics
        self.train_loss_over_time: List[float] = []
        self.train_steps: List[int] = []
        self.avg_forgetting_over_time: List[float] = []
        
        # Task boundaries: (start_step, end_step) for each task
        self.task_boundaries: List[Tuple[int, int]] = []
        self.current_step = 0
        
        # Evaluation metrics
        self.eval_metrics: Dict[str, Dict[str, float]] = {}
        
        # Hessian eigenvalues (for visualization)
        self.hessian_eigs: Dict[str, np.ndarray] = {}
        
    def record_task_boundary(self, task_id: int, start_step: int, end_step: int):
        """Record task boundary."""
        self.task_boundaries.append((start_step, end_step))
    
    def record_accuracy(self, task_id: int, eval_task_id: int, accuracy: float):
        """
        Record accuracy on a task.
        
        Args:
            task_id: Current task being trained (after training this task)
            eval_task_id: Task being evaluated
            accuracy: Accuracy value (0-100 or 0-1)
        """
        # Normalize to 0-1 if needed
        if accuracy > 1.0:
            accuracy = accuracy / 100.0
        
        self.acc_history[task_id][eval_task_id] = accuracy
    
    def compute_forgetting(self, task_id: int, eval_task_id: int) -> float:
        """
        Compute forgetting for a task.
        
        Forgetting = max accuracy on task i - current accuracy on task i
        
        Args:
            task_id: Current task (after training this task)
            eval_task_id: Task to compute forgetting for
            
        Returns:
            Forgetting value
        """
        # Find maximum accuracy on this task when it was just learned
        max_acc = 0.0
        if eval_task_id in self.acc_history.get(eval_task_id, {}):
            max_acc = self.acc_history[eval_task_id][eval_task_id]
        
        # Also check all evaluations up to current task
        for t in range(eval_task_id, task_id + 1):
            if eval_task_id in self.acc_history.get(t, {}):
                max_acc = max(max_acc, self.acc_history[t][eval_task_id])
        
        # Get current accuracy
        if eval_task_id not in self.acc_history[task_id]:
            # If not evaluated yet, return 0
            return 0.0
        
        current_acc = self.acc_history[task_id][eval_task_id]
        
        # Forgetting = max accuracy - current accuracy
        forgetting = max(0.0, max_acc - current_acc)
        
        self.forgetting[task_id][eval_task_id] = forgetting
        return forgetting
    
    def compute_average_forgetting(self, task_id: int) -> float:
        """Compute average forgetting across all previous tasks."""
        if task_id == 0:
            return 0.0
        
        forgettings = []
        for i in range(task_id):
            forget = self.compute_forgetting(task_id, i)
            forgettings.append(forget)
        
        avg_forget = np.mean(forgettings) if forgettings else 0.0
        return avg_forget
    
    def record_train_loss(self, loss: float, step: Optional[int] = None):
        """Record training loss."""
        self.train_loss_over_time.append(loss)
        if step is not None:
            self.train_steps.append(step)
        else:
            self.train_steps.append(self.current_step)
            self.current_step += 1
    
    def record_eval_metrics(self, method_name: str, id_acc: float, ood_acc: float, 
                           avg_forgetting: Optional[float] = None):
        """Record evaluation metrics for a method."""
        self.eval_metrics[method_name] = {
            "id_accuracy": id_acc,
            "ood_accuracy": ood_acc,
            "avg_forgetting": avg_forgetting or 0.0,
        }
    
    def record_hessian_eigenvalues(self, task_name: str, eigenvalues: np.ndarray):
        """Record Hessian eigenvalues for a task."""
        self.hessian_eigs[task_name] = eigenvalues
    
    def get_forgetting_matrix(self) -> np.ndarray:
        """
        Get forgetting matrix.
        
        Returns:
            Matrix of shape (num_tasks, num_tasks) where [t, i] = forgetting of task i after task t
        """
        matrix = np.zeros((self.num_tasks, self.num_tasks))
        
        for t in range(self.num_tasks):
            for i in range(t):
                if i in self.forgetting.get(t, {}):
                    matrix[t, i] = self.forgetting[t][i]
        
        return matrix
    
    def get_accuracy_history_matrix(self) -> np.ndarray:
        """
        Get accuracy history matrix.
        
        Returns:
            Matrix of shape (num_tasks, num_tasks) where [t, i] = accuracy on task i after task t
        """
        matrix = np.zeros((self.num_tasks, self.num_tasks))
        
        for t in range(self.num_tasks):
            for i in range(t + 1):
                if i in self.acc_history.get(t, {}):
                    matrix[t, i] = self.acc_history[t][i]
        
        return matrix
    
    def get_task_retention_data(self) -> Dict[int, List[float]]:
        """
        Get task retention data for plotting.
        
        Returns:
            Dictionary mapping task_id to list of accuracies across subsequent tasks
        """
        retention = defaultdict(list)
        acc_matrix = self.get_accuracy_history_matrix()
        
        for task_id in range(self.num_tasks):
            for t in range(task_id, self.num_tasks):
                retention[task_id].append(acc_matrix[t, task_id])
        
        return dict(retention)
    
    def update_avg_forgetting_over_time(self):
        """Update average forgetting over time."""
        for t in range(self.num_tasks):
            avg_forget = self.compute_average_forgetting(t)
            self.avg_forgetting_over_time.append(avg_forget)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for saving."""
        return {
            "acc_history": dict(self.acc_history),
            "forgetting": dict(self.forgetting),
            "train_loss_over_time": self.train_loss_over_time,
            "train_steps": self.train_steps,
            "avg_forgetting_over_time": self.avg_forgetting_over_time,
            "task_boundaries": self.task_boundaries,
            "task_names": self.task_names,
            "eval_metrics": self.eval_metrics,
            "hessian_eigs": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in self.hessian_eigs.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ContinualLearningMetrics":
        """Load metrics from dictionary."""
        num_tasks = len(data.get("task_names", []))
        metrics = cls(num_tasks, data.get("task_names"))
        metrics.acc_history = defaultdict(dict, data.get("acc_history", {}))
        metrics.forgetting = defaultdict(dict, data.get("forgetting", {}))
        metrics.train_loss_over_time = data.get("train_loss_over_time", [])
        metrics.train_steps = data.get("train_steps", [])
        metrics.avg_forgetting_over_time = data.get("avg_forgetting_over_time", [])
        metrics.task_boundaries = data.get("task_boundaries", [])
        metrics.eval_metrics = data.get("eval_metrics", {})
        metrics.hessian_eigs = {
            k: np.array(v) if isinstance(v, list) else v 
            for k, v in data.get("hessian_eigs", {}).items()
        }
        return metrics

