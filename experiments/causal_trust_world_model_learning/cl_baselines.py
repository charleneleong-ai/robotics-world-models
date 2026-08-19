"""
Continual Learning Baselines for Comparison

Implements standard CL methods:
- EWC (Elastic Weight Consolidation)
- LwF (Learning without Forgetting)
- PackNet
- Experience Replay
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CLResult:
    """Result from a CL experiment."""
    task_accuracies: List[List[float]]  # [task][time_step]
    average_accuracy: List[float]  # [time_step]
    backward_transfer: List[float]  # [time_step]
    forward_transfer: List[float]  # [time_step]
    forgetting: List[float]  # [task]
    params_per_task: int  # trainable parameters per task


class EWC:
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017)."""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 5000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0
        
    def compute_fisher(self, dataloader: DataLoader):
        """Compute Fisher information matrix diagonal."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        for batch in dataloader:
            self.model.zero_grad()
            output = self.model(batch[0])
            loss = F.cross_entropy(output, batch[1])
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        
        # Average over samples
        n_samples = len(dataloader.dataset)
        for n in fisher:
            fisher[n] /= n_samples
            
        return fisher
    
    def consolidate(self, dataloader: DataLoader):
        """Save Fisher information and optimal parameters after task."""
        self.fisher_information[self.task_count] = self.compute_fisher(dataloader)
        self.optimal_params[self.task_count] = {
            n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.task_count += 1
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty."""
        penalty = 0.0
        for task_id in range(self.task_count):
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.fisher_information[task_id]:
                    diff = p - self.optimal_params[task_id][n]
                    penalty += (self.fisher_information[task_id][n] * diff.pow(2)).sum()
        return self.lambda_ewc * penalty


class LwF:
    """Learning without Forgetting (Li & Hoiem, 2017)."""
    
    def __init__(self, model: nn.Module, lambda_lwf: float = 2.0):
        self.model = model
        self.lambda_lwf = lambda_lwf
        self.previous_models: List[nn.Module] = []
        self.task_count = 0
        
    def consolidate(self):
        """Save model snapshot after task."""
        prev_model = copy.deepcopy(self.model)
        prev_model.eval()
        self.previous_models.append(prev_model)
        self.task_count += 1
    
    def penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LwF penalty using knowledge distillation."""
        if len(self.previous_models) == 0:
            return torch.tensor(0.0)
        
        penalty = 0.0
        for prev_model in self.previous_models:
            with torch.no_grad():
                prev_output = prev_model(x)
            current_output = self.model(x)
            
            # Knowledge distillation loss
            penalty += F.kl_div(
                F.log_softmax(current_output, dim=1),
                F.softmax(prev_output, dim=1),
                reduction='batchmean'
            )
        
        return self.lambda_lwf * penalty


class PackNet:
    """PackNet (Mallya & Lazebnik, 2018)."""
    
    def __init__(self, model: nn.Module, prune_ratio: float = 0.2):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks: List[Dict[str, torch.Tensor]] = []
        self.task_count = 0
        
    def prune_and_freeze(self):
        """Prune least important weights and freeze for new task."""
        mask = {}
        
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # Compute importance by magnitude
                importance = p.data.abs()
                
                # Prune least important weights
                threshold = torch.quantile(importance, self.prune_ratio)
                mask[n] = (importance >= threshold).float()
                
                # Apply mask
                p.data *= mask[n]
                p.requires_grad = False
        
        self.masks.append(mask)
        self.task_count += 1
    
    def get_trainable_mask(self) -> Dict[str, torch.Tensor]:
        """Get mask of trainable parameters (not frozen by previous tasks)."""
        combined_mask = {}
        
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # Start with all ones (trainable)
                mask = torch.ones_like(p.data)
                
                # Zero out frozen parameters
                for prev_mask in self.masks:
                    if n in prev_mask:
                        mask *= (1 - prev_mask[n])
                
                combined_mask[n] = mask
        
        return combined_mask


class ExperienceReplay:
    """Experience Replay with reservoir sampling."""
    
    def __init__(self, buffer_size: int = 500):
        self.buffer_size = buffer_size
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self.task_ids: List[int] = []
        self.seen_count = 0
        
    def add_to_buffer(self, x: torch.Tensor, y: torch.Tensor, task_id: int):
        """Add samples to buffer using reservoir sampling."""
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            self.seen_count += 1
            
            if len(self.buffer_x) < self.buffer_size:
                # Buffer not full, add sample
                self.buffer_x.append(x[i])
                self.buffer_y.append(y[i])
                self.task_ids.append(task_id)
            else:
                # Buffer full, reservoir sampling
                j = np.random.randint(0, self.seen_count)
                if j < self.buffer_size:
                    self.buffer_x[j] = x[i]
                    self.buffer_y[j] = y[i]
                    self.task_ids[j] = task_id
    
    def get_buffer_loader(self, batch_size: int = 32) -> Optional[DataLoader]:
        """Get DataLoader from buffer."""
        if len(self.buffer_x) == 0:
            return None
        
        x = torch.stack(self.buffer_x)
        y = torch.stack(self.buffer_y)
        
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_current_task_loader(self, dataloader: DataLoader, task_id: int) -> DataLoader:
        """Combine current task data with buffer samples."""
        buffer_loader = self.get_buffer_loader()
        
        if buffer_loader is None:
            return dataloader
        
        # Combine datasets
        all_x = []
        all_y = []
        
        # Add current task data
        for batch in dataloader:
            all_x.append(batch[0])
            all_y.append(batch[1])
        
        # Add buffer data
        for batch in buffer_loader:
            all_x.append(batch[0])
            all_y.append(batch[1])
        
        combined_x = torch.cat(all_x)
        combined_y = torch.cat(all_y)
        
        dataset = TensorDataset(combined_x, combined_y)
        return DataLoader(dataset, batch_size=32, shuffle=True)


class TrustAwareCL:
    """Trust-Aware Continual Learning (Your Method)."""
    
    def __init__(self, model: nn.Module, trust_threshold: float = 0.7):
        self.model = model
        self.trust_threshold = trust_threshold
        self.task_models: List[nn.Module] = []
        self.task_trust_scores: List[float] = []
        self.task_count = 0
        
    def compute_trust_score(self, dataloader: DataLoader) -> float:
        """Compute trust score for current task."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                output = self.model(batch[0])
                pred = output.argmax(dim=1)
                correct += (pred == batch[1]).sum().item()
                total += batch[1].shape[0]
        
        return correct / total if total > 0 else 0.0
    
    def should_consolidate(self, trust_score: float) -> bool:
        """Decide whether to consolidate based on trust score."""
        return trust_score >= self.trust_threshold
    
    def consolidate(self, dataloader: DataLoader):
        """Save model snapshot if trust is high enough."""
        trust_score = self.compute_trust_score(dataloader)
        
        if self.should_consolidate(trust_score):
            prev_model = copy.deepcopy(self.model)
            prev_model.eval()
            self.task_models.append(prev_model)
            self.task_trust_scores.append(trust_score)
            self.task_count += 1
            return True
        return False
    
    def penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute trust-weighted penalty."""
        if len(self.task_models) == 0:
            return torch.tensor(0.0)
        
        penalty = 0.0
        for i, prev_model in enumerate(self.task_models):
            with torch.no_grad():
                prev_output = prev_model(x)
            current_output = self.model(x)
            
            # Weight by trust score
            weight = self.task_trust_scores[i]
            
            # Knowledge distillation loss
            penalty += weight * F.kl_div(
                F.log_softmax(current_output, dim=1),
                F.softmax(prev_output, dim=1),
                reduction='batchmean'
            )
        
        return penalty


def compute_cl_metrics(
    task_accuracies: List[List[float]],
    num_tasks: int
) -> Dict[str, List[float]]:
    """Compute standard CL metrics."""
    
    T = len(task_accuracies[0])  # number of time steps
    
    # Average Accuracy at each time step
    avg_accuracy = []
    for t in range(T):
        acc = sum(task_accuracies[i][t] for i in range(num_tasks)) / num_tasks
        avg_accuracy.append(acc)
    
    # Backward Transfer (BWT) - how much accuracy on old tasks changed after learning new tasks
    bwt = []
    for t in range(1, T):
        bwt_t = 0
        for i in range(t):
            # BWT for task i at time t = final_acc - acc_when_i_was_last_trained
            bwt_t += task_accuracies[i][T-1] - task_accuracies[i][i]
        bwt_t /= t
        bwt.append(bwt_t)
    
    # Forward Transfer (FWT) - how much knowing old tasks helps learn new tasks
    fwt = []
    for t in range(1, T):
        fwt_t = 0
        count = 0
        for i in range(t, num_tasks):
            # FWT for task i at time t = acc_on_i_at_time_t - acc_on_i_before_training
            if t > 0 and t-1 < len(task_accuracies[i]):
                fwt_t += task_accuracies[i][t] - task_accuracies[i][t-1]
                count += 1
        fwt_t /= max(count, 1)
        fwt.append(fwt_t)
    
    # Forgetting Measure - max accuracy drop for each task
    forgetting = []
    for i in range(num_tasks):
        max_acc = max(task_accuracies[i][:i+1]) if i < len(task_accuracies[i]) else task_accuracies[i][0]
        final_acc = task_accuracies[i][-1] if task_accuracies[i][-1] > 0 else task_accuracies[i][i]
        forgetting.append(max_acc - final_acc)
    
    return {
        'average_accuracy': avg_accuracy,
        'backward_transfer': bwt,
        'forward_transfer': fwt,
        'forgetting': forgetting
    }


def run_cl_experiment(
    model_class,
    train_fn,
    eval_fn,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    num_tasks: int,
    device: str = 'cpu'
) -> Dict[str, CLResult]:
    """Run full CL experiment with all methods."""
    
    results = {}
    
    # Method 1: Naive Fine-Tuning (baseline)
    model = model_class().to(device)
    task_accs = []
    
    for t in range(num_tasks):
        # Train on current task
        train_fn(model, train_loaders[t], epochs=10, device=device)
        
        # Evaluate on all tasks
        accs = []
        for i in range(t + 1):
            acc = eval_fn(model, test_loaders[i], device=device)
            accs.append(acc)
        
        # Pad with zeros for future tasks
        accs.extend([0.0] * (num_tasks - len(accs)))
        task_accs.append(accs)
    
    # Transpose to [task][time]
    task_accs_transposed = list(map(list, zip(*task_accs)))
    metrics = compute_cl_metrics(task_accs_transposed, num_tasks)
    
    results['fine_tuning'] = CLResult(
        task_accuracies=task_accs_transposed,
        average_accuracy=metrics['average_accuracy'],
        backward_transfer=metrics['backward_transfer'],
        forward_transfer=metrics['forward_transfer'],
        forgetting=metrics['forgetting'],
        params_per_task=0
    )
    
    return results
