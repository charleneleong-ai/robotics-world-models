"""
Multimodal Continual Learning Extension

Extends TC-WM to handle vision, tactile, and proprioceptive signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class CrossModalAttention(nn.Module):
    """Cross-attention for fusing multiple modalities."""
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_q, embed_dim] - primary modality (e.g., vision)
            key_value: [batch, seq_kv, embed_dim] - secondary modalities (tactile + proprio)
        Returns:
            fused: [batch, seq_q, embed_dim]
        """
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]
        
        Q = self.query(query).view(B, L_q, self.num_heads, D // self.num_heads).transpose(1, 2)
        K = self.key(key_value).view(B, L_kv, self.num_heads, D // self.num_heads).transpose(1, 2)
        V = self.value(key_value).view(B, L_kv, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).contiguous().view(B, L_q, D)
        
        return self.out_proj(attn)


class MultimodalEncoder(nn.Module):
    """Encodes vision, tactile, and proprioceptive signals."""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Vision encoder (simple CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embed_dim)
        )
        
        # Tactile encoder (1D signal)
        self.tactile_encoder = nn.Sequential(
            nn.Linear(64, 128),  # 64-dim tactile signal
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # Proprioception encoder (joint angles + velocities)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(14, 64),  # 7 joints × 2 (pos + vel)
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # Cross-modal attention
        self.cross_attn = CrossModalAttention(embed_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self, 
        vision: torch.Tensor,
        tactile: torch.Tensor,
        proprio: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vision: [batch, 3, H, W] - RGB image
            tactile: [batch, 64] - tactile sensor readings
            proprio: [batch, 14] - joint positions and velocities
        Returns:
            fused: [batch, embed_dim] - multimodal embedding
        """
        # Encode each modality
        v = self.vision_encoder(vision)  # [B, embed_dim]
        t = self.tactile_encoder(tactile)  # [B, embed_dim]
        p = self.proprio_encoder(proprio)  # [B, embed_dim]
        
        # Reshape for cross-attention: [B, 1, embed_dim]
        v_seq = v.unsqueeze(1)
        tp_seq = torch.stack([t, p], dim=1)  # [B, 2, embed_dim]
        
        # Cross-attention: vision attends to tactile+proprio
        v_enhanced = self.cross_attn(v_seq, tp_seq).squeeze(1)  # [B, embed_dim]
        
        # Fusion
        fused = self.fusion(torch.cat([v_enhanced, t, p], dim=-1))
        
        return fused


class MultimodalTrustScorer(nn.Module):
    """Computes modality-specific trust scores from raw inputs."""
    
    def __init__(self, num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities
        
        # Per-modality encoders + trust heads (input sizes match raw data)
        self.vision_trust = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.tactile_trust = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.proprio_trust = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Learned modality weights
        self.modality_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
    
    def forward(
        self, 
        vision: torch.Tensor,
        tactile: torch.Tensor,
        proprio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with trust scores per modality and fused trust
        """
        trust_v = self.vision_trust(vision)
        trust_t = self.tactile_trust(tactile)
        trust_p = self.proprio_trust(proprio)
        
        weights = F.softmax(self.modality_weights, dim=0)
        trust_fused = weights[0] * trust_v + weights[1] * trust_t + weights[2] * trust_p
        
        return {
            'trust_vision': trust_v,
            'trust_tactile': trust_t,
            'trust_proprio': trust_p,
            'trust_fused': trust_fused,
            'modality_weights': weights
        }


class MultimodalContinualLearner(nn.Module):
    """Complete multimodal continual learning model."""
    
    def __init__(self, embed_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.encoder = MultimodalEncoder(embed_dim)
        self.trust_scorer = MultimodalTrustScorer(num_modalities=3)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Trust threshold
        self.trust_threshold = 0.7
        
    def forward(
        self,
        vision: torch.Tensor,
        tactile: torch.Tensor,
        proprio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with logits and trust scores
        """
        # Encode multimodal input
        fused = self.encoder(vision, tactile, proprio)
        
        # Classify
        logits = self.classifier(fused)
        
        # Compute trust scores
        trust_scores = self.trust_scorer(vision, tactile, proprio)
        
        return {
            'logits': logits,
            **trust_scores
        }
    
    def should_consolidate(self, trust_scores: Dict[str, torch.Tensor]) -> bool:
        """Decide whether to consolidate based on trust scores."""
        return trust_scores['trust_fused'].mean().item() >= self.trust_threshold


class MultimodalCLExperiment:
    """Run multimodal continual learning experiments."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def create_synthetic_multimodal_data(
        self, 
        num_samples: int = 500,
        num_tasks: int = 3
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Create synthetic multimodal data for testing."""
        
        train_loaders = []
        test_loaders = []
        
        for task_id in range(num_tasks):
            # Generate synthetic data
            vision = torch.randn(num_samples, 3, 32, 32)
            tactile = torch.randn(num_samples, 64)
            proprio = torch.randn(num_samples, 14)
            labels = torch.randint(0, 2, (num_samples,))
            
            # Split train/test
            train_size = int(0.8 * num_samples)
            
            train_dataset = TensorDataset(
                vision[:train_size], 
                tactile[:train_size], 
                proprio[:train_size], 
                labels[:train_size]
            )
            test_dataset = TensorDataset(
                vision[train_size:], 
                tactile[train_size:], 
                proprio[train_size:], 
                labels[train_size:]
            )
            
            train_loaders.append(DataLoader(train_dataset, batch_size=32, shuffle=True))
            test_loaders.append(DataLoader(test_dataset, batch_size=32))
        
        return train_loaders, test_loaders
    
    def train_epoch(
        self,
        model: MultimodalContinualLearner,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        for batch in dataloader:
            vision, tactile, proprio, labels = [b.to(self.device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(vision, tactile, proprio)
            loss = criterion(outputs['logits'], labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(
        self,
        model: MultimodalContinualLearner,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        correct = 0
        total = 0
        trust_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                vision, tactile, proprio, labels = [b.to(self.device) for b in batch]
                
                outputs = model(vision, tactile, proprio)
                pred = outputs['logits'].argmax(dim=1)
                
                correct += (pred == labels).sum().item()
                total += labels.shape[0]
                
                trust_scores.append(outputs['trust_fused'].mean().item())
        
        return {
            'accuracy': correct / total,
            'avg_trust': sum(trust_scores) / len(trust_scores)
        }
    
    def run_experiment(
        self,
        num_tasks: int = 3,
        epochs_per_task: int = 5
    ) -> Dict:
        """Run full multimodal CL experiment."""
        
        # Create data
        train_loaders, test_loaders = self.create_synthetic_multimodal_data(
            num_samples=500,
            num_tasks=num_tasks
        )
        
        # Create model
        model = MultimodalContinualLearner(
            embed_dim=256,
            num_classes=2
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Track results
        task_accuracies = []
        trust_evolution = []
        
        for t in range(num_tasks):
            print(f"\nTraining on task {t}...")
            
            # Train
            for epoch in range(epochs_per_task):
                loss = self.train_epoch(model, train_loaders[t], optimizer)
            
            # Evaluate on all tasks
            accs = []
            trusts = []
            for i in range(t + 1):
                metrics = self.evaluate(model, test_loaders[i])
                accs.append(metrics['accuracy'])
                trusts.append(metrics['avg_trust'])
                print(f"  Task {i}: acc={metrics['accuracy']:.4f}, trust={metrics['avg_trust']:.4f}")
            
            task_accuracies.append(accs)
            trust_evolution.append(trusts)
        
        return {
            'task_accuracies': task_accuracies,
            'trust_evolution': trust_evolution,
            'final_weights': model.trust_scorer.modality_weights.data.cpu().numpy()
        }


def main():
    """Run multimodal CL experiment."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    experiment = MultimodalCLExperiment(device=device)
    results = experiment.run_experiment(num_tasks=3, epochs_per_task=5)
    
    print("\n" + "="*60)
    print("MULTIMODAL CL RESULTS")
    print("="*60)
    
    for t, (accs, trusts) in enumerate(zip(results['task_accuracies'], results['trust_evolution'])):
        print(f"\nAfter task {t}:")
        for i, (acc, trust) in enumerate(zip(accs, trusts)):
            print(f"  Task {i}: acc={acc:.4f}, trust={trust:.4f}")
    
    print(f"\nFinal modality weights: {results['final_weights']}")


if __name__ == '__main__':
    main()
