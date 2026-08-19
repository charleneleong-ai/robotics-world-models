"""
Continual Learning Benchmarks

Standard CL datasets:
- Split MNIST
- Permuted MNIST
- Split CIFAR-10
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict
import numpy as np


def create_split_mnist(
    num_tasks: int = 5,
    batch_size: int = 32,
    data_dir: str = './data'
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Create Split MNIST benchmark.
    
    Each task: binary classification of 2 digits.
    Task 0: 0 vs 1
    Task 1: 2 vs 3
    Task 2: 4 vs 5
    Task 3: 6 vs 7
    Task 4: 8 vs 9
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full MNIST
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Create task splits
    train_loaders = []
    test_loaders = []
    
    for task_id in range(num_tasks):
        # Get class indices for this task
        class1 = 2 * task_id
        class2 = 2 * task_id + 1
        
        # Filter training data
        train_indices = [i for i, (x, y) in enumerate(train_dataset) if y in [class1, class2]]
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        
        # Remap labels to 0/1
        train_subset = RemapLabels(train_subset, {class1: 0, class2: 1})
        
        # Filter test data
        test_indices = [i for i, (x, y) in enumerate(test_dataset) if y in [class1, class2]]
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        test_subset = RemapLabels(test_subset, {class1: 0, class2: 1})
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


def create_permuted_mnist(
    num_tasks: int = 10,
    batch_size: int = 32,
    data_dir: str = './data'
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Create Permuted MNIST benchmark.
    
    Each task: same 10 classes but with different random permutation.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full MNIST
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Convert to tensors
    train_x = torch.stack([x for x, y in train_dataset])
    train_y = torch.tensor([y for x, y in train_dataset])
    test_x = torch.stack([x for x, y in test_dataset])
    test_y = torch.tensor([y for x, y in test_dataset])
    
    # Flatten images
    train_x = train_x.view(train_x.shape[0], -1)
    test_x = test_x.view(test_x.shape[0], -1)
    
    # Create permutations
    train_loaders = []
    test_loaders = []
    
    for task_id in range(num_tasks):
        # Create random permutation
        perm = torch.randperm(784)
        
        # Apply permutation
        train_x_perm = train_x[:, perm]
        test_x_perm = test_x[:, perm]
        
        # Create data loaders
        train_dataset_perm = TensorDataset(train_x_perm, train_y)
        test_dataset_perm = TensorDataset(test_x_perm, test_y)
        
        train_loader = DataLoader(train_dataset_perm, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset_perm, batch_size=batch_size, shuffle=False)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


def create_split_cifar10(
    num_tasks: int = 5,
    batch_size: int = 32,
    data_dir: str = './data'
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Create Split CIFAR-10 benchmark.
    
    Each task: binary classification of 2 classes.
    Task 0: airplane vs automobile
    Task 1: bird vs cat
    Task 2: deer vs dog
    Task 3: frog vs horse
    Task 4: ship vs truck
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load full CIFAR-10
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    
    # Create task splits
    train_loaders = []
    test_loaders = []
    
    for task_id in range(num_tasks):
        # Get class indices for this task
        class1 = 2 * task_id
        class2 = 2 * task_id + 1
        
        # Filter training data
        train_indices = [i for i, (x, y) in enumerate(train_dataset) if y in [class1, class2]]
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        
        # Remap labels to 0/1
        train_subset = RemapLabels(train_subset, {class1: 0, class2: 1})
        
        # Filter test data
        test_indices = [i for i, (x, y) in enumerate(test_dataset) if y in [class1, class2]]
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        test_subset = RemapLabels(test_subset, {class1: 0, class2: 1})
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


class RemapLabels:
    """Remap labels in a dataset."""
    
    def __init__(self, dataset, label_map: Dict[int, int]):
        self.dataset = dataset
        self.label_map = label_map
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, self.label_map[y]


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST/CIFAR-10."""
    
    def __init__(self, num_classes: int = 2, input_channels: int = 1):
        super().__init__()
        self.input_channels = input_channels
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        if input_channels == 1:
            feat_size = 64 * 7 * 7
        else:
            feat_size = 64 * 8 * 8
            
        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) != self.input_channels:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3 and self.input_channels == 1:
            x = x.unsqueeze(1)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for Permuted MNIST."""
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    device: str = 'cpu'
) -> float:
    """Train model on a single task."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    
    return correct / total


def run_benchmark(
    benchmark_name: str = 'split_mnist',
    num_tasks: int = 5,
    method: str = 'fine_tuning',
    epochs_per_task: int = 10,
    device: str = 'cpu'
) -> Dict:
    """Run a full CL benchmark."""
    
    # Create benchmark
    if benchmark_name == 'split_mnist':
        train_loaders, test_loaders = create_split_mnist(num_tasks=num_tasks)
        model = SimpleCNN(num_classes=2).to(device)
    elif benchmark_name == 'permuted_mnist':
        train_loaders, test_loaders = create_permuted_mnist(num_tasks=num_tasks)
        model = SimpleMLP(input_size=784, num_classes=10).to(device)
    elif benchmark_name == 'split_cifar10':
        train_loaders, test_loaders = create_split_cifar10(num_tasks=num_tasks)
        model = SimpleCNN(num_classes=2).to(device)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    # Run experiment
    task_accuracies = []
    
    for t in range(num_tasks):
        # Train on current task
        train_model(model, train_loaders[t], epochs=epochs_per_task, device=device)
        
        # Evaluate on all tasks
        accs = []
        for i in range(t + 1):
            acc = evaluate_model(model, test_loaders[i], device=device)
            accs.append(acc)
        
        # Pad with zeros for future tasks
        accs.extend([0.0] * (num_tasks - len(accs)))
        task_accuracies.append(accs)
    
    # Transpose to [task][time]
    task_accuracies_transposed = list(map(list, zip(*task_accuracies)))
    
    return {
        'benchmark': benchmark_name,
        'method': method,
        'task_accuracies': task_accuracies_transposed,
        'num_tasks': num_tasks
    }
