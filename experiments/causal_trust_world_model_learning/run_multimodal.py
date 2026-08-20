#!/usr/bin/env python3
"""Multimodal CL experiment: MNIST vision + correlated tactile/proprio."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, json, time, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from cl_baselines import compute_cl_metrics

device = 'cpu'


class MultimodalEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.vision_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(), nn.Linear(64 * 16, embed_dim)
        )
        self.tactile_enc = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.proprio_enc = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, embed_dim))
        self.fusion = nn.Sequential(nn.Linear(embed_dim * 3, embed_dim), nn.ReLU())
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, vision, tactile, proprio, mask_modalities=None):
        v = self.vision_enc(vision)
        t = self.tactile_enc(tactile)
        p = self.proprio_enc(proprio)
        if mask_modalities is not None:
            if mask_modalities[0]: v = torch.zeros_like(v)
            if mask_modalities[1]: t = torch.zeros_like(t)
            if mask_modalities[2]: p = torch.zeros_like(p)
        fused = self.fusion(torch.cat([v, t, p], dim=-1))
        return self.classifier(fused)


class TrustAwareMultimodalCL:
    def __init__(self, model, trust_threshold=0.7):
        self.model = model
        self.trust_threshold = trust_threshold
        self.prev_models = []
        self.trust_scores = []

    def compute_trust(self, dataloader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in dataloader:
                x, t, p, y = batch
                out = self.model(x, t, p)
                correct += (out.argmax(1) == y).sum().item()
                total += y.shape[0]
        return correct / total if total > 0 else 0.0

    def consolidate(self, dataloader):
        trust = self.compute_trust(dataloader)
        self.prev_models.append(copy.deepcopy(self.model))
        self.trust_scores.append(trust)

    def penalty(self, vision, tactile, proprio):
        if not self.prev_models:
            return torch.tensor(0.0)
        pen = torch.tensor(0.0)
        for m, s in zip(self.prev_models, self.trust_scores):
            with torch.no_grad():
                old_out = m(vision, tactile, proprio)
            new_out = self.model(vision, tactile, proprio)
            pen = pen + s * F.kl_div(
                F.log_softmax(new_out / 2.0, dim=1),
                F.softmax(old_out / 2.0, dim=1),
                reduction='batchmean'
            )
        return pen


def create_multimodal_mnist(num_tasks=3, batch_size=32):
    """Create Split MNIST with correlated synthetic tactile/proprio."""
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    tasks_per_class = 10 // num_tasks
    train_loaders, test_loaders = [], []

    for t in range(num_tasks):
        sc = t * tasks_per_class
        ec = (t + 1) * tasks_per_class

        for split, dataset in [('train', full_train), ('test', full_test)]:
            indices = [i for i, (_, y) in enumerate(dataset) if sc <= y < ec]
            sub = torch.utils.data.Subset(dataset, indices)

            vision_list, tactile_list, proprio_list, label_list = [], [], [], []
            for i in range(len(sub)):
                img, y = sub[i]
                vision_list.append(img)
                # Correlated synthetic: tactile encodes digit mean, proprio encodes digit std
                digit = y - sc
                tactile = torch.randn(32) * 0.1 + float(digit) / tasks_per_class
                proprio = torch.randn(8) * 0.1 + float(digit) / tasks_per_class * 0.5
                tactile_list.append(tactile)
                proprio_list.append(proprio)
                label_list.append(torch.tensor(y % tasks_per_class))

            V = torch.stack(vision_list)
            T = torch.stack(tactile_list)
            P = torch.stack(proprio_list)
            Y = torch.stack(label_list)
            ds = TensorDataset(V, T, P, Y)

            if split == 'train':
                train_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
            else:
                test_loaders.append(DataLoader(ds, batch_size=batch_size))

    return train_loaders, test_loaders


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, t, p, y in loader:
            out = model(x, t, p)
            correct += (out.argmax(1) == y).sum().item()
            total += y.shape[0]
    return correct / total if total > 0 else 0.0


def evaluate_modality_drop(model, loader, drop_vision=False):
    """Evaluate with vision dropped (modality failure)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, t, p, y in loader:
            if drop_vision:
                x = torch.zeros_like(x)
            out = model(x, t, p)
            correct += (out.argmax(1) == y).sum().item()
            total += y.shape[0]
    return correct / total if total > 0 else 0.0


def run_experiment(method, train_loaders, test_loaders, num_tasks, epochs):
    model = MultimodalEncoder()

    if method == 'trust_aware':
        algo = TrustAwareMultimodalCL(model, trust_threshold=0.7)
    else:
        algo = None

    task_accs = []
    for t in range(num_tasks):
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        model.train()
        for ep in range(epochs):
            for batch in train_loaders[t]:
                x, tactile, proprio, y = batch
                opt.zero_grad()
                logits = model(x, tactile, proprio)
                loss = crit(logits, y)
                if algo and algo.prev_models:
                    loss = loss + algo.penalty(x, tactile, proprio)
                loss.backward()
                opt.step()

        if algo:
            algo.consolidate(train_loaders[t])

        accs = [evaluate(model, test_loaders[i]) for i in range(t + 1)]
        accs += [0.0] * (num_tasks - len(accs))
        task_accs.append(accs)

    transposed = list(map(list, zip(*task_accs)))
    return compute_cl_metrics(transposed, num_tasks)


def run_modality_drop_experiment(model, test_loaders, num_tasks):
    """Evaluate robustness to visual modality failure."""
    results = []
    for t in range(num_tasks):
        acc_normal = evaluate(model, test_loaders[t])
        acc_drop = evaluate_modality_drop(model, test_loaders[t], drop_vision=True)
        results.append({'normal': acc_normal, 'vision_dropped': acc_drop})
    return results


if __name__ == '__main__':
    print('MULTIMODAL CL EXPERIMENT')
    print('=' * 55)
    tl, vel = create_multimodal_mnist(num_tasks=3)

    all_results = {}
    for method in ['fine_tuning', 'trust_aware']:
        print(f'\n{method}...', flush=True)
        t0 = time.time()
        metrics = run_experiment(method, tl, vel, num_tasks=3, epochs=5)
        elapsed = time.time() - t0
        all_results[method] = {k: v for k, v in metrics.items()}
        avg = metrics['average_accuracy'][-1]
        bwt = metrics['backward_transfer'][-1] if metrics['backward_transfer'] else 0
        print(f'  AvgAcc={avg:.4f} BWT={bwt:.4f} ({elapsed:.0f}s)')

    # Modality drop test for trust_aware
    print('\nModality dropout test (trust_aware)...', flush=True)
    model = MultimodalEncoder()
    algo = TrustAwareMultimodalCL(model, trust_threshold=0.7)
    for t in range(3):
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        model.train()
        for ep in range(5):
            for batch in tl[t]:
                x, tactile, proprio, y = batch
                opt.zero_grad()
                loss = crit(model(x, tactile, proprio), y)
                if algo.prev_models:
                    loss = loss + algo.penalty(x, tactile, proprio)
                loss.backward()
                opt.step()
        algo.consolidate(tl[t])

    drop_results = run_modality_drop_experiment(model, vel, 3)
    all_results['modality_drop'] = drop_results

    for i, r in enumerate(drop_results):
        print(f'  Task {i}: normal={r["normal"]:.4f}, vision_dropped={r["vision_dropped"]:.4f}')

    with open('multimodal_cl_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print('\nDone.')
