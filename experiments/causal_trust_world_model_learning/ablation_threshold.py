#!/usr/bin/env python3
"""Ablation: trust threshold sensitivity on Split MNIST."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, json, time
import torch.nn as nn
from cl_baselines import TrustAwareCL, compute_cl_metrics
from cl_benchmarks import SimpleMLP, create_split_mnist, train_model, evaluate_model

device = 'cpu'

def run_ablation(threshold, train_loaders, test_loaders, num_tasks=5, epochs=5):
    model = SimpleMLP(input_size=784, num_classes=2)
    algo = TrustAwareCL(model, trust_threshold=threshold)
    task_accs = []

    for t in range(num_tasks):
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        model.train()
        for ep in range(epochs):
            for batch in train_loaders[t]:
                x, y = batch[0].to(device), batch[1].to(device)
                opt.zero_grad()
                loss = crit(model(x), y) + algo.penalty(x)
                loss.backward()
                opt.step()
        algo.consolidate(train_loaders[t])

        accs = [evaluate_model(model, test_loaders[i], device=device) for i in range(t+1)]
        accs.extend([0.0] * (num_tasks - len(accs)))
        task_accs.append(accs)

    transposed = list(map(list, zip(*task_accs)))
    metrics = compute_cl_metrics(transposed, num_tasks)
    return {
        'avg_acc': metrics['average_accuracy'][-1],
        'bwt': metrics['backward_transfer'][-1] if metrics['backward_transfer'] else 0,
        'fwt': metrics['forward_transfer'][-1] if metrics['forward_transfer'] else 0,
    }

if __name__ == '__main__':
    thresholds = [0.3, 0.5, 0.7, 0.9]
    results = {}

    print('Ablation: Trust Threshold Sensitivity (Split MNIST)')
    print('='*55)
    tl, vel = create_split_mnist(num_tasks=5)

    for th in thresholds:
        print(f'threshold={th}...', end=' ', flush=True)
        t0 = time.time()
        r = run_ablation(th, tl, vel)
        elapsed = time.time() - t0
        results[str(th)] = r
        print(f'AvgAcc={r["avg_acc"]:.4f} BWT={r["bwt"]:.4f} FWT={r["fwt"]:.4f} ({elapsed:.0f}s)')

    with open('ablation_threshold.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"Threshold":<12} {"AvgAcc":>10} {"BWT":>10} {"FWT":>10}')
    print('-'*45)
    for th, r in results.items():
        print(f'{th:<12} {r["avg_acc"]:>10.4f} {r["bwt"]:>10.4f} {r["fwt"]:>10.4f}')
