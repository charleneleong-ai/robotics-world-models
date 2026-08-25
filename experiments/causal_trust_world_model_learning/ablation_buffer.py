#!/usr/bin/env python3
"""Ablation: memory buffer size sensitivity on Split MNIST."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, json, time
import torch.nn as nn
from cl_baselines import ExperienceReplay, TrustAwareCL, compute_cl_metrics
from cl_benchmarks import SimpleMLP, create_split_mnist, train_model, evaluate_model

device = 'cpu'

def run_buffer_ablation(buffer_size, train_loaders, test_loaders, num_tasks=5, epochs=5):
    model = SimpleMLP(input_size=784, num_classes=2)
    algo = ExperienceReplay(buffer_size=buffer_size)
    task_accs = []

    for t in range(num_tasks):
        cl = algo.get_current_task_loader(train_loaders[t], t)
        train_model(model, cl, epochs=epochs, device=device)
        for batch in train_loaders[t]:
            algo.add_to_buffer(batch[0], batch[1], t)

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
    buffer_sizes = [100, 250, 500, 1000]
    results = {}

    print('Ablation: Buffer Size Sensitivity (Split MNIST)')
    print('='*55)
    tl, vel = create_split_mnist(num_tasks=5)

    for bs in buffer_sizes:
        print(f'buffer_size={bs}...', end=' ', flush=True)
        t0 = time.time()
        r = run_buffer_ablation(bs, tl, vel)
        elapsed = time.time() - t0
        results[str(bs)] = r
        print(f'AvgAcc={r["avg_acc"]:.4f} BWT={r["bwt"]:.4f} FWT={r["fwt"]:.4f} ({elapsed:.0f}s)')

    with open('ablation_buffer.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"BufferSize":<12} {"AvgAcc":>10} {"BWT":>10} {"FWT":>10}')
    print('-'*45)
    for bs, r in results.items():
        print(f'{bs:<12} {r["avg_acc"]:>10.4f} {r["bwt"]:>10.4f} {r["fwt"]:>10.4f}')
