#!/usr/bin/env python3
"""Quick CL experiment runner - reduced settings for fast results."""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, json, time
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cl_baselines import EWC, LwF, ExperienceReplay, TrustAwareCL, compute_cl_metrics
from cl_benchmarks import (
    SimpleMLP, SimpleCNN, create_split_mnist, create_permuted_mnist,
    train_model, evaluate_model
)

device = 'cpu'

def run_experiment(model_fn, train_loaders, test_loaders, num_tasks, epochs):
    results = {}
    for method_name in ['fine_tuning', 'ewc', 'lwf', 'experience_replay', 'trust_aware_cl']:
        print(f'{method_name}...', end=' ', flush=True)
        model = model_fn()
        task_accs = []

        if method_name == 'ewc':
            algo = EWC(model, lambda_ewc=5000)
        elif method_name == 'lwf':
            algo = LwF(model, lambda_lwf=2.0)
        elif method_name == 'experience_replay':
            algo = ExperienceReplay(buffer_size=500)
        elif method_name == 'trust_aware_cl':
            algo = TrustAwareCL(model, trust_threshold=0.7)
        else:
            algo = None

        for t in range(num_tasks):
            if method_name == 'fine_tuning':
                train_model(model, train_loaders[t], epochs=epochs, device=device)
            elif method_name == 'ewc':
                opt = torch.optim.Adam(model.parameters(), lr=0.001)
                crit = nn.CrossEntropyLoss()
                model.train()
                for ep in range(epochs):
                    for batch in train_loaders[t]:
                        x, y = batch[0].to(device), batch[1].to(device)
                        opt.zero_grad()
                        loss = crit(model(x), y) + algo.penalty()
                        loss.backward()
                        opt.step()
                algo.consolidate(train_loaders[t])
            elif method_name == 'lwf':
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
                algo.consolidate()
            elif method_name == 'experience_replay':
                cl = algo.get_current_task_loader(train_loaders[t], t)
                train_model(model, cl, epochs=epochs, device=device)
                for batch in train_loaders[t]:
                    algo.add_to_buffer(batch[0], batch[1], t)
            elif method_name == 'trust_aware_cl':
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
        results[method_name] = {k: v for k, v in metrics.items()}
        avg = metrics['average_accuracy'][-1]
        bwt = metrics['backward_transfer'][-1] if metrics['backward_transfer'] else 0
        fwt = metrics['forward_transfer'][-1] if metrics['forward_transfer'] else 0
        print(f'AvgAcc={avg:.4f} BWT={bwt:.4f} FWT={fwt:.4f}')

    return results

if __name__ == '__main__':
    all_results = {}

    print('=== SPLIT MNIST (5 tasks, 5 epochs) ===')
    t0 = time.time()
    tl, vel = create_split_mnist(num_tasks=5)
    all_results['split_mnist'] = run_experiment(
        lambda: SimpleMLP(input_size=784, num_classes=2),
        tl, vel, 5, 5
    )
    print(f'  ({time.time()-t0:.0f}s)\n')

    print('=== PERMUTED MNIST (5 tasks, 5 epochs) ===')
    t0 = time.time()
    tl, vel = create_permuted_mnist(num_tasks=5)
    all_results['permuted_mnist'] = run_experiment(
        lambda: SimpleMLP(input_size=784, num_classes=10),
        tl, vel, 5, 5
    )
    print(f'  ({time.time()-t0:.0f}s)\n')

    print('=== SPLIT CIFAR-10 (5 tasks, 10 epochs) ===')
    t0 = time.time()
    from run_split_cifar10 import create_split_cifar10
    tl, vel = create_split_cifar10(num_tasks=5)
    all_results['split_cifar10'] = run_experiment(
        lambda: SimpleCNN(num_classes=2, input_channels=3),
        tl, vel, 5, 10
    )
    print(f'  ({time.time()-t0:.0f}s)\n')

    with open('cl_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print('=== SUMMARY ===')
    for bench, methods in all_results.items():
        print(f'\n{bench}:')
        print(f'{"Method":<20} {"AvgAcc":>10} {"BWT":>10} {"FWT":>10}')
        print('-'*50)
        for m, d in methods.items():
            avg = d['average_accuracy'][-1]
            bwt = d['backward_transfer'][-1] if d['backward_transfer'] else 0
            fwt = d['forward_transfer'][-1] if d['forward_transfer'] else 0
            print(f'{m:<20} {avg:>10.4f} {bwt:>10.4f} {fwt:>10.4f}')
