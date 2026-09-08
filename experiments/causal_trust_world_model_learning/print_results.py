import json

with open("results_maniskill_backbone_sweep/aggregated.json") as f:
    agg = json.load(f)

trust_methods = ["none", "ema", "multi_step", "ensemble"]
header = "backbone".ljust(12) + " " + " ".join(t.rjust(16) for t in trust_methods)
print(header)
for b, methods in agg.items():
    row = b.ljust(12) + " "
    for t in trust_methods:
        m = methods[t]["mean"]
        s = methods[t]["std"]
        row += f"{m:.4f}+-{s:.4f}  "
    print(row)
