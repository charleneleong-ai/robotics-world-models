import json
import numpy as np
from scipy import stats

d = json.load(open("task_order_mlp_ema.json"))
raw_with = np.array(d["raw_errors_with_trust"])
raw_without = np.array(d["raw_errors_without_trust"])
print("raw_with shape:", raw_with.shape)
print("raw_without shape:", raw_without.shape)

# Correct: cumsum along the task axis (axis=1) for BOTH conditions
cum_with = np.mean(np.cumsum(raw_with, axis=1), axis=0)
cum_without = np.mean(np.cumsum(raw_without, axis=1), axis=0)

print("Corrected cum_with (first/mid/last):", cum_with[0], cum_with[4], cum_with[-1])
print("Corrected cum_without (first/mid/last):", cum_without[0], cum_without[4], cum_without[-1])
reduction_pct = (cum_without[-1] - cum_with[-1]) / cum_without[-1] * 100
print(f"Corrected total error reduction at final task: {reduction_pct:.2f}%")

# Significance test on final-task cumulative error, paired by ordering x seed row
final_with = np.cumsum(raw_with, axis=1)[:, -1]
final_without = np.cumsum(raw_without, axis=1)[:, -1]
t, p = stats.ttest_rel(final_without, final_with)
print(f"Paired t-test on final cumulative error: t={t:.3f}, p={p:.4f}")
