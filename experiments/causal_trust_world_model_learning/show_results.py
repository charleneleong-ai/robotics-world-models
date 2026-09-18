import json
d = json.load(open("results_cl_baselines_full/aggregated.json"))
print(f"{'Method':<22}{'AvgAcc':>16}{'BWT':>16}{'FWT':>16}")
for name, m in sorted(d.items(), key=lambda x: -x[1]["avg_accuracy_mean"]):
    acc = "%.3f +/- %.3f" % (m["avg_accuracy_mean"], m["avg_accuracy_std"])
    bwt = "%.3f +/- %.3f" % (m["bwt_mean"], m["bwt_std"])
    fwt = "%.3f +/- %.3f" % (m["fwt_mean"], m["fwt_std"])
    print(f"{name:<22}{acc:>16}{bwt:>16}{fwt:>16}")
