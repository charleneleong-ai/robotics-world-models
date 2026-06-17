"""AnomalyDINO-style few-shot anomaly detection on MVTec-AD cable.

Training-free: build a memory bank of DINOv2 patch tokens from k normal images,
score each test image by its worst-patch nearest-neighbour cosine distance to the
bank, and report image-AUROC vs k. The point: foundation-model patch features give
strong few-shot AD where PatchCore's WideResNet 1-shot is weak (0.76).
"""
import glob
import json
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "/workspace/proj5/mvtec/cable"
DEFECTS = ["bent_wire", "cable_swap", "combined", "cut_inner_insulation",
           "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation"]
IMG = 518  # 37x37 patches at patch-size 14

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
tf = T.Compose([
    T.Resize((IMG, IMG)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def patches(path):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEV)
    feat = model.forward_features(x)["x_norm_patchtokens"][0]  # [N, 384]
    return torch.nn.functional.normalize(feat, dim=1)


@torch.no_grad()
def score(test_feat, bank):
    sim = test_feat @ bank.T          # cosine sim [N_test, M_bank]
    return (1.0 - sim.max(dim=1).values).max().item()  # worst-patch NN distance


def auroc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


normal = sorted(glob.glob(ROOT + "/train/good/*.png"))
random.seed(0)
random.shuffle(normal)
test_good = sorted(glob.glob(ROOT + "/test/good/*.png"))
test_anom = [p for d in DEFECTS for p in glob.glob(ROOT + "/test/" + d + "/*.png")]
print("test: %d good + %d anom" % (len(test_good), len(test_anom)), flush=True)

# cache test features once (independent of k)
tg = [patches(p) for p in test_good]
ta = [patches(p) for p in test_anom]

results = []
for k in [1, 2, 4, 8]:
    bank = torch.cat([patches(p) for p in normal[:k]], dim=0)
    scores = [score(f, bank) for f in tg] + [score(f, bank) for f in ta]
    labels = [0] * len(tg) + [1] * len(ta)
    a = auroc(labels, scores)
    results.append({"model": "AnomalyDINO(DINOv2-S/14)", "k": k, "auroc": round(float(a), 3)})
    print("DINO k=%d: AUROC=%.3f" % (k, a), flush=True)

json.dump(results, open("/workspace/proj5/anomalydino.json", "w"), indent=2)
print("=== ANOMALYDINO FEW-SHOT (MVTec cable) ===", flush=True)
for r in results:
    print("  k=%-2d AUROC=%s" % (r["k"], r["auroc"]), flush=True)
