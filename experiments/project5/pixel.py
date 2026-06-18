"""Pixel-level localization metrics for PatchCore on MVTec-AD cable.

Builds flat abnormal/ + mask/ dirs with matching basenames (so anomalib pairs each
defect image with its GT mask), then runs PatchCore and reports pixel-AUROC (+ AUPRO
if the evaluator API is available) alongside image-AUROC. Pixel metrics = "where the
defect is", not just whether one exists.
"""
import os
import glob
import json
import shutil

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

CABLE = "/workspace/proj5/mvtec/cable"
ROOT = "/workspace/proj5"
PX = ROOT + "/cable_px"
DEFECTS = ["bent_wire", "cable_swap", "combined", "cut_inner_insulation",
           "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation"]

shutil.rmtree(PX, ignore_errors=True)
os.makedirs(PX + "/abnormal")
os.makedirs(PX + "/mask")
n = 0
for d in DEFECTS:
    for img in sorted(glob.glob("%s/test/%s/*.png" % (CABLE, d))):
        stem = os.path.splitext(os.path.basename(img))[0]
        mask = "%s/ground_truth/%s/%s_mask.png" % (CABLE, d, stem)
        if not os.path.exists(mask):
            continue
        shutil.copy(img, "%s/abnormal/%s_%s.png" % (PX, d, stem))
        shutil.copy(mask, "%s/mask/%s_%s.png" % (PX, d, stem))
        n += 1
print("paired %d abnormal images with masks" % n, flush=True)

dm = Folder(
    name="cable", root=ROOT,
    normal_dir="mvtec/cable/train/good",
    abnormal_dir="cable_px/abnormal",
    normal_test_dir="mvtec/cable/test/good",
    mask_dir="cable_px/mask",
    train_batch_size=8, eval_batch_size=8, num_workers=4,
)


def run(model, tag):
    eng = Engine(max_epochs=1, accelerator="gpu", devices=1,
                 default_root_dir="/workspace/proj5/px_%s" % tag)
    eng.fit(model, datamodule=dm)
    return eng.test(model, datamodule=dm)


res = None
try:
    from anomalib.metrics import Evaluator, AUROC, AUPRO
    metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
    ]
    res = run(Patchcore(evaluator=Evaluator(test_metrics=metrics)), "eval")
    print("CUSTOM EVALUATOR OK (AUPRO included)", flush=True)
except Exception as e:
    print("custom-evaluator path failed: %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
    res = run(Patchcore(), "default")

print("RESULTS:", res, flush=True)
json.dump(res, open("/workspace/proj5/pixel_metrics.json", "w"), default=str, indent=2)
print("=== PIXEL LOCALIZATION (PatchCore, MVTec cable) ===", flush=True)
for k, v in (res[0].items() if res else []):
    print("  %s = %s" % (k, v), flush=True)
