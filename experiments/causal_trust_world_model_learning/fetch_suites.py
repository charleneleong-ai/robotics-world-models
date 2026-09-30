"""Restore the libero_object and libero_goal suites, which vanished from disk.

libero_spatial survives as a symlink into the HuggingFace cache; these two were
removed. Re-download them from the same dataset revision and link them the same
way, so the three suites resolve identically.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "yifengzhu-hf/LIBERO-datasets"
LIBERO = Path("/home/ubuntu/robotics_world_models/LIBERO")
SUITES = ["libero_object", "libero_goal"]

path = Path(snapshot_download(
    REPO, repo_type="dataset",
    allow_patterns=[f"{s}/*" for s in SUITES],
))
print(f"snapshot: {path}", flush=True)

for suite in SUITES:
    link, target = LIBERO / suite, path / suite
    n = len(list(target.glob("*.hdf5")))
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)
    print(f"  {suite}: {n} hdf5 files -> {target}", flush=True)
print("done", flush=True)
