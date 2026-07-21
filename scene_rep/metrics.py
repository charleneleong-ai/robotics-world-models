"""Held-out-view eval metrics for the semantic 3DGS scene-representation project.

Photometric (PSNR/SSIM/LPIPS) scores novel-view synthesis; semantic (mIoU/PQ)
scores segmentation rendered from held-out views — the differentiator. All
metrics are pure-numpy and stateless except LPIPS, which needs a learned net
(torch + the `lpips` package) and is imported lazily so the rest of the harness
runs with numpy alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------------------------------------------------------- #
# photometric — novel-view synthesis quality
# ----------------------------------------------------------------------------- #


def _to_float(img: np.ndarray, data_range: float) -> np.ndarray:
    return np.asarray(img, dtype=np.float64) / data_range


def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    pred, gt = _to_float(pred, data_range), _to_float(gt, data_range)
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g1 = np.exp(-(ax**2) / (2.0 * sigma**2))
    g1 /= g1.sum()
    return np.outer(g1, g1)


def _windowed_mean(chan: np.ndarray, window: np.ndarray) -> np.ndarray:
    # Gaussian-weighted local mean over valid positions (no padding).
    patches = sliding_window_view(chan, window.shape)
    return np.tensordot(patches, window, axes=([2, 3], [0, 1]))


def ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Gaussian-window SSIM (Wang et al. 2004), per-channel then averaged."""
    pred, gt = _to_float(pred, data_range), _to_float(gt, data_range)
    if pred.ndim == 2:
        pred, gt = pred[..., None], gt[..., None]
    if min(pred.shape[:2]) < win_size:
        raise ValueError(f"image {pred.shape[:2]} smaller than SSIM window {win_size}")

    window = _gaussian_window(win_size, sigma)
    c1, c2 = 0.01**2, 0.03**2  # data already normalized to [0, 1]

    scores = []
    for c in range(pred.shape[2]):
        x, y = pred[..., c], gt[..., c]
        mu_x, mu_y = _windowed_mean(x, window), _windowed_mean(y, window)
        mu_xx, mu_yy, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
        var_x = _windowed_mean(x * x, window) - mu_xx
        var_y = _windowed_mean(y * y, window) - mu_yy
        cov_xy = _windowed_mean(x * y, window) - mu_xy
        smap = ((2 * mu_xy + c1) * (2 * cov_xy + c2)) / (
            (mu_xx + mu_yy + c1) * (var_x + var_y + c2)
        )
        scores.append(smap.mean())
    return float(np.mean(scores))


def lpips(pred: np.ndarray, gt: np.ndarray, net: str = "alex", data_range: float = 1.0) -> float:
    """Learned perceptual distance (lower = better). Needs torch + `lpips`.

    Heavy optional dep loaded lazily (per project import policy): on the box its
    weights download once; locally the module still imports without torch.
    """
    try:
        import lpips as lpips_lib  # noqa: PLC0415 — optional heavy dep, see docstring
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        raise ModuleNotFoundError(
            "lpips() needs torch + the `lpips` package (install in the box env)"
        ) from exc

    def _chw(img: np.ndarray) -> "torch.Tensor":
        a = _to_float(img, data_range)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float()
        return t * 2.0 - 1.0  # lpips expects [-1, 1]

    model = lpips_lib.LPIPS(net=net)
    with torch.no_grad():
        return float(model(_chw(pred), _chw(gt)).item())


# ----------------------------------------------------------------------------- #
# semantic — segmentation rendered from held-out views (the differentiator)
# ----------------------------------------------------------------------------- #


def confusion_matrix(
    pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int | None = None
) -> np.ndarray:
    pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()
    if ignore_index is not None:
        keep = gt != ignore_index
        pred, gt = pred[keep], gt[keep]
    idx = num_classes * gt.astype(np.int64) + pred.astype(np.int64)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=1) + cm.sum(axis=0) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / union
    return np.where(union > 0, iou, np.nan)  # absent classes -> nan, excluded from mean


def mean_iou(
    pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int | None = None
) -> float:
    iou = per_class_iou(confusion_matrix(pred, gt, num_classes, ignore_index))
    return float(np.nanmean(iou))


def panoptic_quality(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.5,
    ignore_id: int = 0,
) -> dict[str, float]:
    """Per-image panoptic quality on integer segment-id maps (PQ = SQ x RQ).

    Segments are matched greedily by IoU; a match with IoU > threshold is a TP.
    `ignore_id` (default 0) marks unlabeled pixels, excluded from matching.
    """
    pred, gt = np.asarray(pred), np.asarray(gt)
    gt_ids = [i for i in np.unique(gt) if i != ignore_id]
    pred_ids = [i for i in np.unique(pred) if i != ignore_id]

    pred_masks = {p: pred == p for p in pred_ids}
    matched_pred: set[int] = set()
    tp, iou_sum = 0, 0.0
    for g in gt_ids:
        gm = gt == g
        best_iou, best_p = 0.0, None
        for p in pred_ids:
            if p in matched_pred:
                continue
            inter = np.logical_and(gm, pred_masks[p]).sum()
            if inter == 0:
                continue
            union = np.logical_or(gm, pred_masks[p]).sum()
            iou = inter / union
            if iou > best_iou:
                best_iou, best_p = iou, p
        if best_iou > iou_threshold:
            tp += 1
            iou_sum += best_iou
            matched_pred.add(best_p)

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - tp
    sq = iou_sum / tp if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


# ----------------------------------------------------------------------------- #
# aggregation — one combined table across held-out views
# ----------------------------------------------------------------------------- #


@dataclass
class ViewMetrics:
    view: str
    psnr: float
    ssim: float
    miou: float
    lpips: float | None = None


def combined_table(rows: list[ViewMetrics]) -> str:
    if not rows:
        raise ValueError("no views to tabulate")
    has_lpips = all(r.lpips is not None for r in rows)
    header = ["view", "PSNR", "SSIM", *(["LPIPS"] if has_lpips else []), "mIoU"]

    def fmt(r: ViewMetrics) -> list[str]:
        cells = [r.view, f"{r.psnr:.2f}", f"{r.ssim:.3f}"]
        if has_lpips:
            cells.append(f"{r.lpips:.3f}")
        cells.append(f"{r.miou:.3f}")
        return cells

    mean = ViewMetrics(
        view="**mean**",
        psnr=float(np.mean([r.psnr for r in rows])),
        ssim=float(np.mean([r.ssim for r in rows])),
        miou=float(np.mean([r.miou for r in rows])),
        lpips=float(np.mean([r.lpips for r in rows])) if has_lpips else None,
    )
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
        *["| " + " | ".join(fmt(r)) + " |" for r in rows],
        "| " + " | ".join(fmt(mean)) + " |",
    ]
    return "\n".join(lines)
