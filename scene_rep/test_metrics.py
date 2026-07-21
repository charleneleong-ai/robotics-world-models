"""Known-answer tests for the held-out-view eval harness."""

from __future__ import annotations

import numpy as np
import pytest

from scene_rep import metrics
from scene_rep.metrics import ViewMetrics


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _img(rng: np.random.Generator, h: int = 32, w: int = 32) -> np.ndarray:
    return rng.random((h, w, 3))


class TestPSNR:
    """Peak signal-to-noise ratio on [0,1] and uint8 inputs."""

    def test_identical_is_infinite(self, rng):
        x = _img(rng)
        assert metrics.psnr(x, x) == float("inf")

    def test_known_mse_gives_known_db(self):
        gt = np.full((8, 8, 3), 0.5)
        pred = gt + 0.1  # MSE = 0.01 -> 10*log10(1/0.01) = 20 dB
        assert metrics.psnr(pred, gt) == pytest.approx(20.0)

    def test_uint8_data_range(self):
        gt = np.full((8, 8), 128, dtype=np.uint8)
        pred = np.full((8, 8), 128 + 25, dtype=np.uint8)  # 25/255 diff
        expected = 10 * np.log10(1.0 / (25 / 255) ** 2)
        assert metrics.psnr(pred, gt, data_range=255) == pytest.approx(expected)

    def test_more_noise_is_lower(self, rng):
        gt = _img(rng)
        near = np.clip(gt + rng.normal(0, 0.02, gt.shape), 0, 1)
        far = np.clip(gt + rng.normal(0, 0.2, gt.shape), 0, 1)
        assert metrics.psnr(near, gt) > metrics.psnr(far, gt)


class TestSSIM:
    """Gaussian-window SSIM properties + bounds."""

    def test_identical_is_one(self, rng):
        x = _img(rng)
        assert metrics.ssim(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_degraded_below_identical_and_bounded(self, rng):
        gt = _img(rng)
        noisy = np.clip(gt + rng.normal(0, 0.15, gt.shape), 0, 1)
        s = metrics.ssim(noisy, gt)
        assert s < 1.0
        assert -1.0 <= s <= 1.0

    def test_grayscale_supported(self, rng):
        x = rng.random((32, 32))
        assert metrics.ssim(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_raises_when_smaller_than_window(self):
        small = np.zeros((5, 5, 3))
        with pytest.raises(ValueError, match="smaller than SSIM window"):
            metrics.ssim(small, small)


class TestSemanticIoU:
    """Confusion-matrix mIoU with absent-class and ignore handling."""

    def test_perfect_prediction(self):
        gt = np.array([0, 0, 1, 1, 2, 2])
        assert metrics.mean_iou(gt, gt, num_classes=3) == pytest.approx(1.0)

    def test_disjoint_prediction_is_zero(self):
        gt = np.zeros(6, dtype=int)
        pred = np.ones(6, dtype=int)
        # only classes 0 and 1 appear (each with IoU 0); class 2 absent -> nan-excluded
        assert metrics.mean_iou(pred, gt, num_classes=3) == pytest.approx(0.0)

    def test_hand_computed_partial(self):
        gt = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 1, 1])  # class0 IoU=1/2, class1 IoU=2/3
        assert metrics.mean_iou(pred, gt, num_classes=2) == pytest.approx((0.5 + 2 / 3) / 2)

    def test_absent_class_excluded_from_mean(self):
        gt = np.array([0, 0, 1, 1])  # class 2 never present
        assert metrics.mean_iou(gt, gt, num_classes=3) == pytest.approx(1.0)

    def test_ignore_index_drops_pixels(self):
        gt = np.array([0, 0, 255, 255])
        pred = np.array([0, 0, 1, 1])  # the 255 pixels would be wrong if not ignored
        assert metrics.mean_iou(pred, gt, num_classes=2, ignore_index=255) == pytest.approx(1.0)


class TestPanopticQuality:
    """PQ = SQ x RQ with greedy IoU>0.5 segment matching."""

    def test_perfect_match_is_one(self):
        seg = np.array([[1, 1, 0], [1, 1, 0], [2, 2, 0]])
        out = metrics.panoptic_quality(seg, seg)
        assert out["pq"] == pytest.approx(1.0)
        assert (out["tp"], out["fp"], out["fn"]) == (2, 0, 0)

    def test_one_tp_one_fp_one_fn(self):
        gt = np.array([[1, 1, 0], [1, 1, 0], [2, 2, 0]])
        pred = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 3]])
        # gt1 vs pred1: IoU 4/5=0.8 -> TP; gt2 unmatched -> FN; pred3 spurious -> FP
        out = metrics.panoptic_quality(gt, pred)
        assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 1)
        assert out["sq"] == pytest.approx(0.8)
        assert out["rq"] == pytest.approx(0.5)
        assert out["pq"] == pytest.approx(0.4)

    def test_below_threshold_match_is_miss(self):
        gt = np.array([[1, 1, 1, 1]])
        pred = np.array([[2, 2, 0, 0]])  # IoU 2/4=0.5, not > 0.5
        out = metrics.panoptic_quality(gt, pred)
        assert (out["tp"], out["fp"], out["fn"]) == (0, 1, 1)


class TestCombinedTable:
    """Markdown table aggregation across held-out views."""

    def test_appends_mean_row(self):
        rows = [ViewMetrics("v0", 27.0, 0.80, 0.70), ViewMetrics("v1", 29.0, 0.90, 0.80)]
        table = metrics.combined_table(rows)
        assert "**mean**" in table
        assert "| 28.00 | 0.850 | 0.750 |" in table  # column-wise means

    def test_lpips_column_only_when_all_present(self):
        with_lpips = [ViewMetrics("v0", 27.0, 0.8, 0.7, lpips=0.2)]
        without = [ViewMetrics("v0", 27.0, 0.8, 0.7)]
        assert "LPIPS" in metrics.combined_table(with_lpips)
        assert "LPIPS" not in metrics.combined_table(without)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no views"):
            metrics.combined_table([])


class TestLPIPS:
    """Perceptual distance — gated on the optional torch + lpips deps."""

    def test_identical_near_zero_and_ordered(self, rng):
        pytest.importorskip("torch")
        pytest.importorskip("lpips")
        gt = _img(rng, 64, 64)
        noisy = np.clip(gt + rng.normal(0, 0.2, gt.shape), 0, 1)
        assert metrics.lpips(gt, gt) < metrics.lpips(noisy, gt)

    def test_missing_deps_message(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *a, **k):
            if name in {"torch", "lpips"}:
                raise ModuleNotFoundError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(ModuleNotFoundError, match="needs torch"):
            metrics.lpips(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))
