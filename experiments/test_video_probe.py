"""Tests for the video-backbone audit: shared frame/cache/probe helpers and the sequential-task probe."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from continual_probe import ElasticPenalty, ReplayBuffer, SequenceResult
from probe_common import (
    DEVICE,
    ActionHead,
    LiberoSuite,
    extract_suite,
    fit_probe,
    held_out_mask,
    lagged_frames,
    load_task,
)

STRENGTH = 3.0
Frames = tuple[torch.Tensor, torch.Tensor]
DEMO_LENGTHS = {"demo_0": 5, "demo_1": 6, "demo_10": 4, "demo_2": 7}


@pytest.fixture
def net() -> ActionHead:
    torch.manual_seed(0)
    return ActionHead(2, 1, width=2).to(DEVICE)


@pytest.fixture
def batch() -> Frames:
    torch.manual_seed(1)
    return torch.randn(8, 2, device=DEVICE), torch.randn(8, 1, device=DEVICE)


@pytest.fixture
def suite(tmp_path: Path) -> LiberoSuite:
    """One task whose demos are tagged so every array says which demo and frame it came from.

    Each demo has one extra rgb frame, so trimming to the shortest stream is observable.
    """
    task_dir = tmp_path / "libero_toy"
    task_dir.mkdir()
    with h5py.File(task_dir / "task_a.hdf5", "w") as h:
        h.create_dataset("data/attrs_not_a_demo", data=np.zeros(1))
        for key, n in DEMO_LENGTHS.items():
            tag = int(key.split("_")[1])
            h[f"data/{key}/obs/agentview_rgb"] = np.full((n + 1, 4, 4, 3), tag, np.uint8)
            h[f"data/{key}/actions"] = np.full((n, 7), tag, np.float32)
            for i, part in enumerate(("ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states")):
                h[f"data/{key}/obs/{part}"] = np.full((n, 1), 10 * tag + i, np.float32)
    return LiberoSuite("toy", n_tasks=1, max_demos=3, root=tmp_path)


def consolidated(net: ActionHead, batch: Frames, strength: float = STRENGTH) -> ElasticPenalty:
    penalty = ElasticPenalty(strength)
    penalty.observe(net, *batch, batch=len(batch[0]))
    return penalty


def shift(net: ActionHead) -> None:
    with torch.no_grad():
        for param in net.parameters():
            param.add_(0.1)


class TestLiberoSuite:
    """Reading demonstrations in the order every cache was written in."""

    def test_demos_follow_sorted_key_order_and_stop_at_max(self, suite: LiberoSuite) -> None:
        """Lexicographic: demo_10 sorts before demo_2, so it is in the first three and demo_2 is not."""
        demos = list(suite.demos(suite.task_files[0]))
        assert [int(d.action[0, 0]) for d in demos] == [0, 1, 10]
        assert [d.index for d in demos] == [0, 1, 2]

    def test_streams_are_trimmed_to_the_shortest(self, suite: LiberoSuite) -> None:
        for demo in suite.demos(suite.task_files[0]):
            assert len(demo.rgb) == len(demo.action) == len(demo.state)

    def test_state_concatenates_parts_in_order(self, suite: LiberoSuite) -> None:
        demo = next(suite.demos(suite.task_files[0]))
        assert demo.state[0].tolist() == [0, 1, 2, 3, 4]


class TestExtractSuite:
    """One encoder pass feeding several caches that must stay frame-aligned."""

    def test_every_cache_holds_the_same_frames(self, suite: LiberoSuite, tmp_path: Path) -> None:
        out = {"a": tmp_path / "a", "b": tmp_path / "b"}
        logged: list[dict[str, float]] = []
        total = extract_suite(suite, lambda rgb: {"a": rgb[:, 0, 0, :1], "b": -rgb[:, 0, 0, :2].astype(np.float32)},
                              out, "toy", logged.append)
        task = load_task(out, 0)
        assert total == sum(list(DEMO_LENGTHS.values())[:3]) == len(task.action)
        assert task.demo.tolist() == [0] * 5 + [1] * 6 + [2] * 4
        assert np.array_equal(task.features["a"][:, 0], task.action[:, 0])
        assert np.array_equal(task.features["b"][:, 0], -task.action[:, 0])
        assert logged[0]["frames"] == total


class TestLoadTask:
    @pytest.mark.parametrize(("demo_b", "latents_b", "error"), [
        ([0, 1, 1], 3, "not aligned"),
        ([0, 0, 1], 2, "2 latents for 3 frames"),
    ], ids=["different_demos", "truncated_latents"])
    def test_inconsistent_caches_are_refused(
        self, tmp_path: Path, demo_b: list[int], latents_b: int, error: str,
    ) -> None:
        for name, demo, n_latents in (("a", [0, 0, 1], 3), ("b", demo_b, latents_b)):
            (tmp_path / name).mkdir()
            np.savez(tmp_path / name / "task00.npz", latent=np.zeros((n_latents, 2)), action=np.zeros((3, 7)),
                     state=np.zeros((3, 21)), demo=np.array(demo))
        with pytest.raises(ValueError, match=error):
            load_task({"a": tmp_path / "a", "b": tmp_path / "b"}, 0)


class TestFrameHelpers:
    @pytest.mark.parametrize(("lags", "expected"), [
        ([2, 1, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]]),
        ([3, 0], [[0, 0], [0, 1], [0, 2], [0, 3]]),
    ])
    def test_lagged_frames_clamp_at_the_first_frame(self, lags: list[int], expected: list[list[int]]) -> None:
        assert lagged_frames(4, lags).tolist() == expected

    def test_held_out_mask_takes_whole_last_demos(self) -> None:
        demo = np.array([0, 0, 1, 1, 1, 2, 3, 3])
        assert held_out_mask(demo, 2).tolist() == [False] * 5 + [True] * 3


class TestFitProbe:
    def test_recovers_a_linear_policy_on_held_out_frames(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 5)).astype(np.float32)
        y = (x @ rng.normal(size=(5, 2))).astype(np.float32)
        mse = fit_probe(x[:500], y[:500], x[500:], y[500:], seed=0)
        assert mse < 0.05 * y[500:].var()

    def test_seeded_runs_are_identical(self) -> None:
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(64, 3)).astype(np.float32), rng.normal(size=(64, 1)).astype(np.float32)
        assert fit_probe(x, y, x, y, seed=3, epochs=2) == fit_probe(x, y, x, y, seed=3, epochs=2)


class TestElasticPenalty:
    """Diagonal-Fisher EWC."""

    @pytest.mark.parametrize("strength", [None, 0.0])
    def test_no_term_until_a_task_is_consolidated_at_nonzero_strength(
            self, net: ActionHead, batch: Frames, strength: float | None) -> None:
        penalty = ElasticPenalty(STRENGTH) if strength is None else consolidated(net, batch, strength)
        assert not penalty.applies

    def test_zero_at_the_anchor(self, net: ActionHead, batch: Frames) -> None:
        assert consolidated(net, batch)(net).item() == 0.0

    def test_gradient_is_strength_times_fisher_times_displacement(self, net: ActionHead, batch: Frames) -> None:
        """Pins the 0.5 factor: d/dtheta of 0.5*s*F*(theta-anchor)^2 is s*F*(theta-anchor)."""
        penalty = consolidated(net, batch)
        fisher, anchor = penalty.terms[0]
        shift(net)
        net.zero_grad()
        penalty(net).backward()
        for name, param in net.named_parameters():
            expected = STRENGTH * fisher[name] * (param.detach() - anchor[name])
            assert torch.allclose(param.grad, expected, atol=1e-6)

    def test_terms_accumulate_rather_than_collapse(self, net: ActionHead, batch: Frames) -> None:
        """Two identical consolidations penalise twice as hard as one, rather than replacing it."""
        once, twice = consolidated(net, batch), consolidated(net, batch)
        twice.observe(net, *batch, batch=len(batch[0]))
        shift(net)
        assert len(twice.terms) == 2
        assert twice(net).item() == pytest.approx(2 * once(net).item(), rel=1e-5)

    def test_fisher_is_the_squared_gradient_of_the_task_loss(self, net: ActionHead, batch: Frames) -> None:
        x, y = batch
        net.zero_grad()
        torch.nn.functional.mse_loss(net(x), y).backward()
        reference = {n: p.grad.detach().clone() ** 2 for n, p in net.named_parameters()}
        net.zero_grad()
        fisher, _ = consolidated(net, batch).terms[0]
        for name, value in reference.items():
            assert torch.allclose(fisher[name], value, atol=1e-8)


class TestOnlineElasticPenalty:
    """The variant that keeps one running Fisher instead of one per task."""

    GAMMA = 0.5

    def penalty(self, net: ActionHead, batch: Frames, n_tasks: int) -> ElasticPenalty:
        penalty = ElasticPenalty(STRENGTH, online=True, gamma=self.GAMMA)
        for _ in range(n_tasks):
            penalty.observe(net, *batch, batch=len(batch[0]))
        return penalty

    @pytest.mark.parametrize("n_tasks", [2, 5])
    def test_storage_does_not_grow_with_the_task_count(self, net: ActionHead, batch: Frames, n_tasks: int) -> None:
        """The whole reason to prefer online: two copies of the parameters, however many tasks."""
        assert len(self.penalty(net, batch, n_tasks).terms) == 1

    @pytest.mark.parametrize("n_tasks", [2, 3])
    def test_running_fisher_is_the_decayed_sum(self, net: ActionHead, batch: Frames, n_tasks: int) -> None:
        """Three tasks are needed to pin which side decays.

        Decaying the history gives 1 + g + g**2; decaying the new estimate instead gives 1 + 2g.
        Those agree at two tasks and separate at three.
        """
        once, _ = self.penalty(net, batch, 1).terms[0]
        running, _ = self.penalty(net, batch, n_tasks).terms[0]
        expected = sum(self.GAMMA**k for k in range(n_tasks))
        for name, value in once.items():
            assert torch.allclose(running[name], expected * value, atol=1e-8)

    def test_anchor_follows_the_latest_parameters(self, net: ActionHead, batch: Frames) -> None:
        penalty = self.penalty(net, batch, 1)
        shift(net)
        penalty.observe(net, *batch, batch=len(batch[0]))
        _, anchor = penalty.terms[0]
        assert all(torch.equal(anchor[n], p.detach()) for n, p in net.named_parameters())

    def test_earlier_curvature_survives_re_anchoring(self, net: ActionHead, batch: Frames) -> None:
        """The difference from simply keeping the last task, which no other test would catch."""
        accumulated = self.penalty(net, batch, 1)
        latest_only = ElasticPenalty(STRENGTH, online=True, gamma=self.GAMMA)
        shift(net)
        for penalty in (accumulated, latest_only):
            penalty.observe(net, *batch, batch=len(batch[0]))
        shift(net)
        assert accumulated(net).item() > latest_only(net).item()


class TestSequenceResult:
    """Metrics read off the [position, step] error matrix.

    Task 0 trained first at 0.10 then decays to 0.40; task 1 trained last at 0.20. Forgetting
    counts only task 0, since task 1 has had no chance to decay; plasticity counts both.
    """

    RESULT = SequenceResult(np.array([[0.10, 0.40], [0.00, 0.20]]))

    @pytest.mark.parametrize(("metric", "expected"), [
        ("just_after", [0.10, 0.20]),
        ("final_error", 0.30),
        ("abs_forgetting", 0.30),
        ("plasticity", 0.15),
        ("retention_curve", [0.10, 0.30]),
    ])
    def test_metric(self, metric: str, expected: float | list[float]) -> None:
        value = getattr(self.RESULT, metric)
        assert np.asarray(value).tolist() == pytest.approx(expected)


class TestReplayBuffer:
    """Fixed-size memory balanced across the tasks seen so far."""

    CAPACITY, FRAMES = 30, 40

    @classmethod
    def task(cls, marker: float) -> Frames:
        """Frames tagged with their task in x and with a globally unique id in y.

        The id is what makes eviction observable: without it every frame of a task looks
        alike and a buffer that resampled would be indistinguishable from one that evicted.
        Ids encode their own task as `id // FRAMES`, so pairing can be checked too.
        """
        ids = marker * cls.FRAMES + torch.arange(cls.FRAMES, dtype=torch.float32, device=DEVICE)
        return torch.full((cls.FRAMES, 2), marker, device=DEVICE), ids.unsqueeze(1)

    @classmethod
    def fill(cls, capacity: int, n_tasks: int) -> ReplayBuffer:
        buffer = ReplayBuffer(capacity)
        for t in range(n_tasks):
            buffer.add(*cls.task(float(t)))
        return buffer

    @staticmethod
    def ids(buffer: ReplayBuffer, marker: float) -> set[float]:
        return {float(v) for v in buffer.y[buffer.x[:, 0] == marker].flatten()}

    @pytest.mark.parametrize(("capacity", "n_tasks"), [(0, 3), (CAPACITY, 0)])
    def test_inapplicable_without_capacity_or_tasks(self, capacity: int, n_tasks: int) -> None:
        assert not self.fill(capacity, n_tasks).applies

    def test_capacity_zero_consumes_no_randomness(self) -> None:
        """What makes the replay-size 0 arm reproduce the unregularised numbers exactly."""
        torch.manual_seed(7)
        self.fill(0, 3)
        drawn = torch.randn(4, device=DEVICE)
        torch.manual_seed(7)
        assert torch.equal(drawn, torch.randn(4, device=DEVICE))

    @pytest.mark.parametrize("n_tasks", [1, 2, 3, 4])
    def test_quota_is_split_evenly_across_tasks(self, n_tasks: int) -> None:
        """Four tasks over 30 slots keeps 7 each and leaves 2 unused, rather than unbalancing."""
        buffer = self.fill(self.CAPACITY, n_tasks)
        counts = [len(self.ids(buffer, float(t))) for t in range(n_tasks)]
        assert counts == [self.CAPACITY // n_tasks] * n_tasks

    def test_later_tasks_evict_rather_than_resample(self) -> None:
        buffer = ReplayBuffer(self.CAPACITY)
        buffer.add(*self.task(0.0))
        kept = self.ids(buffer, 0.0)
        for marker in (1.0, 2.0):
            buffer.add(*self.task(marker))
            shrunk = self.ids(buffer, 0.0)
            assert shrunk < kept
            kept = shrunk

    def test_batch_keeps_frames_paired_with_their_actions(self) -> None:
        buffer = self.fill(self.CAPACITY, 3)
        x, y = buffer.batch(100)
        assert len(x) == self.CAPACITY
        assert torch.equal(torch.div(y.flatten(), self.FRAMES, rounding_mode="floor"), x[:, 0])

    def test_capacity_below_task_count_is_refused(self) -> None:
        buffer = ReplayBuffer(2)
        buffer.add(*self.task(0.0))
        buffer.add(*self.task(1.0))
        with pytest.raises(ValueError, match="one frame per task"):
            buffer.add(*self.task(2.0))
