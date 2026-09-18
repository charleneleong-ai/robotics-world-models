"""Tests for the sequential-task probe: the EWC penalty, the replay memory, and the metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from continual_probe import DEVICE, ActionHead, ElasticPenalty, ReplayBuffer, SequenceResult

STRENGTH = 3.0
Frames = tuple[torch.Tensor, torch.Tensor]


@pytest.fixture
def net() -> ActionHead:
    torch.manual_seed(0)
    return ActionHead(2, 1, width=2).to(DEVICE)


@pytest.fixture
def batch() -> Frames:
    torch.manual_seed(1)
    return torch.randn(8, 2, device=DEVICE), torch.randn(8, 1, device=DEVICE)


def consolidated(net: ActionHead, batch: Frames,
                 strength: float = STRENGTH) -> ElasticPenalty:
    penalty = ElasticPenalty(strength)
    penalty.observe(net, *batch, batch=len(batch[0]))
    return penalty


class TestElasticPenalty:
    """Diagonal-Fisher EWC."""

    def test_inert_until_a_task_is_consolidated(self) -> None:
        assert not ElasticPenalty(STRENGTH).applies

    def test_strength_zero_records_nothing(self, net: ActionHead, batch: Frames) -> None:
        assert not consolidated(net, batch, strength=0.0).applies

    def test_zero_at_the_anchor(self, net: ActionHead, batch: Frames) -> None:
        assert consolidated(net, batch)(net).item() == 0.0

    def test_gradient_is_strength_times_fisher_times_displacement(self, net: ActionHead, batch: Frames) -> None:
        """Pins the 0.5 factor: d/dtheta of 0.5*s*F*(theta-anchor)^2 is s*F*(theta-anchor)."""
        penalty = consolidated(net, batch)
        fisher, anchor = penalty.terms[0]
        with torch.no_grad():
            for param in net.parameters():
                param.add_(0.1)
        net.zero_grad()
        penalty(net).backward()
        for name, param in net.named_parameters():
            expected = STRENGTH * fisher[name] * (param.detach() - anchor[name])
            assert torch.allclose(param.grad, expected, atol=1e-6)

    def test_terms_accumulate_rather_than_collapse(self, net: ActionHead, batch: Frames) -> None:
        """Two identical consolidations penalise twice as hard as one, rather than replacing it."""
        once, twice = consolidated(net, batch), consolidated(net, batch)
        twice.observe(net, *batch, batch=len(batch[0]))
        with torch.no_grad():
            for param in net.parameters():
                param.add_(0.1)
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


class TestSequenceResult:
    """Metrics read off the [position, step] error matrix."""

    @pytest.fixture
    def result(self) -> SequenceResult:
        # Task 0 trained first at 0.10 then decays to 0.40; task 1 trained last at 0.20.
        return SequenceResult(np.array([[0.10, 0.40], [0.00, 0.20]]))

    def test_just_after_reads_the_diagonal(self, result: SequenceResult) -> None:
        assert result.just_after.tolist() == [0.10, 0.20]

    def test_final_error_averages_the_last_column(self, result: SequenceResult) -> None:
        assert result.final_error == pytest.approx(0.30)

    def test_forgetting_excludes_the_task_trained_last(self, result: SequenceResult) -> None:
        """Only task 0 can have decayed; task 1 has had no chance to."""
        assert result.abs_forgetting == pytest.approx(0.30)

    def test_plasticity_includes_every_task(self, result: SequenceResult) -> None:
        assert result.plasticity == pytest.approx(0.15)

    def test_retention_curve_averages_only_tasks_seen_so_far(self, result: SequenceResult) -> None:
        assert result.retention_curve == pytest.approx([0.10, 0.30])


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

    def test_capacity_zero_never_applies(self) -> None:
        assert not self.fill(0, 3).applies

    def test_empty_until_a_task_is_stored(self) -> None:
        assert not ReplayBuffer(self.CAPACITY).applies

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
