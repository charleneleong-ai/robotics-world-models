"""Tests for the sequential-task probe: the EWC penalty and the forgetting metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from continual_probe import DEVICE, ActionHead, ElasticPenalty, SequenceResult

STRENGTH = 3.0


@pytest.fixture
def net() -> ActionHead:
    torch.manual_seed(0)
    return ActionHead(2, 1, width=2).to(DEVICE)


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randn(8, 2, device=DEVICE), torch.randn(8, 1, device=DEVICE)


def consolidated(net: ActionHead, batch: tuple[torch.Tensor, torch.Tensor],
                 strength: float = STRENGTH) -> ElasticPenalty:
    penalty = ElasticPenalty(strength)
    penalty.observe(net, *batch, batch=len(batch[0]))
    return penalty


class TestElasticPenalty:
    """Diagonal-Fisher EWC."""

    def test_inert_until_a_task_is_consolidated(self, net: ActionHead) -> None:
        assert not ElasticPenalty(STRENGTH).applies

    def test_strength_zero_records_nothing(self, net: ActionHead, batch) -> None:
        assert not consolidated(net, batch, strength=0.0).applies

    def test_zero_at_the_anchor(self, net: ActionHead, batch) -> None:
        assert consolidated(net, batch)(net).item() == 0.0

    def test_gradient_is_strength_times_fisher_times_displacement(self, net: ActionHead, batch) -> None:
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

    def test_terms_accumulate_rather_than_collapse(self, net: ActionHead, batch) -> None:
        """Two identical consolidations penalise twice as hard as one, rather than replacing it."""
        once, twice = consolidated(net, batch), consolidated(net, batch)
        twice.observe(net, *batch, batch=len(batch[0]))
        with torch.no_grad():
            for param in net.parameters():
                param.add_(0.1)
        assert len(twice.terms) == 2
        assert twice(net).item() == pytest.approx(2 * once(net).item(), rel=1e-5)

    def test_fisher_is_the_squared_gradient_of_the_task_loss(self, net: ActionHead, batch) -> None:
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
