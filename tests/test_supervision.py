"""Checks isolating normalization, teacher forcing, and assembled generation."""

import pytest
import torch

from spin_quantum_diffusion.quantum import single_spin as toy
from spin_quantum_diffusion.quantum.correlated import joint_multipole_coefficients
from spin_quantum_diffusion.quantum.multipoles import irreducible_spherical_tensors
from spin_quantum_diffusion.quantum.training import reverse_training_loss
from tasks.two_spin_supervision import summarize


def setup_problem():
    jx, jy, jz, identity = toy.spin_operators(0.5)
    target = identity / 2 + 0.3 * jz
    forward = toy.build_forward_trajectory(target, jx, jy, jz, 1, 0.35, 2)
    generators = toy.build_reverse_generators(jx, jy, jz, identity)
    params = (0.1 * torch.randn(2, 1, len(generators),
                              generator=torch.Generator().manual_seed(7))).requires_grad_()
    return forward, generators, params, identity / 2


def test_path_normalization_scales_loss_and_gradient_exactly():
    forward, generators, params, prior = setup_problem()
    summed, _ = reverse_training_loss(forward, params, generators, prior)
    normalized, _ = reverse_training_loss(
        forward, params, generators, prior, supervision="path-normalized"
    )
    summed_grad = torch.autograd.grad(summed, params)[0]
    normalized_grad = torch.autograd.grad(normalized, params)[0]
    torch.testing.assert_close(normalized, summed / 2, atol=1e-14, rtol=0)
    torch.testing.assert_close(normalized_grad, summed_grad / 2, atol=1e-14, rtol=0)


def test_stepwise_uses_exact_inputs_and_removes_cross_step_gradient_paths():
    forward, generators, params, prior = setup_problem()
    loss, last = reverse_training_loss(
        forward, params, generators, prior, supervision="stepwise"
    )
    first = toy.reverse_collision_channel(forward[2], params[0], generators)
    final = toy.reverse_collision_channel(forward[1], params[1], generators)
    expected = (toy.frobenius_loss(first, forward[1])
                + toy.frobenius_loss(final, forward[0])) / 2
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(last, final)
    # The final teacher-forced prediction has no dependence on the first channel.
    grad = torch.autograd.grad(last.real.sum(), params)[0]
    assert torch.count_nonzero(grad[0]) == 0
    assembled = toy.generate_density(prior, params.detach(), generators)
    assert not torch.allclose(assembled, last)


@pytest.mark.parametrize("mode", ["path-normalized", "stepwise"])
def test_new_training_modes_produce_finite_updates_and_physical_assembled_chain(mode):
    forward, generators, _, prior = setup_problem()
    diagnostics = []
    params, history = toy.train_reverse(
        forward, generators, 1, 2, 0.04, 7,
        supervision=mode, training_diagnostics=diagnostics,
    )
    assert len(history) == len(diagnostics) == 2
    assert all(row["gradient_norm"] > 0 and row["update_norm"] > 0
               for row in diagnostics)
    for state in toy.generate_trajectory(prior, params, generators):
        toy.assert_physical_density(state, mode)


def test_joint_complex_coefficients_obey_parseval_and_detect_correlations():
    jx, jy, jz, identity = toy.spin_operators(0.5)
    basis = irreducible_spherical_tensors(0.5, jx, jy)
    # Both states have identical maximally mixed marginals but different correlation.
    target = torch.eye(4, dtype=toy.CDTYPE) / 4
    prediction = target + 0.1 * torch.kron(jz, jz)
    target_coeffs = joint_multipole_coefficients(target, basis)
    pred_coeffs = joint_multipole_coefficients(prediction, basis)
    errors = {k: abs(pred_coeffs[k] - target_coeffs[k]) ** 2 for k in target_coeffs}
    assert float(sum(errors.values())) == pytest.approx(
        float(toy.frobenius_loss(prediction, target)), abs=1e-14
    )
    assert float(sum(v for (a, _, b, _), v in errors.items() if a == 0 or b == 0)) < 1e-14
    assert float(sum(errors.values())) > 0


def selection_rows():
    rows = []
    for seed in [67, 71, 73, 79, 83]:
        for mode, trace in [("path", 0.4), ("path-normalized", 0.39),
                            ("final", 0.3), ("stepwise", 0.35)]:
            rows.append({"seed": seed, "model": mode, "physicality_passed": True,
                         "validation": {"trace_distance": trace, "infidelity": 0.1,
                                        "connected_correlation_error": 0.1,
                                        "mutual_information_absolute_error": 0.1,
                                        "marginal_trace_distance_mean": 0.1}})
    return rows


def test_selection_uses_five_paired_units_and_keeps_confirmation_closed():
    summary = summarize(selection_rows())
    assert summary["validation_candidate"] == "final"
    assert summary["go_to_confirmation"]
    assert not summary["test_accessed"]
    assert summary["paired_minus_path"]["final"]["trace_distance"]["n"] == 5


def test_selection_rejects_correlation_regression_despite_better_trace_distance():
    rows = selection_rows()
    for row in rows:
        if row["model"] == "final":
            row["validation"]["connected_correlation_error"] = 0.2
    summary = summarize(rows)
    assert "final" not in summary["eligible"]
    assert summary["validation_candidate"] == "stepwise"


def test_selection_does_not_pass_an_improvement_on_only_three_of_five_datasets():
    rows = selection_rows()
    for row in rows:
        if row["model"] in {"path-normalized", "stepwise"}:
            row["validation"]["trace_distance"] = 0.41
        if row["model"] == "final" and row["seed"] in {79, 83}:
            row["validation"]["trace_distance"] = 0.41
    summary = summarize(rows)
    assert summary["validation_candidate"] == "final"
    assert summary["nonworse_seed_count"] == 3
    assert not summary["go_to_confirmation"]
