import pytest
import torch

import spin_diffusion_toy as toy
from spin_losses import (
    build_husimi_grid,
    fidelity_loss,
    frobenius_loss,
    husimi_js_divergence,
    multipole_loss,
    quantum_fidelity,
    trace_distance,
)
from spin_multipoles import irreducible_spherical_tensors


def test_fidelity_and_trace_distance_on_identical_and_orthogonal_states():
    zero = torch.tensor([1.0, 0.0], dtype=toy.CDTYPE)
    one = torch.tensor([0.0, 1.0], dtype=toy.CDTYPE)
    rho_zero = torch.outer(zero, zero.conj())
    rho_one = torch.outer(one, one.conj())

    assert float(quantum_fidelity(rho_zero, rho_zero)) == pytest.approx(1.0)
    assert float(fidelity_loss(rho_zero, rho_zero)) == pytest.approx(0.0)
    assert float(trace_distance(rho_zero, rho_zero)) == pytest.approx(0.0)
    assert float(quantum_fidelity(rho_zero, rho_one)) == pytest.approx(0.0)
    assert float(trace_distance(rho_zero, rho_one)) == pytest.approx(1.0)


def test_uniform_multipole_loss_equals_frobenius_loss():
    j = 2.0
    Jx, Jy, _, identity = toy.spin_operators(j)
    tensors = irreducible_spherical_tensors(j, Jx, Jy)
    predicted = identity / identity.shape[0]
    psi = toy.spin_coherent_state(j, theta=0.8, phi=1.2)
    target = torch.outer(psi, psi.conj())

    assert torch.allclose(
        multipole_loss(predicted, target, tensors, weighting="uniform"),
        frobenius_loss(predicted, target),
        atol=3e-12,
        rtol=3e-12,
    )


def test_husimi_js_is_symmetric_zero_on_identity_and_differentiable():
    j = 1.0
    _, _, _, identity = toy.spin_operators(j)
    grid = build_husimi_grid(
        j,
        toy.spin_coherent_state,
        n_theta=8,
        n_phi=16,
    )
    psi_a = toy.spin_coherent_state(j, theta=0.6, phi=0.4)
    psi_b = toy.spin_coherent_state(j, theta=1.7, phi=2.2)
    rho_a = torch.outer(psi_a, psi_a.conj()).requires_grad_(True)
    rho_b = torch.outer(psi_b, psi_b.conj())

    assert float(husimi_js_divergence(identity / 3, identity / 3, grid)) == pytest.approx(
        0.0, abs=1e-15
    )
    ab = husimi_js_divergence(rho_a, rho_b, grid)
    ba = husimi_js_divergence(rho_b, rho_a, grid)
    assert float(ab.detach()) > 0.0
    assert torch.allclose(ab, ba, atol=1e-14, rtol=1e-14)

    ab.backward()
    assert rho_a.grad is not None
    assert torch.all(torch.isfinite(rho_a.grad))
