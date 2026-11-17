"""Analytic checks of the quantities encoded by spherical heatmap colors."""

import numpy as np
import pytest
import torch

from spin_quantum_diffusion.quantum import single_spin as toy
from spin_quantum_diffusion.quantum.sphere_plots import (
    husimi_density,
    real_spherical_harmonic,
    sphere_grid,
)


def quadrature():
    x, weights = np.polynomial.legendre.leggauss(16)
    theta = np.arccos(x[::-1])
    phi = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    return theta, phi, weights[::-1, None] * (2 * np.pi / phi.size)


def test_real_harmonic_orientation_and_orthonormality():
    theta, phi, weights = quadrature()
    scale = np.sqrt(3 / (4 * np.pi))
    assert np.allclose(real_spherical_harmonic(0, 0, theta, phi), 1 / np.sqrt(4 * np.pi))
    assert np.allclose(real_spherical_harmonic(1, 1, theta, phi),
                       scale * np.sin(theta)[:, None] * np.cos(phi))
    assert np.allclose(real_spherical_harmonic(1, -1, theta, phi),
                       scale * np.sin(theta)[:, None] * np.sin(phi))
    assert np.allclose(real_spherical_harmonic(1, 0, theta, phi), scale * np.cos(theta)[:, None])
    fields = np.array([real_spherical_harmonic(ell, m, theta, phi)
                       for ell in range(4) for m in range(-ell, ell + 1)])
    gram = np.einsum("aij,bij,ij->ab", fields, fields, np.broadcast_to(weights, fields.shape[1:]))
    assert np.allclose(gram, np.eye(16), atol=2e-14, rtol=0)


def test_qubit_husimi_matches_bloch_formula_and_spherical_measure():
    theta, phi, weights = quadrature()
    plus_x = np.array([[0.5, 0.5], [0.5, 0.5]])
    q = husimi_density(plus_x, 0.5, theta, phi)
    expected = (1 + np.sin(theta)[:, None] * np.cos(phi)) / (4 * np.pi)
    assert np.allclose(q, expected, atol=2e-15)
    assert abs(np.sum(q * weights) - 1) < 2e-15
    mixed = husimi_density(np.eye(6) / 6, 2.5, theta, phi)
    assert np.allclose(mixed, 1 / (4 * np.pi), atol=2e-15)


def test_vectorized_husimi_agrees_with_coherent_state_definition():
    theta, phi = sphere_grid(5, 8)
    psi = toy.spin_coherent_state(2.5, 1.2, 2.3)
    state = torch.outer(psi, psi.conj())
    values = husimi_density(state.numpy(), 2.5, theta, phi)
    for i, t in enumerate(theta):
        for k, p in enumerate(phi):
            direction = toy.spin_coherent_state(2.5, float(t), float(p))
            expected = 6 / (4 * np.pi) * float((direction.conj() @ state @ direction).real)
            assert abs(values[i, k] - expected) < 2e-15
    assert np.allclose(values[0], values[0, 0], atol=2e-15)
    assert np.allclose(values[-1], values[-1, 0], atol=2e-15)


@pytest.mark.parametrize("state", [np.diag([1.2, -0.2]), np.eye(2),
                                  np.array([[0.5, 1], [0, 0.5]]),
                                  np.array([[np.nan, 0], [0, 1]])])
def test_husimi_rejects_invalid_states_without_repair(state):
    theta, phi = sphere_grid(5, 8)
    with pytest.raises(ValueError):
        husimi_density(state, 0.5, theta, phi)


@pytest.mark.parametrize("ell,m", [(-1, 0), (1, 2), (1.5, 0)])
def test_invalid_harmonics_are_rejected(ell, m):
    theta, phi = sphere_grid(5, 8)
    with pytest.raises(ValueError):
        real_spherical_harmonic(ell, m, theta, phi)
