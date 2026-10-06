import math

import pytest
import torch

import spin_diffusion_toy as toy
from spin_multipoles import (
    irreducible_spherical_tensors,
    max_multipole_decay_error,
    multipole_coefficients,
    multipole_powers,
    multipole_trajectory,
)


@pytest.mark.parametrize("j", [0.5, 1.0, 1.5, 2.0])
def test_irreducible_tensors_form_an_orthonormal_operator_basis(j):
    Jx, Jy, Jz, _ = toy.spin_operators(j)
    tensors = irreducible_spherical_tensors(j, Jx, Jy)
    dimension = int(round(2 * j + 1))

    assert len(tensors) == dimension**2
    for key_a, tensor_a in tensors.items():
        ell, m = key_a
        casimir = sum(
            J @ (J @ tensor_a - tensor_a @ J)
            - (J @ tensor_a - tensor_a @ J) @ J
            for J in (Jx, Jy, Jz)
        )
        assert torch.allclose(
            casimir,
            ell * (ell + 1) * tensor_a,
            atol=3e-12,
            rtol=3e-12,
        )
        assert torch.allclose(
            tensor_a.conj().T,
            ((-1) ** m) * tensors[(ell, -m)],
            atol=3e-12,
            rtol=3e-12,
        )

        for key_b, tensor_b in tensors.items():
            expected = 1.0 if key_a == key_b else 0.0
            overlap = torch.trace(tensor_a.conj().T @ tensor_b)
            assert abs(complex(overlap) - expected) < 3e-12


def test_multipoles_reconstruct_density_matrix():
    j = 2.0
    Jx, Jy, _, identity = toy.spin_operators(j)
    tensors = irreducible_spherical_tensors(j, Jx, Jy)
    psi = toy.spin_coherent_state(j, theta=1.1, phi=0.7)
    rho = torch.outer(psi, psi.conj())
    coefficients = multipole_coefficients(rho, tensors)
    reconstructed = sum(
        coefficients[key] * tensor for key, tensor in tensors.items()
    )

    assert reconstructed.shape == identity.shape
    assert torch.allclose(reconstructed, rho, atol=3e-12, rtol=3e-12)


def test_forward_multipoles_follow_theoretical_decay_for_every_rank_and_m():
    j = 2.0
    D = 0.83
    dt = 0.17
    steps = 7
    Jx, Jy, Jz, _ = toy.spin_operators(j)
    points = toy.make_direction_dataset(n_samples=200, sigma=0.22, seed=8)
    rho0 = toy.empirical_density(points, j)
    states = toy.build_forward_trajectory(rho0, Jx, Jy, Jz, D, dt, steps)
    tensors = irreducible_spherical_tensors(j, Jx, Jy)
    coefficients = multipole_trajectory(states, tensors)
    powers = multipole_powers(coefficients)
    times = [step * dt for step in range(steps + 1)]

    assert max_multipole_decay_error(coefficients, times, D) < 2e-12
    for ell, observed in powers.items():
        expected = torch.tensor(
            [
                float(observed[0])
                * math.exp(-2 * D * ell * (ell + 1) * time)
                for time in times
            ]
        )
        assert torch.allclose(observed, expected, atol=2e-12, rtol=2e-10)
