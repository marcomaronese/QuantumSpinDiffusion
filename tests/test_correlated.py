import torch

import spin_diffusion_toy as toy
from spin_correlated import (
    build_two_spin_forward_trajectory,
    build_two_spin_reverse_generators,
    empirical_joint_density,
    factorized_density,
    make_correlated_direction_pairs,
    partial_trace_two_spin,
    quantum_mutual_information,
)


def test_two_spin_encoding_marginals_and_factorized_baseline_are_physical():
    pairs = make_correlated_direction_pairs(
        n_samples=100,
        sigma=0.18,
        rotation_angle=0.45,
        seed=17,
    )
    density = empirical_joint_density(pairs, j=1.0)
    factorized = factorized_density(density, dimension=3)

    toy.assert_physical_density(density, "two-spin empirical density")
    toy.assert_physical_density(factorized, "factorized baseline")
    assert quantum_mutual_information(density, dimension=3) > 0.0
    assert abs(quantum_mutual_information(factorized, dimension=3)) < 1e-10
    first = partial_trace_two_spin(density, dimension=3, keep=0)
    second = partial_trace_two_spin(density, dimension=3, keep=1)
    assert torch.allclose(
        factorized,
        torch.kron(first.contiguous(), second.contiguous()),
        atol=1e-12,
        rtol=0.0,
    )


def test_two_spin_forward_and_interacting_reverse_channel_are_physical():
    pairs = make_correlated_direction_pairs(60, 0.18, 0.45, seed=19)
    density = empirical_joint_density(pairs, j=1.0)
    Jx, Jy, Jz, identity = toy.spin_operators(1.0)
    forward = build_two_spin_forward_trajectory(
        density,
        [Jx, Jy, Jz],
        identity,
        diffusion_rate=1.0,
        time_step=0.3,
        steps=2,
    )
    for step, state in enumerate(forward):
        toy.assert_physical_density(state, f"two-spin forward state {step}")

    local, local_names = build_two_spin_reverse_generators(
        [Jx, Jy, Jz], identity, include_direct_interactions=False
    )
    interacting, interacting_names = build_two_spin_reverse_generators(
        [Jx, Jy, Jz], identity, include_direct_interactions=True
    )
    assert len(interacting) == len(local) + 2
    assert len(interacting_names) == len(local_names) + 2
    theta = 0.05 * torch.randn(1, len(interacting), dtype=toy.RDTYPE)
    output = toy.reverse_collision_channel(density, theta, interacting)
    toy.assert_physical_density(output, "two-spin reverse output")
