import torch

import spin_diffusion_toy as toy


TOLERANCE = 1e-10


def test_two_ancilla_generator_family_defines_a_cptp_channel():
    torch.manual_seed(31)
    Jx, Jy, Jz, identity = toy.spin_operators(1.0)
    generators, names = toy.build_reverse_generators(
        Jx,
        Jy,
        Jz,
        identity,
        ancilla_count=2,
        generator_set="full",
        return_names=True,
    )
    assert len(generators) == len(names) == 14

    theta = 0.1 * torch.randn(1, len(generators), dtype=toy.RDTYPE)
    unitary = toy.reverse_collision_unitary(theta, generators)
    kraus = toy.collision_kraus_operators(
        unitary,
        system_dimension=3,
        ancilla_dimension=4,
    )
    completeness = sum(operator.conj().T @ operator for operator in kraus)
    choi = toy.choi_matrix_from_kraus(kraus)

    assert len(kraus) == 4
    assert torch.allclose(completeness, identity, atol=TOLERANCE, rtol=TOLERANCE)
    assert float(torch.linalg.eigvalsh(choi).min()) >= -TOLERANCE


def test_shared_parameters_are_reused_for_all_reverse_steps():
    j = 1.0
    points = toy.make_direction_dataset(n_samples=80, sigma=0.22, seed=9)
    rho_data = toy.empirical_density(points, j)
    Jx, Jy, Jz, identity = toy.spin_operators(j)
    forward = toy.build_forward_trajectory(
        rho_data, Jx, Jy, Jz, D=1.0, dt=0.35, T=2
    )
    generators = toy.build_reverse_generators(
        Jx, Jy, Jz, identity, generator_set="minimal"
    )

    parameters, history = toy.train_reverse(
        forward,
        generators,
        layers=1,
        epochs=20,
        lr=0.05,
        seed=9,
        parameter_sharing="shared",
    )
    trajectory = toy.generate_trajectory(
        identity / identity.shape[0],
        parameters,
        generators,
        n_steps=2,
    )

    assert parameters.shape == (1, len(generators))
    assert len(trajectory) == 3
    assert history[-1] < history[0]
    for step, state in enumerate(trajectory):
        toy.assert_physical_density(state, f"shared reverse state {step}")
