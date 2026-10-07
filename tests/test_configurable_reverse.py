import torch

from spin_quantum_diffusion.quantum import single_spin as toy


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


def test_spectral_exponentials_match_reference_and_preserve_unitarity():
    import torch
    from spin_quantum_diffusion.quantum import single_spin as toy
    from spin_quantum_diffusion.quantum.reverse import reverse_collision_unitary
    jx, jy, jz, identity = toy.spin_operators(2.5)
    generators = toy.build_reverse_generators(jx, jy, jz, identity)
    torch.manual_seed(91)
    theta = (2 * torch.randn(3, len(generators), dtype=torch.float64)).requires_grad_()
    actual = reverse_collision_unitary(theta, generators)
    reference = torch.eye(12, dtype=torch.complex128)
    for layer in theta:
        for angle, generator in zip(layer, generators):
            reference = torch.matrix_exp(-1j * angle * generator) @ reference
    torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-10)
    assert torch.linalg.norm(actual.conj().T @ actual - torch.eye(12, dtype=actual.dtype)) < 1e-12
    actual_gradient = torch.autograd.grad(actual.real.sum(), theta, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference.real.sum(), theta)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-8, atol=1e-9)
    # A cached spectrum must not outlive an in-place change to a control.
    generators[0].mul_(.5)
    updated = reverse_collision_unitary(theta.detach(), generators)
    assert torch.linalg.norm(updated-actual.detach()) > 1e-3
