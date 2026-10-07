import torch

from spin_quantum_diffusion.quantum import single_spin as toy


TOLERANCE = 1e-10


def random_density(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(dimension, dimension, generator=generator)
    imag = torch.randn(dimension, dimension, generator=generator)
    matrix = (real + 1j * imag).to(toy.CDTYPE)
    rho = matrix @ matrix.conj().T
    return rho / torch.trace(rho)


def assert_density_matrix(rho: torch.Tensor) -> None:
    assert torch.allclose(
        torch.trace(rho),
        torch.tensor(1.0, dtype=toy.CDTYPE),
        atol=TOLERANCE,
        rtol=0.0,
    )
    assert torch.allclose(rho, rho.conj().T, atol=TOLERANCE, rtol=0.0)
    assert float(torch.linalg.eigvalsh(rho).min()) >= -TOLERANCE


def test_forward_trajectory_is_physical_and_converges_to_fixed_point():
    j = 2.0
    Jx, Jy, Jz, identity = toy.spin_operators(j)
    rho0 = random_density(identity.shape[0], seed=12)
    trajectory = toy.build_forward_trajectory(
        rho0, Jx, Jy, Jz, D=0.7, dt=0.2, T=8
    )

    prior = identity / identity.shape[0]
    distances = []
    for rho in trajectory:
        assert_density_matrix(rho)
        distances.append(float(torch.linalg.norm(rho - prior)))

    pairs = zip(distances, distances[1:])
    assert all(later <= earlier + 1e-13 for earlier, later in pairs)
    assert distances[-1] < distances[0]


def test_forward_superoperator_is_linear_and_trace_preserving():
    Jx, Jy, Jz, _ = toy.spin_operators(1.0)
    generator = toy.liouvillian_matrix([Jx, Jy, Jz], D=0.4)
    channel = torch.matrix_exp(0.3 * generator)
    rho = random_density(3, seed=4)

    output = toy.apply_superoperator(channel, rho)
    scaled_output = toy.apply_superoperator(channel, 2.0 * rho)

    assert torch.allclose(scaled_output, 2.0 * output, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        torch.trace(output), torch.trace(rho), atol=1e-12, rtol=0.0
    )
    assert_density_matrix(output)


def test_every_reverse_collision_channel_is_cptp():
    torch.manual_seed(23)
    Jx, Jy, Jz, identity = toy.spin_operators(1.5)
    generators = toy.build_reverse_generators(Jx, Jy, Jz, identity)
    parameters = 0.2 * torch.randn(4, 2, len(generators), dtype=toy.RDTYPE)
    rho = random_density(4, seed=3)

    for theta in parameters:
        unitary = toy.reverse_collision_unitary(theta, generators)
        kraus = toy.collision_kraus_operators(unitary, system_dimension=4)
        completeness = sum(K.conj().T @ K for K in kraus)
        choi = toy.choi_matrix_from_kraus(kraus)

        assert torch.allclose(
            completeness, identity, atol=TOLERANCE, rtol=TOLERANCE
        )
        assert torch.allclose(
            choi, choi.conj().T, atol=TOLERANCE, rtol=0.0
        )
        assert float(torch.linalg.eigvalsh(choi).min()) >= -TOLERANCE

        output = toy.reverse_collision_channel(rho, theta, generators)
        scaled_output = toy.reverse_collision_channel(2.0 * rho, theta, generators)
        assert torch.allclose(
            scaled_output, 2.0 * output, atol=TOLERANCE, rtol=TOLERANCE
        )
        assert_density_matrix(output)
        rho = output
