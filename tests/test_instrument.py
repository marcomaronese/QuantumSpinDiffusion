import torch

from spin_quantum_diffusion.quantum import single_spin as toy
from spin_quantum_diffusion.quantum.reverse import (
    instrument_outcomes,
    stochastic_reverse_trajectory,
)


TOLERANCE = 1e-10


def _random_density(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(
        dimension, dimension, generator=generator, dtype=toy.CDTYPE
    )
    density = matrix @ matrix.conj().T
    return density / torch.trace(density)


def test_instrument_average_equals_deterministic_collision_channel():
    torch.manual_seed(41)
    Jx, Jy, Jz, identity = toy.spin_operators(1.0)
    generators = toy.build_reverse_generators(
        Jx, Jy, Jz, identity, ancilla_count=2, generator_set="minimal"
    )
    theta = 0.15 * torch.randn(2, len(generators), dtype=toy.RDTYPE)
    rho = _random_density(3, seed=6)

    probabilities, branches, conditionals = instrument_outcomes(
        rho, theta, generators
    )
    deterministic = toy.reverse_collision_channel(rho, theta, generators)
    conditional_average = sum(
        probability * state
        for probability, state in zip(probabilities, conditionals, strict=True)
        if state is not None
    )

    assert len(probabilities) == 4
    assert float(probabilities.min()) >= -TOLERANCE
    assert torch.allclose(
        probabilities.sum(),
        torch.tensor(1.0, dtype=toy.RDTYPE),
        atol=TOLERANCE,
        rtol=0.0,
    )
    assert torch.allclose(sum(branches), deterministic, atol=TOLERANCE, rtol=0.0)
    assert torch.allclose(
        conditional_average, deterministic, atol=TOLERANCE, rtol=0.0
    )
    for outcome, state in enumerate(conditionals):
        if state is not None:
            toy.assert_physical_density(state, f"conditional outcome {outcome}")


def test_stochastic_reverse_trajectory_is_seed_reproducible():
    torch.manual_seed(43)
    Jx, Jy, Jz, identity = toy.spin_operators(0.5)
    generators = toy.build_reverse_generators(Jx, Jy, Jz, identity)
    parameters = 0.25 * torch.randn(
        3, 1, len(generators), dtype=toy.RDTYPE
    )
    prior = identity / 2

    first_states, first_outcomes, _ = stochastic_reverse_trajectory(
        prior, parameters, generators, seed=101
    )
    second_states, second_outcomes, _ = stochastic_reverse_trajectory(
        prior, parameters, generators, seed=101
    )

    assert first_outcomes == second_outcomes
    assert all(
        torch.allclose(first, second, atol=0.0, rtol=0.0)
        for first, second in zip(first_states, second_states, strict=True)
    )
    for step, state in enumerate(first_states):
        toy.assert_physical_density(state, f"stochastic state {step}")
