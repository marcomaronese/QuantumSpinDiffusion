"""Configurable Stinespring reverse channels and quantum instruments.

The ancilla register is initialized in ``|0...0>``.  Tracing it out gives a
deterministic CPTP channel; measuring it in the computational basis gives the
associated stochastic quantum instrument.  Conditional-state normalization
is performed only after a measurement outcome has been selected.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


CDTYPE = torch.complex128
RDTYPE = torch.float64

GENERATOR_SETS = ("full", "no-twist", "minimal")


def normalized_generator(generator: torch.Tensor) -> torch.Tensor:
    """Return a Hilbert--Schmidt normalized Hermitian generator."""
    norm = torch.linalg.norm(generator)
    if float(norm) == 0.0:
        raise ValueError("cannot normalize the zero generator")
    return generator / norm


def _kron_all(operators: Sequence[torch.Tensor]) -> torch.Tensor:
    result = operators[0]
    for operator in operators[1:]:
        result = torch.kron(result.contiguous(), operator.contiguous())
    return result


def _ancilla_local_operator(
    operator: torch.Tensor,
    index: int,
    ancilla_count: int,
) -> torch.Tensor:
    identity = torch.eye(2, dtype=CDTYPE, device=operator.device)
    factors = [identity for _ in range(ancilla_count)]
    factors[index] = operator
    return _kron_all(factors)


def build_reverse_generators(
    Jx: torch.Tensor,
    Jy: torch.Tensor,
    Jz: torch.Tensor,
    identity: torch.Tensor,
    ancilla_count: int = 1,
    generator_set: str = "full",
    *,
    return_names: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[str]]:
    """Build a configurable system--ancilla generator family.

    ``full`` reproduces the original one-ancilla ansatz by default.  With more
    ancillas, the same three spin--Pauli couplings and two ancilla controls are
    added for each ancilla qubit.  ``no-twist`` omits :math:`J_z^2`, while
    ``minimal`` retains only the three spin--ancilla couplings per ancilla.
    """
    if ancilla_count < 1:
        raise ValueError("ancilla_count must be at least one")
    if generator_set not in GENERATOR_SETS:
        raise ValueError(
            f"unknown generator_set {generator_set!r}; choose from {GENERATOR_SETS}"
        )

    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=Jx.device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE, device=Jx.device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE, device=Jx.device)
    ancilla_identity = torch.eye(
        2**ancilla_count, dtype=CDTYPE, device=Jx.device
    )

    named_generators: list[tuple[str, torch.Tensor]] = []
    if generator_set != "minimal":
        named_generators.extend(
            [
                ("Jx", torch.kron(Jx.contiguous(), ancilla_identity)),
                ("Jy", torch.kron(Jy.contiguous(), ancilla_identity)),
                ("Jz", torch.kron(Jz.contiguous(), ancilla_identity)),
            ]
        )
        if generator_set == "full":
            named_generators.append(
                (
                    "Jz2",
                    torch.kron((Jz @ Jz).contiguous(), ancilla_identity),
                )
            )

    for ancilla_index in range(ancilla_count):
        ancilla_x = _ancilla_local_operator(X, ancilla_index, ancilla_count)
        ancilla_y = _ancilla_local_operator(Y, ancilla_index, ancilla_count)
        ancilla_z = _ancilla_local_operator(Z, ancilla_index, ancilla_count)
        suffix = f"a{ancilla_index}"
        named_generators.extend(
            [
                (
                    f"Jx_X{suffix}",
                    torch.kron(Jx.contiguous(), ancilla_x.contiguous()),
                ),
                (
                    f"Jy_Y{suffix}",
                    torch.kron(Jy.contiguous(), ancilla_y.contiguous()),
                ),
                (
                    f"Jz_Z{suffix}",
                    torch.kron(Jz.contiguous(), ancilla_z.contiguous()),
                ),
            ]
        )
        if generator_set != "minimal":
            named_generators.extend(
                [
                    (
                        f"I_X{suffix}",
                        torch.kron(identity.contiguous(), ancilla_x.contiguous()),
                    ),
                    (
                        f"I_Y{suffix}",
                        torch.kron(identity.contiguous(), ancilla_y.contiguous()),
                    ),
                ]
            )

    names = [name for name, _ in named_generators]
    generators = [normalized_generator(value) for _, value in named_generators]
    if return_names:
        return generators, names
    return generators


def partial_trace_ancilla(
    joint: torch.Tensor,
    system_dimension: int,
    ancilla_dimension: int | None = None,
) -> torch.Tensor:
    """Trace an arbitrary finite ancilla register from a joint operator."""
    if ancilla_dimension is None:
        if joint.shape[0] % system_dimension:
            raise ValueError("joint dimension is inconsistent with the system")
        ancilla_dimension = joint.shape[0] // system_dimension
    expected = system_dimension * ancilla_dimension
    if joint.shape != (expected, expected):
        raise ValueError("joint dimension is inconsistent with system and ancilla")
    blocks = joint.reshape(
        system_dimension,
        ancilla_dimension,
        system_dimension,
        ancilla_dimension,
    )
    return sum(
        blocks[:, outcome, :, outcome]
        for outcome in range(ancilla_dimension)
    )


def reverse_collision_unitary(
    theta: torch.Tensor,
    generators: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Build the system--ancilla unitary for one reverse step."""
    if theta.ndim != 2 or theta.shape[1] != len(generators):
        raise ValueError("theta must have shape [layers, n_generators]")
    dimension = generators[0].shape[0]
    unitary = torch.eye(dimension, dtype=CDTYPE, device=theta.device)
    for layer in range(theta.shape[0]):
        for index, generator in enumerate(generators):
            update = torch.matrix_exp(-1j * theta[layer, index] * generator)
            unitary = update @ unitary
    return unitary


def collision_kraus_operators(
    unitary: torch.Tensor,
    system_dimension: int,
    ancilla_dimension: int | None = None,
) -> list[torch.Tensor]:
    """Extract all Kraus operators ``<k|U|0...0>``."""
    if ancilla_dimension is None:
        if unitary.shape[0] % system_dimension:
            raise ValueError("unitary dimension is inconsistent with the system")
        ancilla_dimension = unitary.shape[0] // system_dimension
    expected = system_dimension * ancilla_dimension
    if unitary.shape != (expected, expected):
        raise ValueError("unitary dimension is inconsistent with system and ancilla")
    blocks = unitary.reshape(
        system_dimension,
        ancilla_dimension,
        system_dimension,
        ancilla_dimension,
    )
    return [blocks[:, outcome, :, 0] for outcome in range(ancilla_dimension)]


def choi_matrix_from_kraus(
    kraus_operators: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Construct the unnormalized Choi matrix from Kraus operators."""
    vectors = [operator.T.contiguous().reshape(-1) for operator in kraus_operators]
    return sum(torch.outer(vector, vector.conj()) for vector in vectors)


def reverse_collision_channel(
    rho: torch.Tensor,
    theta: torch.Tensor,
    generators: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Apply the deterministic CPTP collision channel without repairs."""
    unitary = reverse_collision_unitary(theta, generators)
    kraus = collision_kraus_operators(unitary, rho.shape[0])
    return sum(operator @ rho @ operator.conj().T for operator in kraus)


def parameter_schedule(
    parameters: torch.Tensor,
    n_steps: int | None = None,
) -> list[torch.Tensor]:
    """Expand independent or shared reverse parameters into a time schedule."""
    if parameters.ndim == 3:
        if n_steps is not None and n_steps != parameters.shape[0]:
            raise ValueError("n_steps disagrees with independent parameter count")
        return [parameters[index] for index in range(parameters.shape[0])]
    if parameters.ndim == 2:
        if n_steps is None or n_steps < 1:
            raise ValueError("n_steps is required for shared parameters")
        return [parameters for _ in range(n_steps)]
    raise ValueError("parameters must be [T,layers,generators] or [layers,generators]")


def instrument_outcomes(
    rho: torch.Tensor,
    theta: torch.Tensor,
    generators: Sequence[torch.Tensor],
    tolerance: float = 1e-10,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor | None]]:
    """Return outcome probabilities, unnormalized branches, and conditionals."""
    unitary = reverse_collision_unitary(theta, generators)
    kraus = collision_kraus_operators(unitary, rho.shape[0])
    branches = [operator @ rho @ operator.conj().T for operator in kraus]
    probabilities = torch.stack([torch.real(torch.trace(branch)) for branch in branches])
    if float(probabilities.min()) < -tolerance:
        raise ValueError("quantum instrument produced a negative probability")
    if abs(float(probabilities.sum()) - float(torch.real(torch.trace(rho)))) > tolerance:
        raise ValueError("quantum instrument probabilities do not preserve trace")
    conditional_states = [
        branch / probability if float(probability) > tolerance else None
        for branch, probability in zip(branches, probabilities, strict=True)
    ]
    return probabilities, branches, conditional_states


def sample_instrument_step(
    rho: torch.Tensor,
    theta: torch.Tensor,
    generators: Sequence[torch.Tensor],
    random_generator: torch.Generator,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Measure the ancilla and return one normalized conditional branch."""
    probabilities, _, conditional_states = instrument_outcomes(
        rho, theta, generators
    )
    sampling_probabilities = probabilities / probabilities.sum()
    outcome = int(
        torch.multinomial(
            sampling_probabilities,
            num_samples=1,
            generator=random_generator,
        ).item()
    )
    state = conditional_states[outcome]
    if state is None:
        raise RuntimeError("sampled a zero-probability instrument outcome")
    return state, outcome, probabilities


@torch.no_grad()
def stochastic_reverse_trajectory(
    prior: torch.Tensor,
    parameters: torch.Tensor,
    generators: Sequence[torch.Tensor],
    *,
    seed: int,
    n_steps: int | None = None,
) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor]]:
    """Sample one reproducible measurement-conditioned reverse trajectory."""
    random_generator = torch.Generator(device="cpu").manual_seed(seed)
    states = [prior]
    outcomes: list[int] = []
    probabilities: list[torch.Tensor] = []
    rho = prior
    for theta in parameter_schedule(parameters, n_steps=n_steps):
        rho, outcome, step_probabilities = sample_instrument_step(
            rho,
            theta,
            generators,
            random_generator,
        )
        states.append(rho)
        outcomes.append(outcome)
        probabilities.append(step_probabilities)
    return states, outcomes, probabilities
