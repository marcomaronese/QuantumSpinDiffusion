"""Transparent utilities for the separate two-spin correlated pilot."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.linalg import expm

import spin_diffusion_toy as toy
from spin_reverse import CDTYPE, normalized_generator


def angles_to_vector(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def make_correlated_direction_pairs(
    n_samples: int,
    sigma: float,
    rotation_angle: float,
    seed: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Create paired directions with bimodal marginals and tunable correlation."""
    rng = np.random.default_rng(seed)
    means = [
        np.array([0.80, 0.00, 0.60]),
        np.array([-0.40, 0.70, 0.60]),
    ]
    means = [value / np.linalg.norm(value) for value in means]
    cosine = np.cos(rotation_angle)
    sine = np.sin(rotation_angle)
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pairs = []
    for _ in range(n_samples):
        latent = int(rng.random() > 0.5)
        first = means[latent] + sigma * rng.normal(size=3)
        first = first / np.linalg.norm(first)
        second = rotation @ first + sigma * rng.normal(size=3)
        second = second / np.linalg.norm(second)
        pairs.append((toy.xyz_to_angles(first), toy.xyz_to_angles(second)))
    return pairs


def empirical_joint_density(
    pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    j: float,
) -> torch.Tensor:
    """Encode paired directions as a mixture of product coherent states."""
    dimension = int(round(2 * j + 1))
    density = torch.zeros(
        (dimension**2, dimension**2),
        dtype=CDTYPE,
    )
    for first_angles, second_angles in pairs:
        first = toy.spin_coherent_state(j, *first_angles)
        second = toy.spin_coherent_state(j, *second_angles)
        joint = torch.kron(first.contiguous(), second.contiguous())
        density = density + torch.outer(joint, joint.conj())
    return density / len(pairs)


def partial_trace_two_spin(
    density: torch.Tensor,
    dimension: int,
    keep: int,
) -> torch.Tensor:
    """Return one marginal of a two-spin density matrix."""
    blocks = density.reshape(dimension, dimension, dimension, dimension)
    if keep == 0:
        return sum(blocks[:, index, :, index] for index in range(dimension))
    if keep == 1:
        return sum(blocks[index, :, index, :] for index in range(dimension))
    raise ValueError("keep must be 0 or 1")


def factorized_density(density: torch.Tensor, dimension: int) -> torch.Tensor:
    """Product of the two empirical marginals."""
    first = partial_trace_two_spin(density, dimension, keep=0)
    second = partial_trace_two_spin(density, dimension, keep=1)
    return torch.kron(first.contiguous(), second.contiguous())


def von_neumann_entropy(density: torch.Tensor, tolerance: float = 1e-12) -> float:
    hermitian = 0.5 * (density + density.conj().T)
    eigenvalues = torch.linalg.eigvalsh(hermitian)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("entropy requested for a non-positive matrix")
    positive = eigenvalues[eigenvalues > tolerance]
    return -float(torch.sum(positive * torch.log(positive)))


def quantum_mutual_information(density: torch.Tensor, dimension: int) -> float:
    first = partial_trace_two_spin(density, dimension, keep=0)
    second = partial_trace_two_spin(density, dimension, keep=1)
    return (
        von_neumann_entropy(first)
        + von_neumann_entropy(second)
        - von_neumann_entropy(density)
    )


def connected_spin_correlation(
    density: torch.Tensor,
    operators: Sequence[torch.Tensor],
    identity: torch.Tensor,
) -> torch.Tensor:
    """Connected 3x3 correlation tensor of the two spins."""
    values = torch.zeros((3, 3), dtype=torch.float64)
    first_means = []
    second_means = []
    for operator in operators:
        first_means.append(
            torch.real(
                torch.trace(
                    density
                    @ torch.kron(operator.contiguous(), identity.contiguous())
                )
            )
        )
        second_means.append(
            torch.real(
                torch.trace(
                    density
                    @ torch.kron(identity.contiguous(), operator.contiguous())
                )
            )
        )
    for first_index, first_operator in enumerate(operators):
        for second_index, second_operator in enumerate(operators):
            joint = torch.kron(
                first_operator.contiguous(),
                second_operator.contiguous(),
            )
            joint_mean = torch.real(torch.trace(density @ joint))
            values[first_index, second_index] = (
                joint_mean
                - first_means[first_index] * second_means[second_index]
            )
    return values


def build_two_spin_forward_trajectory(
    density: torch.Tensor,
    operators: Sequence[torch.Tensor],
    identity: torch.Tensor,
    diffusion_rate: float,
    time_step: float,
    steps: int,
) -> list[torch.Tensor]:
    """Apply independent isotropic diffusion to both spins."""
    lindblad_operators = []
    for operator in operators:
        lindblad_operators.extend(
            [
                torch.kron(operator.contiguous(), identity.contiguous()),
                torch.kron(identity.contiguous(), operator.contiguous()),
            ]
        )
    generator = toy.liouvillian_matrix(lindblad_operators, D=diffusion_rate)
    channel = torch.tensor(
        expm((time_step * generator).detach().cpu().numpy()),
        dtype=CDTYPE,
    )
    trajectory = [density]
    current = density
    for _ in range(steps):
        current = toy.apply_superoperator(channel, current)
        trajectory.append(current)
    return trajectory


def build_two_spin_reverse_generators(
    operators: Sequence[torch.Tensor],
    identity: torch.Tensor,
    include_direct_interactions: bool,
) -> tuple[list[torch.Tensor], list[str]]:
    """Build a shared-ancilla two-spin ansatz with an interaction ablation."""
    Jx, Jy, Jz = operators
    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE)
    ancilla_identity = torch.eye(2, dtype=CDTYPE)
    joint_identity = torch.kron(identity.contiguous(), identity.contiguous())
    first = [
        torch.kron(operator.contiguous(), identity.contiguous())
        for operator in operators
    ]
    second = [
        torch.kron(identity.contiguous(), operator.contiguous())
        for operator in operators
    ]
    named = [
        ("Jx1", first[0]),
        ("Jy1", first[1]),
        ("Jz1", first[2]),
        ("Jx2", second[0]),
        ("Jy2", second[1]),
        ("Jz2", second[2]),
    ]
    if include_direct_interactions:
        named.extend(
            [
                ("Jz1Jz2", first[2] @ second[2]),
                ("Jx1Jx2_plus_Jy1Jy2", first[0] @ second[0] + first[1] @ second[1]),
            ]
        )
    totals = [left + right for left, right in zip(first, second, strict=True)]
    for label, total, pauli in zip(
        ["Jx_total_Xa", "Jy_total_Ya", "Jz_total_Za"],
        totals,
        [X, Y, Z],
        strict=True,
    ):
        named.append((label, torch.kron(total.contiguous(), pauli)))
    named.extend(
        [
            ("I_Xa", torch.kron(joint_identity.contiguous(), X)),
            ("I_Ya", torch.kron(joint_identity.contiguous(), Y)),
        ]
    )
    generators = [
        normalized_generator(torch.kron(value.contiguous(), ancilla_identity))
        if value.shape == joint_identity.shape
        else normalized_generator(value)
        for _, value in named
    ]
    return generators, [name for name, _ in named]


def _spherical_heat_step(
    vector: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    tangent = rng.normal(size=3)
    tangent = tangent - vector * float(tangent @ vector)
    norm = np.linalg.norm(tangent)
    if norm < 1e-14:
        return vector.copy()
    direction = tangent / norm
    distance = scale * norm / math.sqrt(2.0)
    return np.cos(distance) * vector + np.sin(distance) * direction


def sample_riemannian_heat_kernel(
    pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    n_samples: int,
    bandwidth: float,
    seed: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Bootstrap paired data and apply independent tangent heat increments."""
    rng = np.random.default_rng(seed)
    sampled = []
    indices = rng.integers(0, len(pairs), size=n_samples)
    for index in indices:
        first_angles, second_angles = pairs[int(index)]
        first = _spherical_heat_step(
            angles_to_vector(*first_angles), bandwidth, rng
        )
        second = _spherical_heat_step(
            angles_to_vector(*second_angles), bandwidth, rng
        )
        sampled.append((toy.xyz_to_angles(first), toy.xyz_to_angles(second)))
    return sampled


def clustered_product_mixture_density(
    pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    j: float,
    bond_dimension: int,
    seed: int,
    iterations: int = 30,
) -> torch.Tensor:
    """Physical separable mixture used as a low-bond tensor-network baseline."""
    if bond_dimension < 1 or bond_dimension > len(pairs):
        raise ValueError("invalid bond_dimension")
    features = np.asarray(
        [
            np.concatenate(
                [angles_to_vector(*first), angles_to_vector(*second)]
            )
            for first, second in pairs
        ]
    )
    rng = np.random.default_rng(seed)
    centers = features[
        rng.choice(len(features), size=bond_dimension, replace=False)
    ].copy()
    labels = np.zeros(len(features), dtype=int)
    for _ in range(iterations):
        distances = np.sum(
            (features[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(bond_dimension):
            members = features[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)

    dimension = int(round(2 * j + 1))
    result = torch.zeros((dimension**2, dimension**2), dtype=CDTYPE)
    for cluster in range(bond_dimension):
        selected = [pairs[index] for index in np.flatnonzero(labels == cluster)]
        if not selected:
            continue
        first_density = toy.empirical_density(
            [first for first, _ in selected], j
        )
        second_density = toy.empirical_density(
            [second for _, second in selected], j
        )
        weight = len(selected) / len(pairs)
        result = result + weight * torch.kron(
            first_density.contiguous(), second_density.contiguous()
        )
    return result
