"""Training objectives and state-distance diagnostics for spin diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from spin_multipoles import TensorKey


@dataclass(frozen=True)
class HusimiGrid:
    """Precomputed coherent-state projectors and spherical quadrature weights."""

    projectors: torch.Tensor
    measure_weights: torch.Tensor


def build_husimi_grid(
    j: float,
    coherent_state: Callable[[float, float, float], torch.Tensor],
    n_theta: int = 16,
    n_phi: int = 32,
) -> HusimiGrid:
    """Build a differentiable, finite Husimi-Q probability grid."""
    if n_theta < 2 or n_phi < 2:
        raise ValueError("Husimi loss grid requires at least two points per axis")

    thetas = torch.linspace(1e-5, math.pi - 1e-5, n_theta)
    phis = torch.linspace(0.0, 2 * math.pi, n_phi + 1)[:-1]
    projectors = []
    weights = []
    for theta in thetas:
        for phi in phis:
            psi = coherent_state(j, float(theta), float(phi))
            projectors.append(torch.outer(psi, psi.conj()))
            weights.append(torch.sin(theta))
    return HusimiGrid(
        projectors=torch.stack(projectors),
        measure_weights=torch.stack(weights),
    )


def husimi_probabilities(
    rho: torch.Tensor,
    grid: HusimiGrid,
    epsilon: float = 1e-14,
) -> torch.Tensor:
    """Evaluate and normalize the discrete Husimi-Q probability mass."""
    expectations = torch.real(
        torch.einsum("ij,gji->g", rho, grid.projectors)
    )
    unnormalized = expectations * grid.measure_weights
    if float(unnormalized.detach().min()) < -1e-10:
        raise ValueError("Husimi-Q loss grid contains a negative probability")
    stabilized = torch.clamp(unnormalized, min=0.0) + epsilon
    return stabilized / torch.sum(stabilized)


def husimi_js_divergence(
    predicted: torch.Tensor,
    target: torch.Tensor,
    grid: HusimiGrid,
) -> torch.Tensor:
    """Jensen--Shannon divergence between discrete Husimi-Q distributions."""
    p = husimi_probabilities(predicted, grid)
    q = husimi_probabilities(target, grid)
    midpoint = 0.5 * (p + q)
    return 0.5 * torch.sum(p * torch.log(p / midpoint)) + 0.5 * torch.sum(
        q * torch.log(q / midpoint)
    )


def frobenius_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared Hilbert--Schmidt/Frobenius distance."""
    difference = predicted - target
    return torch.real(torch.sum(difference.conj() * difference))


def _normalized_rank_weights(
    tensors: Mapping[TensorKey, torch.Tensor],
    weighting: str,
) -> dict[int, float]:
    ranks = sorted({ell for ell, _ in tensors})
    if weighting == "uniform":
        raw = {ell: 1.0 for ell in ranks}
    elif weighting == "rank-balanced":
        raw = {ell: 1.0 / (2 * ell + 1) for ell in ranks}
    elif weighting == "high-rank":
        raw = {ell: float((ell + 1) ** 2) for ell in ranks}
    else:
        raise ValueError(f"unknown multipole weighting: {weighting}")

    component_count = sum(2 * ell + 1 for ell in ranks)
    weighted_count = sum((2 * ell + 1) * raw[ell] for ell in ranks)
    scale = component_count / weighted_count
    return {ell: scale * value for ell, value in raw.items()}


def multipole_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    tensors: Mapping[TensorKey, torch.Tensor],
    weighting: str = "rank-balanced",
) -> torch.Tensor:
    """Weighted squared error between irreducible multipole coefficients.

    ``uniform`` is exactly the Frobenius loss by Parseval's identity.
    ``rank-balanced`` gives each rank equal aggregate raw weight, while
    ``high-rank`` emphasizes fine angular structure. Weights are normalized so
    their mean across the full operator basis is one.
    """
    weights = _normalized_rank_weights(tensors, weighting)
    difference = predicted - target
    loss = torch.zeros((), dtype=torch.float64, device=predicted.device)
    for (ell, _), tensor in tensors.items():
        coefficient = torch.trace(difference @ tensor.conj().T)
        loss = loss + weights[ell] * torch.abs(coefficient) ** 2
    return torch.real(loss)


def _positive_matrix_square_root(matrix: torch.Tensor) -> torch.Tensor:
    """Principal square root of a PSD matrix, tolerating roundoff at zero."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    if float(eigenvalues.detach().min()) < -1e-10:
        raise ValueError("matrix square root received a non-positive matrix")
    roots = torch.sqrt(torch.clamp(eigenvalues, min=0.0))
    return (eigenvectors * roots.unsqueeze(0)) @ eigenvectors.conj().T


def quantum_fidelity(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared Uhlmann fidelity in the conventional ``[0, 1]`` range."""
    target_root = _positive_matrix_square_root(target)
    sandwiched = target_root @ predicted @ target_root
    eigenvalues = torch.linalg.eigvalsh(sandwiched)
    if float(eigenvalues.detach().min()) < -1e-10:
        raise ValueError("fidelity sandwich is not positive semidefinite")
    root_trace = torch.sum(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
    return torch.clamp(torch.real(root_trace**2), min=0.0, max=1.0)


def fidelity_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One minus squared Uhlmann fidelity."""
    return 1.0 - quantum_fidelity(predicted, target)


def trace_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Trace distance ``0.5 ||predicted-target||_1`` for Hermitian states."""
    eigenvalues = torch.linalg.eigvalsh(predicted - target)
    return 0.5 * torch.sum(torch.abs(eigenvalues))


def objective_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    *,
    tensors: Mapping[TensorKey, torch.Tensor] | None = None,
    multipole_weighting: str = "rank-balanced",
    husimi_grid: HusimiGrid | None = None,
) -> torch.Tensor:
    """Dispatch a named reverse-training objective."""
    if objective == "frobenius":
        return frobenius_loss(predicted, target)
    if objective == "multipole":
        if tensors is None:
            raise ValueError("multipole objective requires a tensor basis")
        return multipole_loss(predicted, target, tensors, multipole_weighting)
    if objective == "fidelity":
        return fidelity_loss(predicted, target)
    if objective == "trace-distance":
        return trace_distance(predicted, target)
    if objective == "husimi-js":
        if husimi_grid is None:
            raise ValueError("Husimi-JS objective requires a Husimi grid")
        return husimi_js_divergence(predicted, target, husimi_grid)
    raise ValueError(f"unknown training objective: {objective}")
