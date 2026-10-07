"""Irreducible spherical-tensor diagnostics for a single spin-j system.

The tensors are orthonormal in the Hilbert--Schmidt inner product and use the
Condon--Shortley phase convention.  They are generated algebraically from
the highest-weight operator instead of relying on a particular Clebsch--
Gordan implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


TensorKey = tuple[int, int]
TensorBasis = dict[TensorKey, torch.Tensor]
MultipoleCoefficients = dict[TensorKey, complex]


def irreducible_spherical_tensors(
    j: float,
    Jx: torch.Tensor,
    Jy: torch.Tensor,
) -> TensorBasis:
    """Return all normalized ``T_(ell,m)`` for ``0 <= ell <= 2j``.

    Starting with ``T_(ell,ell) proportional to (-1)^ell J_+^ell``, lower
    tensor components are obtained from

    ``[J_-, T_(ell,m)] = sqrt((ell+m)(ell-m+1)) T_(ell,m-1)``.

    This construction fixes ``T_(0,0) = I/sqrt(2j+1)`` and gives
    ``Tr(T_(ell,m)^dagger T_(ell',m')) = delta_(ell,ell') delta_(m,m')``.
    """
    two_j = int(round(2 * j))
    if j < 0 or not math.isclose(2 * j, two_j, abs_tol=1e-12):
        raise ValueError("j must be a non-negative integer or half-integer")

    d = two_j + 1
    if Jx.shape != (d, d) or Jy.shape != (d, d):
        raise ValueError("spin operators are inconsistent with j")
    if Jx.dtype != Jy.dtype or Jx.device != Jy.device:
        raise ValueError("Jx and Jy must have matching dtype and device")

    identity = torch.eye(d, dtype=Jx.dtype, device=Jx.device)
    tensors: TensorBasis = {(0, 0): identity / math.sqrt(d)}
    J_plus = Jx + 1j * Jy
    J_minus = Jx - 1j * Jy

    for ell in range(1, two_j + 1):
        highest = torch.linalg.matrix_power(J_plus, ell)
        norm = torch.linalg.norm(highest)
        if float(norm) == 0.0:
            raise ValueError(f"failed to construct rank-{ell} tensor")

        current = ((-1) ** ell) * highest / norm
        tensors[(ell, ell)] = current

        for m in range(ell, -ell, -1):
            denominator = math.sqrt((ell + m) * (ell - m + 1))
            current = (J_minus @ current - current @ J_minus) / denominator
            tensors[(ell, m - 1)] = current

    return tensors


def multipole_coefficients(
    rho: torch.Tensor,
    tensors: Mapping[TensorKey, torch.Tensor],
) -> dict[TensorKey, torch.Tensor]:
    """Compute ``c_(ell,m) = Tr[rho T_(ell,m)^dagger]``."""
    return {
        key: torch.trace(rho @ tensor.conj().T)
        for key, tensor in tensors.items()
    }


def multipole_trajectory(
    states: Sequence[torch.Tensor],
    tensors: Mapping[TensorKey, torch.Tensor],
) -> list[dict[TensorKey, torch.Tensor]]:
    """Compute multipole coefficients for every state in a trajectory."""
    return [multipole_coefficients(rho, tensors) for rho in states]


def multipole_powers(
    coefficients: Sequence[Mapping[TensorKey, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Return ``P_ell(t) = sum_m |c_(ell,m)(t)|^2`` for each rank."""
    if not coefficients:
        raise ValueError("at least one coefficient set is required")

    ranks = sorted({ell for ell, _ in coefficients[0]})
    powers: dict[int, torch.Tensor] = {}
    for ell in ranks:
        values = []
        for at_time in coefficients:
            rank_power = sum(
                torch.abs(at_time[(ell, m)]) ** 2
                for m in range(-ell, ell + 1)
            )
            values.append(torch.real(rank_power))
        powers[ell] = torch.stack(values)
    return powers


def max_multipole_decay_error(
    coefficients: Sequence[Mapping[TensorKey, torch.Tensor]],
    times: Sequence[float],
    D: float,
) -> float:
    """Maximum absolute error against ``exp[-D ell(ell+1)t]`` decay."""
    if len(coefficients) != len(times):
        raise ValueError("coefficients and times must have the same length")
    if not coefficients:
        raise ValueError("at least one coefficient set is required")

    maximum = 0.0
    initial = coefficients[0]
    for at_time, time in zip(coefficients, times, strict=True):
        for (ell, m), c0 in initial.items():
            expected = math.exp(-D * ell * (ell + 1) * float(time)) * c0
            error = float(torch.abs(at_time[(ell, m)] - expected))
            maximum = max(maximum, error)
    return maximum
