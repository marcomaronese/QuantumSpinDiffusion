"""Finite spin-coherent measurements and their continuous sampling kernel."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn.functional import normalize


def coherent_kets(directions: torch.Tensor, two_j: int) -> torch.Tensor:
    """Batched Dicke-basis kets for unit Cartesian directions."""
    x, y, z = directions.to(torch.float64).unbind(dim=-1)
    theta = torch.atan2(torch.hypot(x, y), z)
    phi = torch.atan2(y, x)
    cosine = torch.cos(theta / 2)
    sine = torch.sin(theta / 2)
    return torch.stack([
        math.sqrt(math.comb(two_j, k)) * cosine ** (two_j - k) * sine ** k
        * torch.exp(1j * k * phi)
        for k in range(two_j + 1)
    ], dim=-1)


def sample_coherent_directions(
    directions: torch.Tensor, two_j: int, rng: torch.Generator
) -> torch.Tensor:
    """Reparameterized continuous coherent POVM on coherent input states."""
    shape = directions.shape[:-1]
    options = dict(dtype=directions.dtype, device=directions.device, generator=rng)
    return coherent_directions_from_uniforms(
        directions, two_j, torch.rand(shape, **options), torch.rand(shape, **options),
    )


def coherent_directions_from_uniforms(directions, two_j, radial_uniform, azimuth_uniform):
    """Exact coherent POVM quantiles, allowing shared independently saved noise."""
    if (not isinstance(two_j, int) or two_j<0 or directions.shape[-1]!=3
            or radial_uniform.shape!=directions.shape[:-1]
            or azimuth_uniform.shape!=radial_uniform.shape):
        raise ValueError("invalid spin or quantile shapes")
    if (not torch.isfinite(directions).all() or not torch.isfinite(radial_uniform).all()
            or not torch.isfinite(azimuth_uniform).all()
            or torch.any((radial_uniform<0) | (radial_uniform>1))
            or torch.any((azimuth_uniform<0) | (azimuth_uniform>1))):
        raise ValueError("invalid directions or uniform quantiles")
    overlap = radial_uniform.pow(1 / (two_j + 1))
    azimuth = 2 * math.pi * azimuth_uniform
    cosine = (2 * overlap - 1).unsqueeze(-1)
    sine = (2 * torch.sqrt(overlap * (1 - overlap))).unsqueeze(-1)
    axes = torch.eye(3, dtype=directions.dtype, device=directions.device)
    axis = axes[directions.abs().argmin(dim=-1)]
    tangent = normalize(torch.linalg.cross(directions, axis), dim=-1)
    bitangent = torch.linalg.cross(directions, tangent)
    return cosine * directions + sine * (
        azimuth.cos().unsqueeze(-1) * tangent + azimuth.sin().unsqueeze(-1) * bitangent
    )


@dataclass(frozen=True)
class CoherentPOVM:
    """Rank-one effects E_k = |frame[k]><frame[k]| on a direction grid."""

    directions: torch.Tensor
    frame: torch.Tensor

    def probabilities(self, rho: torch.Tensor) -> torch.Tensor:
        return (self.frame.conj() * (rho @ self.frame.T).T).sum(dim=-1).real

    def product(self) -> "CoherentPOVM":
        """Take the tensor product of this measurement with itself."""
        count = len(self.directions)
        pairs = torch.stack([
            self.directions.repeat_interleave(count, dim=0),
            self.directions.repeat((count, 1)),
        ], dim=1)
        frame = torch.einsum("ai,bj->abij", self.frame, self.frame)
        return CoherentPOVM(pairs, frame.reshape(count ** 2, -1))


def coherent_povm(two_j: int, polar_order: int, azimuth_count: int) -> CoherentPOVM:
    """Use Gauss–Legendre in cos(theta) and a uniform azimuth.

    The azimuth sum cancels off-diagonal Dicke terms. The surviving diagonal
    integrands are polynomials of degree two_j, integrated exactly by the
    specified Gauss–Legendre rule.
    """
    if two_j < 0 or 2 * polar_order - 1 < two_j or azimuth_count <= two_j:
        raise ValueError("quadrature order is insufficient for this spin")
    nodes, weights = np.polynomial.legendre.leggauss(polar_order)
    z = torch.from_numpy(nodes).repeat_interleave(azimuth_count)
    phi = (2 * math.pi / azimuth_count) * torch.arange(
        azimuth_count, dtype=z.dtype
    ).repeat(polar_order)
    radius = torch.sqrt(1 - z.square())
    directions = torch.stack([radius * phi.cos(), radius * phi.sin(), z], dim=-1)
    weight = torch.from_numpy(weights).repeat_interleave(azimuth_count)
    weight = weight * (two_j + 1) / (2 * azimuth_count)
    return CoherentPOVM(directions, coherent_kets(directions, two_j) * weight.sqrt()[:, None])
