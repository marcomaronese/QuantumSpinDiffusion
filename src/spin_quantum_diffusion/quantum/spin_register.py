"""Exact product-spin operations without register-sized superoperators."""

from collections.abc import Sequence

import torch

from .coherent_readout import coherent_kets
from .reverse import reverse_collision_unitary


def encode_product_coherent(directions: torch.Tensor, two_j: int) -> torch.Tensor:
    """Average product-coherent projectors for directions of shape [sample, spin, 3]."""
    kets = coherent_kets(directions, two_j)
    product = kets[:, 0]
    for ket in kets[:, 1:].unbind(dim=1):
        product = (product.unsqueeze(-1) * ket.unsqueeze(-2)).flatten(start_dim=1)
    return product.T @ product.conj() / len(product)


def reduced_spin_density(density: torch.Tensor, dimension: int, spins: int, keep: int) -> torch.Tensor:
    """Trace all sites except one from an equally sized spin register."""
    if spins < 1 or dimension < 1 or not 0 <= keep < spins:
        raise ValueError("expected a positive register size and a valid site index")
    if density.shape != (dimension**spins, dimension**spins):
        raise ValueError("density shape is inconsistent with the register")
    rest = [site for site in range(spins) if site != keep]
    axes = [keep, *rest, spins + keep, *[spins + site for site in rest]]
    blocks = density.reshape((dimension,) * (2 * spins)).permute(axes)
    blocks = blocks.reshape(dimension, dimension**(spins - 1), dimension, dimension**(spins - 1))
    return blocks.diagonal(dim1=1, dim2=3).sum(dim=-1)


def independent_spin_channel(
    density: torch.Tensor, channel: torch.Tensor, spins: int,
) -> torch.Tensor:
    """Apply the same column-vectorized one-spin channel to each register site."""
    dimension = round(channel.shape[0] ** 0.5)
    blocks = density.reshape((dimension,) * (2 * spins))
    for spin in range(spins):
        axes = (spins + spin, spin)  # column index first for column vectorization
        columns = blocks.movedim(axes, (0, 1))
        updated = channel @ columns.reshape(dimension**2, -1)
        blocks = updated.reshape(columns.shape).movedim((0, 1), axes)
    return blocks.reshape(density.shape)


def shared_ancilla_kraus(
    angles: torch.Tensor,
    local_generators: Sequence[torch.Tensor],
    ancilla_generators: Sequence[torch.Tensor],
    spins: int,
) -> torch.Tensor:
    """Apply one layer at a time to the input isometry in spin, ancilla order.

    Each layer visits the spins in order, with one block of angles per spin,
    then applies the shared ancilla controls (with an identity on spin 0).
    Local generators act on one spin and the common qubit, so no full unitary
    on the register and ancilla needs to be formed or differentiated.
    """
    dimension = local_generators[0].shape[0] // 2
    size = dimension**spins
    identity = torch.eye(size, dtype=local_generators[0].dtype, device=angles.device)
    isometry = torch.stack((identity, torch.zeros_like(identity)), dim=1)
    isometry = isometry.reshape((dimension,) * spins + (2, size))
    local_count = len(local_generators)
    local_width = spins * local_count
    if angles.ndim != 2 or angles.shape[1] != local_width + len(ancilla_generators):
        raise ValueError("angles must contain each spin block followed by the ancilla angles")

    def apply(unitary, current, spin):
        axes = (spin, spins)
        columns = current.movedim(axes, (0, 1))
        updated = unitary @ columns.reshape(2 * dimension, -1)
        return updated.reshape(columns.shape).movedim((0, 1), axes)

    for layer in angles:
        for spin, block in enumerate(layer[:local_width].split(local_count)):
            unitary = reverse_collision_unitary(block.unsqueeze(0), local_generators)
            isometry = apply(unitary, isometry, spin)
        ancilla = reverse_collision_unitary(layer[None, local_width:], ancilla_generators)
        isometry = apply(ancilla, isometry, 0)
    return isometry.reshape(size, 2, size).transpose(0, 1).contiguous()


def product_born_probabilities(
    density: torch.Tensor, effects: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Contract local effects [outcome, d, d] against a joint density matrix."""
    spins = len(effects)
    dimension = effects[0].shape[-1]
    axes = [axis for spin in range(spins) for axis in (spin, spins + spin)]
    blocks = density.reshape((dimension,) * (2 * spins)).permute(axes)
    joint = blocks.reshape((dimension**2,) * spins)
    for site in effects:
        dual = site.transpose(-2, -1).reshape(len(site), dimension**2)
        joint = torch.tensordot(joint, dual, dims=([0], [1]))
    return joint.real
