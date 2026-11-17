"""Check tensor-axis ordering against dense small-register calculations."""

import itertools

import numpy as np
from scipy.linalg import expm
import torch

from spin_quantum_diffusion.quantum.coherent_readout import coherent_povm
from spin_quantum_diffusion.quantum.correlated import (
    build_two_spin_forward_trajectory,
    empirical_joint_density,
)
from spin_quantum_diffusion.quantum.reverse import (
    build_reverse_generators,
    collision_kraus_operators,
    reverse_collision_unitary,
)
from spin_quantum_diffusion.quantum.single_spin import liouvillian_matrix, spin_operators
from spin_quantum_diffusion.quantum.spin_register import (
    encode_product_coherent,
    independent_spin_channel,
    product_born_probabilities,
    reduced_spin_density,
    shared_ancilla_kraus,
)
from tasks.spin_grid import mode_centers, region_histogram, sample_directions


def test_product_encoding_and_local_heat_match_two_spin_implementation():
    directions, _ = sample_directions(12, 19, mode_centers())
    directions = directions[:, :2]
    pairs = [[(np.arccos(z), np.arctan2(y, x)) for x, y, z in observation]
             for observation in directions]
    *operators, identity = spin_operators(1.0)
    expected = empirical_joint_density(pairs, 1.0)
    encoded = encode_product_coherent(torch.from_numpy(directions), 2)
    torch.testing.assert_close(encoded, expected, rtol=0, atol=1e-12)
    local = torch.from_numpy(expm(0.3 * liouvillian_matrix(operators, 1.0).numpy()))
    reference = build_two_spin_forward_trajectory(expected, operators, identity, 1.0, 0.3, 1)[1]
    torch.testing.assert_close(independent_spin_channel(encoded, local, 2), reference,
                               rtol=0, atol=1e-12)


def test_shared_ancilla_kraus_matches_dense_three_spin_unitary():
    spins, dimension = 3, 2
    size = dimension**spins
    generators = build_reverse_generators(*spin_operators(0.5), generator_set="no-twist")
    local, ancilla = generators[:-2], generators[-2:]
    rng = torch.Generator().manual_seed(21)
    angles = torch.randn((2, spins * len(local) + len(ancilla)), generator=rng,
                         dtype=torch.float64).requires_grad_()
    kraus = shared_ancilla_kraus(angles, local, ancilla, spins)
    full = torch.eye(2 * size, dtype=torch.complex128)
    identity = torch.eye(dimension**(spins - 1), dtype=full.dtype)
    basis = torch.tensor(list(itertools.product(range(dimension), repeat=spins + 1)))
    for layer in angles.detach():
        blocks = list(layer[:-2].split(len(local))) + [layer[-2:]]
        for spin, (block, controls) in enumerate(zip(blocks, [local] * spins + [ancilla], strict=True)):
            site = spin % spins
            spectators = [s for s in range(spins) if s != site]
            order = [*spectators, site, spins]
            permutation = basis[:, order] @ torch.tensor([8, 4, 2, 1])
            unitary = reverse_collision_unitary(block[None], controls)
            embedded = torch.kron(identity, unitary.contiguous())[permutation][:, permutation]
            full = embedded @ full
    reference = torch.stack(collision_kraus_operators(full, size))
    torch.testing.assert_close(kraus, reference, rtol=0, atol=1e-12)
    torch.testing.assert_close((kraus.mH @ kraus).sum(0), torch.eye(size, dtype=full.dtype),
                               rtol=0, atol=1e-12)
    kraus[0, 0, 0].real.backward()
    assert torch.isfinite(angles.grad).all()
    assert angles.grad.norm() > 0


def test_readout_contraction_and_raw_marginals_have_the_expected_site_order():
    centers = mode_centers()
    directions, _ = sample_directions(45, 23, centers)
    raw = region_histogram(directions, centers)
    np.testing.assert_allclose(raw.sum(axis=2), region_histogram(directions[:, :2], centers[:2]))
    density = encode_product_coherent(torch.from_numpy(directions), 1)
    frame = coherent_povm(1, 2, 3).frame
    effects = torch.stack([group.T @ group.conj() for group in frame.split(3)])
    actual = product_born_probabilities(density, [effects] * 3)
    expected = torch.stack([
        torch.trace(density @ torch.kron(torch.kron(a, b), c)).real
        for a, b, c in itertools.product(effects, repeat=3)
    ]).reshape(2, 2, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(actual.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_reduced_density_matches_direct_encoding_of_each_saved_site():
    directions, _ = sample_directions(27, 31, mode_centers())
    directions = torch.from_numpy(directions)
    for spins in (1, 2, 3):
        density = encode_product_coherent(directions[:, :spins], 2)
        for keep in range(spins):
            expected = encode_product_coherent(directions[:, keep:keep + 1], 2)
            torch.testing.assert_close(reduced_spin_density(density, 3, spins, keep), expected,
                                       rtol=0, atol=1e-12)
