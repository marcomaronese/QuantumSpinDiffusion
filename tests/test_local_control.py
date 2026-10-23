"""Structural product-channel and frozen-decision checks required by E5."""

import numpy as np
import pytest
import torch

from spin_quantum_diffusion.quantum import single_spin as toy
from spin_quantum_diffusion.quantum.correlated import (
    build_independent_two_spin_reverse_generators, partial_trace_two_spin,
    build_two_spin_reverse_generators,
)
from spin_quantum_diffusion.quantum.reverse import build_reverse_generators
from tasks.two_spin_local_control import MODELS, initial_angles, summary


@pytest.mark.parametrize("j", [.5, 1.])
def test_independent_channel_equals_product_of_local_maps(j):
    jx, jy, jz, identity = toy.spin_operators(j)
    dimension = identity.shape[0]
    generators, names = build_independent_two_spin_reverse_generators([jx, jy, jz], identity)
    local = build_reverse_generators(jx, jy, jz, identity, generator_set="no-twist")
    assert len(names) == 16
    parameters = .5 * torch.randn(2, 1, 16, dtype=toy.RDTYPE,
                                  generator=torch.Generator().manual_seed(451))
    first = identity/dimension + .1*jz
    second = identity/dimension + .1*jx
    joint = torch.kron(first.contiguous(), second.contiguous())
    # The full-space HS normalization introduces the spectator sqrt(2d).
    for theta in parameters:
        joint = toy.reverse_collision_channel(joint, theta, generators)
        first = toy.reverse_collision_channel(first, theta[:, :8]/np.sqrt(2*dimension), local)
        second = toy.reverse_collision_channel(second, theta[:, 8:]/np.sqrt(2*dimension), local)
        torch.testing.assert_close(joint, torch.kron(first.contiguous(), second.contiguous()), atol=1e-12, rtol=0)
        toy.assert_physical_density(joint, "independent product output")


def test_local_parameters_receive_gradients_and_supplied_initialization_is_preserved():
    jx, jy, jz, identity = toy.spin_operators(.5)
    generators, _ = build_independent_two_spin_reverse_generators([jx, jy, jz], identity)
    prior = torch.eye(4, dtype=toy.CDTYPE)/4
    target = torch.kron((identity/2+.2*jz).contiguous(), (identity/2+.1*jx).contiguous())
    initial = .1*torch.randn(1, 1, 16, dtype=toy.RDTYPE)
    before = initial.clone()
    diagnostics = []
    final, _ = toy.train_reverse([target, prior], generators, 1, 1, .04, 1,
        initial_parameters=initial, objective="fidelity", supervision="final", training_diagnostics=diagnostics)
    torch.testing.assert_close(initial, before)
    assert diagnostics[0]["gradient_norm"] > 0
    assert torch.linalg.norm(final[..., :8]-initial[..., :8]) > 0
    assert torch.linalg.norm(final[..., 8:]-initial[..., 8:]) > 0
    with pytest.raises(ValueError, match="shape"):
        toy.train_reverse([target, prior], generators, 1, 1, .04, 1, initial_parameters=initial[..., :8])


def test_shared_models_pair_initialization_by_generator_name():
    jx, jy, jz, identity = toy.spin_operators(1.)
    _, no_direct = build_two_spin_reverse_generators([jx, jy, jz], identity, False)
    _, direct = build_two_spin_reverse_generators([jx, jy, jz], identity, True)
    first = initial_angles(101, MODELS[1], no_direct, direct)
    second = initial_angles(101, MODELS[2], direct, direct)
    torch.testing.assert_close(first, second[:, :, [direct.index(n) for n in no_direct]], atol=0, rtol=0)


def test_gate_requires_correlation_improvement_and_freezes_test_candidate():
    rows = []
    for seed in (101,103,107,109,113):
        for model, trace in zip(MODELS, (.4,.3,.35)):
            values = {"trace_distance": trace, "infidelity": .2,
                      "connected_correlation_error": .2 if model == MODELS[0] else .1,
                      "mutual_information_absolute_error": .1, "marginal_trace_distance_mean": .1}
            rows.append({"seed": seed, "model": model, "physicality_passed": True,
                         "validation": values.copy(), "test": values.copy()})
    assert summary(rows)["go_to_confirmation"]
    assert summary(rows, "test", MODELS[2])["validation_candidate"] == MODELS[2]
    for row in rows:
        if row["model"] != MODELS[0]:
            row["validation"]["connected_correlation_error"] = .2
    assert not summary(rows)["go_to_confirmation"]
