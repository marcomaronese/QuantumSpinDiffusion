import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import spin_diffusion_toy as toy
from spin_config import ExperimentConfig
from spin_losses import fidelity_loss, multipole_loss, objective_loss
from spin_multipoles import irreducible_spherical_tensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from reverse_diagnostic import match_each_mode, rank_diagnostics


@pytest.mark.parametrize('start', ['mixed', 'forward'])
@pytest.mark.parametrize('supervision', ['path', 'final'])
def test_training_uses_declared_start_and_loss_terms(start, supervision):
    jx, jy, jz, identity = toy.spin_operators(1)
    target = toy.empirical_density(toy.make_direction_dataset(30, .22, 5), 1)
    forward = toy.build_forward_trajectory(target, jx, jy, jz, 1, .35, 2)
    generators = toy.build_reverse_generators(jx, jy, jz, identity)
    torch.manual_seed(5)
    initial_parameters = .03 * torch.randn(2, 1, len(generators), dtype=toy.RDTYPE)
    prior = identity / 3 if start == 'mixed' else forward[-1]
    trajectory = toy.generate_trajectory(prior, initial_parameters, generators)
    expected = sum(toy.frobenius_loss(trajectory[i+1], forward[1-i])
                   for i in range(2) if supervision == 'path' or i == 1)
    params, history = toy.train_reverse(forward, generators, 1, 1, .04, 5,
                                       starting_state=start, supervision=supervision)
    assert history[0] == pytest.approx(float(expected), abs=1e-13)
    assert torch.isfinite(params).all()
    assert not torch.equal(params, initial_parameters)


def test_default_training_is_unchanged():
    jx, jy, jz, identity = toy.spin_operators(.5)
    target = toy.empirical_density(toy.make_direction_dataset(20, .22, 7), .5)
    forward = toy.build_forward_trajectory(target, jx, jy, jz, 1, .35, 2)
    generators = toy.build_reverse_generators(jx, jy, jz, identity)
    default = toy.train_reverse(forward, generators, 1, 2, .04, 7)
    explicit = toy.train_reverse(forward, generators, 1, 2, .04, 7,
                                 starting_state='mixed', supervision='path')
    torch.testing.assert_close(default[0], explicit[0], rtol=0, atol=0)
    assert default[1] == explicit[1]


def test_unmatched_weak_mode_has_no_recovered_height():
    modes = [dict(theta=1., phi=0., relative_height=1.),
             dict(theta=2., phi=0., relative_height=.7)]
    matches = match_each_mode(modes, modes[:1])
    assert matches[0]['recovered']
    assert not matches[1]['recovered']
    assert matches[1]['angle_radians'] is None
    assert matches[1]['recovered_relative_height'] is None
    assert not any(m['recovered'] for m in match_each_mode(modes, []))


def test_rank_errors_obey_parseval_and_detect_wrong_orientation():
    jx, jy, _, _ = toy.spin_operators(2.5)
    tensors = irreducible_spherical_tensors(2.5, jx, jy)
    a = np.eye(6) / 6 + .01 * tensors[(4, 0)].numpy()
    b = np.eye(6) / 6 - .01 * tensors[(4, 0)].numpy()
    ranks = rank_diagnostics(a, b, tensors)
    assert sum(r['squared_coefficient_error'] for r in ranks.values()) == pytest.approx(np.linalg.norm(a-b)**2)
    assert ranks['4']['power_ratio'] == pytest.approx(1)
    assert ranks['4']['relative_coefficient_error'] == pytest.approx(2)


@pytest.mark.parametrize('weight', [0., .25, 1.])
def test_hybrid_formula_and_gradient(weight):
    jx, jy, _, identity = toy.spin_operators(1.)
    tensors = irreducible_spherical_tensors(1., jx, jy)
    target = identity / 3 + .05 * jx
    predicted = (identity / 3 + .03 * jy).requires_grad_()
    loss = objective_loss(predicted, target, 'hybrid', tensors=tensors, hybrid_lambda=weight)
    expected = (1-weight) * fidelity_loss(predicted, target) + weight * multipole_loss(predicted, target, tensors, 'high-rank')
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


@pytest.mark.parametrize('kwargs', [dict(starting_state='bad'), dict(supervision='bad'), dict(hybrid_lambda=1.1)])
def test_invalid_diagnostic_config(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs).validate()


def test_hybrid_selection_enforces_guardrails_and_uses_validation_seeds_only():
    from hybrid_objective_study import choose_lambda, LAMBDAS, VALIDATION_SEEDS
    rows = []
    for weight in LAMBDAS:
        for seed in VALIDATION_SEEDS:
            rows.append(dict(seed=seed, hybrid_lambda=weight, target_mode_count=2,
                             all_target_modes_recovered=weight != 0,
                             infidelity=.02 if weight != .75 else .1,
                             trace_distance=.03, q_js_divergence=.001,
                             ranks={'4': {'squared_coefficient_error': .01*(1-weight)},
                                    '5': {'squared_coefficient_error': .01*(1-weight)}}))
    result = choose_lambda(rows)
    assert result['selected_lambda'] == .5
    assert .75 not in result['eligible_lambdas']
    rows[0]['seed'] = 41
    with pytest.raises(ValueError):
        choose_lambda(rows)
