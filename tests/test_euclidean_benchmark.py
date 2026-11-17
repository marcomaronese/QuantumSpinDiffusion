import numpy as np
import pytest
import torch

from spin_quantum_diffusion.classical.synthetic import TARGETS, sample_discrepancies, sample_target
from spin_quantum_diffusion.quantum.coherent_readout import coherent_povm
from spin_quantum_diffusion.quantum.euclidean import EuclideanSpinChart
from tasks.synthetic_benchmark import encoded_density


@pytest.mark.parametrize("dimensions", [1, 2])
def test_chart_round_trip_clipping_and_training_only_scale(dimensions):
    chart = EuclideanSpinChart(np.zeros(dimensions), np.ones(dimensions), radius=3)
    raw = np.zeros((3, dimensions))
    raw[:, 0] = [.5, -1, 6]
    directions, clipped = chart.encode(raw)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1)
    np.testing.assert_array_equal(clipped, [False, False, True])
    np.testing.assert_allclose(chart.decode(directions)[:, 0], [.5, -1, 3])
    training = np.arange(1, 21).reshape(-1, dimensions)
    fitted = EuclideanSpinChart.fit(training)
    before = fitted.to_dict()
    fitted.encode(training+1000)
    assert fitted.to_dict() == before
    np.testing.assert_allclose(fitted.location, training.mean(0))


def test_spherical_povm_is_complete_on_an_asymmetric_euclidean_target():
    chart = EuclideanSpinChart(np.array([0., 0.]), np.array([1., 1.]))
    directions, _ = chart.encode(np.array([[.3, .7], [1.2, -.8], [-.2, 1.1]]))
    rho = encoded_density(directions)
    povm = coherent_povm(5, 12, 24)
    probability = povm.probabilities(rho)
    np.testing.assert_allclose(probability.sum(), 1, atol=1e-12)
    assert probability.min() >= 0
    # A spin-coherent POVM attenuates the first moment by j/(j+1).
    observed = probability @ povm.directions
    torch.testing.assert_close(observed, torch.from_numpy(directions.mean(0)*(2.5/3.5)), rtol=0, atol=1e-12)


def test_metrics_have_correct_translation_and_empirical_measure_semantics():
    values = np.arange(100, dtype=float)[:, None]/100
    assert sample_discrepancies(values, values) == {"sliced_w1": 0., "energy_v": 0.}
    assert sample_discrepancies(values, values+2)["sliced_w1"] == pytest.approx(2)
    permuted = values[np.random.default_rng(3).permutation(len(values))]
    assert sample_discrepancies(values, permuted)["sliced_w1"] == pytest.approx(0)
    assert sample_discrepancies(values, permuted)["energy_v"] == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("name", TARGETS)
def test_targets_have_declared_dimensions_and_isolated_reproducible_seeds(name):
    first = sample_target(name, 32, 999)
    np.testing.assert_array_equal(first, sample_target(name, 32, 999))
    assert first.shape == (32, TARGETS[name]["dimensions"])
    assert np.isfinite(first).all()
    assert not np.array_equal(first, sample_target(name, 32, 1000))


def test_invalid_chart_cannot_hide_degenerate_scales_or_dimension_mismatch():
    with pytest.raises(ValueError):
        EuclideanSpinChart.fit(np.ones((10, 1)))
    chart = EuclideanSpinChart(np.zeros(1), np.ones(1))
    with pytest.raises(ValueError):
        chart.encode(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        chart.decode(np.zeros((5, 3)))
