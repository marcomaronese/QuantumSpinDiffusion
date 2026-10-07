"""Mathematical, learning, reproducibility, and experiment regression checks."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import spin_diffusion_toy as toy
from spin_classical_benchmark import require_classical_benchmark
from spin_classical_diffusion import (
    ClassicalDDPM, ClassicalDiffusionConfig, angles_to_cartesian,
    cartesian_to_angles, load_classical_model, save_classical_model,
    train_classical_diffusion,
)
from spin_correlated import make_correlated_direction_pairs


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dimensions", [3, 6])
def test_forward_gaussian_moments_and_terminal_prior(dimensions):
    model = ClassicalDDPM(dimensions)
    assert torch.all((model.beta > 0) & (model.beta < 1))
    assert torch.all(torch.diff(model.alpha_bar) < 0)
    assert model.alpha_bar[-1] < 1e-5
    rng = torch.Generator().manual_seed(91)
    clean = torch.full((50000, dimensions), .4, dtype=torch.float64)
    for step in (0, 31, 63):
        steps = torch.full((len(clean),), step, dtype=torch.long)
        noise = torch.randn(clean.shape, generator=rng, dtype=torch.float64)
        noisy = model.forward_sample(clean, steps, noise)
        expected_mean = model.alpha_bar[step].sqrt() * .4
        expected_variance = 1 - model.alpha_bar[step]
        torch.testing.assert_close(noisy.mean(0), expected_mean.expand(dimensions), atol=.015, rtol=0)
        torch.testing.assert_close(noisy.var(0), expected_variance.expand(dimensions), atol=.02, rtol=0)


def test_reverse_posterior_matches_gaussian_conditioning_and_final_step():
    model = ClassicalDDPM()
    clean = torch.tensor([[.3, -.2, .7]], dtype=torch.float64)
    noise = torch.tensor([[.4, .8, -.6]], dtype=torch.float64)
    for step in (0, 12, 63):
        steps = torch.tensor([step])
        noisy = model.forward_sample(clean, steps, noise)
        mean, variance = model.reverse_moments(noisy, steps, noise)
        previous = torch.tensor(1.) if step == 0 else model.alpha_bar[step - 1]
        # Condition the joint Gaussian (x_{t-1}, x_t) directly.
        prior_mean = previous.sqrt() * clean
        prior_variance = 1 - previous
        observation_variance = 1 - model.alpha_bar[step]
        cross_covariance = (1 - model.beta[step]).sqrt() * prior_variance
        expected = prior_mean + cross_covariance / observation_variance * (
            noisy - model.alpha_bar[step].sqrt() * clean)
        expected_variance = prior_variance - cross_covariance.square() / observation_variance
        torch.testing.assert_close(mean, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(variance.squeeze(), expected_variance, atol=1e-12, rtol=1e-12)
        if step == 0:
            assert variance.item() == 0
            torch.testing.assert_close(mean, clean)


@pytest.mark.parametrize("pairs", [False, True])
def test_training_sampling_checkpoint_reproducibility_and_rng_isolation(tmp_path, pairs):
    data = (make_correlated_direction_pairs(24, .2, .55, 5) if pairs
            else toy.make_direction_dataset(24, .2, 5))
    config = ClassicalDiffusionConfig(steps=8, epochs=3, batch_size=12, hidden_width=16)
    rng_before = torch.random.get_rng_state().clone()
    first, history = train_classical_diffusion(data, config, seed=4)
    second, repeated = train_classical_diffusion(data, config, seed=4)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert history == repeated
    assert np.isfinite(history).all()
    sample = first.sample(20, seed=8)
    np.testing.assert_array_equal(sample, second.sample(20, seed=8))
    assert not np.array_equal(sample, first.sample(20, seed=9))
    assert sample.shape == ((20, 2, 2) if pairs else (20, 2))
    assert np.isfinite(sample).all()
    vectors = angles_to_cartesian(sample).reshape(20, -1, 3)
    torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(20, 2 if pairs else 1))
    checkpoint = tmp_path / "model.pt"
    save_classical_model(first, checkpoint)
    loaded = load_classical_model(checkpoint)
    np.testing.assert_array_equal(sample, loaded.sample(20, seed=8))
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_learned_diffusion_improves_on_uniform_prior():
    data = toy.make_direction_dataset(256, .22, 5)
    config = ClassicalDiffusionConfig(steps=32, epochs=100, batch_size=128, hidden_width=32)
    model, history = train_classical_diffusion(data, config, seed=5)
    assert np.mean(history[-10:]) < .65 * np.mean(history[:10])
    generated = toy.empirical_density(model.sample(1000, seed=6), j=1.)
    # Evaluate on fresh data, not the examples optimized by the denoiser.
    target = toy.empirical_density(toy.make_direction_dataset(1000, .22, 105), j=1.)
    prior = torch.eye(3, dtype=toy.CDTYPE) / 3
    assert toy.frobenius_loss(generated, target) < .3 * toy.frobenius_loss(prior, target)
    toy.assert_physical_density(generated, "classical smoke-test density")


def test_joint_diffusion_learns_correlation_beyond_factorized_marginals():
    train = make_correlated_direction_pairs(256, .18, .55, 17)
    test = make_correlated_direction_pairs(1000, .18, .55, 1017)
    config = ClassicalDiffusionConfig(epochs=100, steps=32, batch_size=128, hidden_width=32)
    model, _ = train_classical_diffusion(train, config, seed=17)

    def correlation(points):
        vectors = angles_to_cartesian(points).reshape(-1, 2, 3)
        centered = vectors - vectors.mean(0)
        return centered[:, 0].T @ centered[:, 1] / len(centered)

    target = correlation(test)
    generated = correlation(model.sample(1000, seed=18))
    # A factorized distribution has zero connected cross-correlation.
    assert torch.linalg.norm(target - generated) < .8 * torch.linalg.norm(target)


@pytest.mark.parametrize("updates", [dict(steps=1), dict(epochs=0), dict(batch_size=-1),
                                     dict(hidden_width=0), dict(learning_rate=float("nan")),
                                     dict(learning_rate=0)])
def test_invalid_configuration(updates):
    with pytest.raises(ValueError):
        replace(ClassicalDiffusionConfig(), **updates).validate()


def test_direction_validation_and_roundtrip():
    with pytest.raises(ValueError):
        angles_to_cartesian([])
    with pytest.raises(ValueError):
        angles_to_cartesian([[float("nan"), 0]])
    with pytest.raises(ValueError):
        cartesian_to_angles(torch.zeros(1, 3))
    points = np.asarray(toy.make_direction_dataset(20, .2, 3))
    np.testing.assert_allclose(cartesian_to_angles(angles_to_cartesian(points)), points, atol=1e-14)


def test_cli_produces_comparable_classical_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / "spin_diffusion_toy.py"),
        "--j", "1", "--n-data", "32", "--T", "1", "--layers", "1", "--epochs", "2",
        "--q-grid-theta", "10", "--q-grid-phi", "20", "--classical-epochs", "3",
        "--classical-steps", "8", "--classical-hidden-width", "16", "--outdir", str(tmp_path),
    ], env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg"), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    require_classical_benchmark(tmp_path)
    metrics = json.loads((tmp_path / "validation_metrics.json").read_text())
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["classical"]["epochs"] == 3
    assert metrics["classical"]["n_train"] == config["n_data"]
    baseline = tmp_path / "classical"
    samples = np.load(baseline / "generated_theta_phi.npy")
    restored = load_classical_model(baseline / "model.pt")
    np.testing.assert_array_equal(samples, restored.sample(32, config["seed"] + 1))
    generated = torch.tensor(np.load(baseline / "rho_generated.npy"))
    torch.testing.assert_close(generated, toy.empirical_density(samples, 1.))
    target = torch.tensor(np.load(tmp_path / "rho_data.npy"))
    expected_error = float(torch.sum(torch.abs(generated - target).square()))
    assert metrics["classical"]["final_frobenius_state_error"] == pytest.approx(expected_error)
    comparison = json.loads((tmp_path / "classical_comparison.json").read_text())["runs"][0]
    assert comparison["quantum_minus_classical"]["final_frobenius_state_error"] == pytest.approx(
        metrics["reverse"]["final_frobenius_state_error"] - expected_error)


def test_old_cached_runs_cannot_silently_skip_benchmark(tmp_path):
    (tmp_path / "validation_metrics.json").write_text('{"reverse": {}}')
    with pytest.raises(ValueError, match="predates the classical benchmark"):
        require_classical_benchmark(tmp_path)


def test_two_spin_experiment_includes_and_independently_audits_ddpm(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    commands = [
        [str(root / "experiments/two_spin_correlated.py"), "--n-train", "32", "--n-test", "32",
         "--quantum-epochs", "1", "--neural-epochs", "1", "--classical-epochs", "2",
         "--classical-steps", "8", "--classical-hidden-width", "16", "--outdir", str(tmp_path)],
        [str(root / "experiments/validate_two_spin.py"), "--pilot-dir", str(tmp_path)],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, *command], env=environment,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads((tmp_path / "independent_validation.json").read_text())
    assert "Classical DDPM" in {row["model"] for row in audit["models"]}
    assert audit["maximum_absolute_discrepancy"] < 2e-9
    model = load_classical_model(tmp_path / "classical" / "model.pt")
    assert model.dimensions == 6
    np.testing.assert_array_equal(model.sample(32, seed=18),
                                  np.load(tmp_path / "classical" / "generated_theta_phi.npy"))
