"""Shared artifact and metric integration for the classical diffusion benchmark."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .diffusion import save_classical_model, train_classical_diffusion


def run_classical_benchmark(points, config, seed, n_samples, output):
    """Train only on the supplied data and persist everything needed to resample."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, history = train_classical_diffusion(points, config, seed)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    samples = model.sample(n_samples, seed + 1)
    sampling_seconds = time.perf_counter() - started
    metadata = {
        "model": "classical_ddpm",
        "config": asdict(config),
        "seed": seed,
        "sampling_seed": seed + 1,
        "n_train": len(points),
        "n_generated": n_samples,
        "data_sha256": hashlib.sha256(np.asarray(points, dtype=np.float64).tobytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("diffusion.py", "benchmark.py")
        },
        "dimensions": model.dimensions,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer_updates": config.epochs * math.ceil(len(points) / config.batch_size),
        "training_seconds": training_seconds,
        "sampling_seconds": sampling_seconds,
        "initial_training_loss": history[0],
        "final_training_loss": history[-1],
        "terminal_alpha_bar": float(model.alpha_bar[-1]),
        "protocol": "Cartesian epsilon-prediction DDPM; cosine schedule; clipped x0; "
                    "fixed posterior variance; Gaussian prior; final sphere projection",
        "comparison": "Same training data and spin encoding; separate fixed classical budget. "
                      "Neither parameter counts nor optimizer updates are matched. "
                      "Encoded generated states have finite-sample error.",
    }
    save_classical_model(model, output / "model.pt")
    np.save(output / "training_theta_phi.npy", np.asarray(points))
    np.save(output / "generated_theta_phi.npy", samples)
    np.save(output / "training_history.npy", np.asarray(history))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return samples, metadata


def single_spin_benchmark(points, target, experiment, output):
    """Use exactly the quantum run's target, spin resolution, and Q grid."""
    from ..quantum import single_spin as toy

    samples, metadata = run_classical_benchmark(
        points, experiment.classical, experiment.seed, experiment.n_data, output)
    started = time.perf_counter()
    generated = toy.empirical_density(samples, experiment.j)
    metadata["encoding_seconds"] = time.perf_counter() - started
    physical = toy.assert_physical_density(generated, "classical DDPM encoded state")
    th, ph, _, p = toy.husimi_q_grid(target, experiment.j,
                                    experiment.q_grid_theta, experiment.q_grid_phi)
    _, _, generated_q, q = toy.husimi_q_grid(generated, experiment.j,
                                            experiment.q_grid_theta, experiment.q_grid_phi)
    p, q = p.ravel() + 1e-14, q.ravel() + 1e-14
    p, q = p / p.sum(), q / q.sum()
    mixture = .5 * (p + q)
    metrics = {
        **metadata,
        "evaluation_target": "empirical training density (same target as quantum run)",
        "final_frobenius_state_error": float(toy.frobenius_loss(generated, target)),
        "final_quantum_fidelity": float(toy.quantum_fidelity(generated, target)),
        "final_trace_distance": float(toy.trace_distance(generated, target)),
        "grid_jensen_shannon_divergence": float(.5 * (p * np.log(p / mixture)).sum()
                                              + .5 * (q * np.log(q / mixture)).sum()),
        "physicality": physical,
    }
    output = Path(output)
    np.save(output / "rho_generated.npy", generated.numpy())
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    toy.plot_q_map(th, ph, generated_q, "Classical DDPM (encoded Husimi Q)", output / "generated_Q.png")
    toy.plot_training(np.load(output / "training_history.npy"), output / "training_loss.png",
                      title="Classical DDPM training", ylabel="noise-prediction MSE")
    return metrics


def require_classical_benchmark(directory):
    """Prevent legacy cached runs from silently omitting the new benchmark."""
    directory = Path(directory)
    metrics = json.loads((directory / "validation_metrics.json").read_text())
    if "classical" not in metrics or not (directory / "classical" / "metrics.json").exists():
        raise ValueError(f"{directory} predates the classical benchmark; use a fresh output "
                         "directory or rerun the training study with --force")


def write_classical_comparison(directories, output):
    """Save paired results without pooling repeated seeds across quantum variants."""
    rows = []
    metric_names = ("final_frobenius_state_error", "final_trace_distance",
                    "grid_jensen_shannon_divergence")
    for directory in sorted(map(Path, directories)):
        require_classical_benchmark(directory)
        config = json.loads((directory / "config.json").read_text())
        metrics = json.loads((directory / "validation_metrics.json").read_text())
        quantum, classical = metrics["reverse"], metrics["classical"]
        q_values = {name: quantum[name] for name in metric_names}
        c_values = {name: classical[name] for name in metric_names}
        q_values["infidelity"] = 1 - quantum["final_quantum_fidelity"]
        c_values["infidelity"] = 1 - classical["final_quantum_fidelity"]
        rows.append({
            "run_directory": str(directory), "seed": config["seed"],
            "quantum_config": config, "classical_config": classical["config"],
            "quantum": q_values, "classical": c_values,
            "quantum_minus_classical": {name: q_values[name] - c_values[name] for name in q_values},
            "quantum_parameter_count": metrics["architecture"]["trainable_parameter_count"],
            "classical_parameter_count": classical["trainable_parameter_count"],
            "quantum_training_seconds": quantum["training_seconds"],
            "classical_training_seconds": classical["training_seconds"],
        })
    Path(output).write_text(json.dumps({
        "comparison": "Paired by dataset/seed. Lower errors are better; positive differences "
                      "favor classical. Repeated seeds across configurations are not independent trials.",
        "runs": rows,
    }, indent=2, allow_nan=False))
