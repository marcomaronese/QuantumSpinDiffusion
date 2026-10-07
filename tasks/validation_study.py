#!/usr/bin/env python3
"""Run the five-seed, representation-resolution, and objective study.

All training configurations use the same data size, architecture, optimizer,
learning rate, diffusion schedule, and epoch budget. Only the requested loss
objective (and its declared multipole weighting) changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.ndimage import maximum_filter
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from spin_quantum_diffusion.quantum import single_spin as toy  # noqa: E402
from spin_quantum_diffusion.classical.benchmark import (  # noqa: E402
    require_classical_benchmark,
    write_classical_comparison,
)


OBJECTIVE_CONFIGURATIONS = {
    "frobenius": ("frobenius", "rank-balanced"),
    "multipole_rank_balanced": ("multipole", "rank-balanced"),
    "multipole_high_rank": ("multipole", "high-rank"),
    "fidelity": ("fidelity", "rank-balanced"),
    "husimi_js": ("husimi-js", "rank-balanced"),
}

METRIC_PATHS = {
    "frobenius_error": "final_frobenius_state_error",
    "trace_distance": "final_trace_distance",
    "infidelity": "final_quantum_fidelity",
    "q_js_divergence": "grid_jensen_shannon_divergence",
    "q_correlation": "q_grid_correlation",
    "q_peak_shift": "q_peak_angular_shift_radians",
    "rank_balanced_multipole_loss": "final_rank_balanced_multipole_loss",
    "high_rank_multipole_loss": "final_high_rank_multipole_loss",
}


def parse_number_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def run_training_job(
    output_root: Path,
    label: str,
    objective: str,
    weighting: str,
    seed: int,
    epochs: int,
    force: bool,
) -> Path:
    """Run one isolated CLI experiment and return its metrics path."""
    run_directory = output_root / "runs" / label / f"seed_{seed}"
    metrics_path = run_directory / "validation_metrics.json"
    if metrics_path.exists() and not force:
        require_classical_benchmark(run_directory)
        return metrics_path

    run_directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "spin_quantum_diffusion.quantum.single_spin",
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--objective",
        objective,
        "--multipole-weighting",
        weighting,
        "--outdir",
        str(run_directory),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(SRC) + os.pathsep + environment.get("PYTHONPATH", ""),
            "MPLCONFIGDIR": str(output_root / ".matplotlib"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    (run_directory / "run.log").write_text(
        result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"training failed for {label}, seed {seed}; see {run_directory / 'run.log'}"
        )
    if not metrics_path.exists():
        raise RuntimeError(f"training did not produce {metrics_path}")
    return metrics_path


def load_training_rows(
    output_root: Path,
    seeds: list[int],
) -> list[dict[str, float | int | str]]:
    rows = []
    for label in OBJECTIVE_CONFIGURATIONS:
        for seed in seeds:
            metrics_path = (
                output_root
                / "runs"
                / label
                / f"seed_{seed}"
                / "validation_metrics.json"
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            reverse = metrics["reverse"]
            independent = independently_recompute_state_metrics(
                metrics_path.parent / "rho_data.npy",
                metrics_path.parent / "rho_generated.npy",
                j=2.0,
            )
            declared = {
                "frobenius_error": reverse["final_frobenius_state_error"],
                "trace_distance": reverse["final_trace_distance"],
                "infidelity": 1.0 - reverse["final_quantum_fidelity"],
                "q_js_divergence": reverse["grid_jensen_shannon_divergence"],
                "q_correlation": reverse["q_grid_correlation"],
                "q_peak_shift": reverse["q_peak_angular_shift_radians"],
            }
            for metric_name, declared_value in declared.items():
                if not math.isclose(
                    float(declared_value),
                    independent[metric_name],
                    rel_tol=2e-9,
                    abs_tol=2e-11,
                ):
                    raise AssertionError(
                        f"independent {metric_name} check failed for "
                        f"{label}, seed {seed}: {declared_value} vs "
                        f"{independent[metric_name]}"
                    )
            row: dict[str, float | int | str] = {
                "configuration": label,
                "seed": seed,
                "initial_training_loss": reverse["initial_training_loss"],
                "final_training_loss": reverse["final_training_loss"],
                "max_trace_error": reverse["max_trace_error"],
                "minimum_eigenvalue": reverse["minimum_eigenvalue"],
                "max_trace_preservation_error": reverse[
                    "max_trace_preservation_error"
                ],
                "minimum_choi_eigenvalue": reverse["minimum_choi_eigenvalue"],
                "multipole_decay_error": metrics["forward"][
                    "max_multipole_decay_error"
                ],
            }
            for output_name, source_name in METRIC_PATHS.items():
                if output_name in independent:
                    row[output_name] = independent[output_name]
                else:
                    row[output_name] = float(reverse[source_name])
            rows.append(row)
    return rows


def independently_recompute_state_metrics(
    target_path: Path,
    generated_path: Path,
    j: float,
) -> dict[str, float]:
    """Recompute primary outcome metrics without the training loss helpers."""
    target = np.load(target_path)
    generated = np.load(generated_path)
    difference = generated - target
    frobenius_error = float(np.sum(np.abs(difference) ** 2).real)
    trace_distance_value = 0.5 * float(
        np.sum(np.abs(np.linalg.eigvalsh(difference)))
    )

    target_root = sqrtm(target)
    fidelity_sandwich = target_root @ generated @ target_root
    fidelity_sandwich = 0.5 * (
        fidelity_sandwich + fidelity_sandwich.conj().T
    )
    fidelity_eigenvalues = np.linalg.eigvalsh(fidelity_sandwich)
    fidelity = float(
        np.sum(np.sqrt(np.clip(fidelity_eigenvalues, 0.0, None))) ** 2
    )

    thetas = np.linspace(1e-5, np.pi - 1e-5, 80)
    phis = np.linspace(0.0, 2 * np.pi, 160, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing="ij")
    flat_theta = theta_grid.reshape(-1)
    flat_phi = phi_grid.reshape(-1)
    N = int(round(2 * j))
    amplitudes = []
    for k in range(N + 1):
        amplitudes.append(
            math.sqrt(math.comb(N, k))
            * np.cos(flat_theta / 2) ** (N - k)
            * np.sin(flat_theta / 2) ** k
            * np.exp(1j * k * flat_phi)
        )
    states = np.stack(amplitudes, axis=1)
    states = states / np.linalg.norm(states, axis=1, keepdims=True)

    def q_grid(rho: np.ndarray) -> np.ndarray:
        expectations = np.einsum(
            "gi,ij,gj->g",
            states.conj(),
            rho,
            states,
        ).real
        return ((2 * j + 1) / (4 * np.pi) * expectations).reshape(80, 160)

    target_q = q_grid(target)
    generated_q = q_grid(generated)
    target_mass = target_q * np.sin(thetas)[:, None]
    generated_mass = generated_q * np.sin(thetas)[:, None]
    target_mass = target_mass / target_mass.sum()
    generated_mass = generated_mass / generated_mass.sum()
    epsilon = 1e-14
    p = target_mass.reshape(-1) + epsilon
    q = generated_mass.reshape(-1) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    q_js_divergence = float(
        0.5 * np.sum(p * np.log(p / midpoint))
        + 0.5 * np.sum(q * np.log(q / midpoint))
    )
    q_correlation = float(
        np.corrcoef(target_q.reshape(-1), generated_q.reshape(-1))[0, 1]
    )
    target_peak = np.unravel_index(np.argmax(target_q), target_q.shape)
    generated_peak = np.unravel_index(np.argmax(generated_q), generated_q.shape)
    target_vector = direction_vector(
        float(thetas[target_peak[0]]),
        float(phis[target_peak[1]]),
    )
    generated_vector = direction_vector(
        float(thetas[generated_peak[0]]),
        float(phis[generated_peak[1]]),
    )
    q_peak_shift = math.acos(
        float(np.clip(target_vector @ generated_vector, -1.0, 1.0))
    )
    return {
        "frobenius_error": frobenius_error,
        "trace_distance": trace_distance_value,
        "infidelity": 1.0 - fidelity,
        "q_js_divergence": q_js_divergence,
        "q_correlation": q_correlation,
        "q_peak_shift": q_peak_shift,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict], seeds: list[int]) -> dict:
    summary = {
        "seeds": seeds,
        "run_count": len(rows),
        "configurations": {},
    }
    for label in OBJECTIVE_CONFIGURATIONS:
        selected = [row for row in rows if row["configuration"] == label]
        configuration = {}
        for metric in METRIC_PATHS:
            values = [float(row[metric]) for row in selected]
            configuration[metric] = {
                "mean": statistics.fmean(values),
                "sample_standard_deviation": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "minimum": min(values),
                "maximum": max(values),
            }
        configuration["all_losses_decreased"] = all(
            float(row["final_training_loss"])
            < float(row["initial_training_loss"])
            for row in selected
        )
        summary["configurations"][label] = configuration

    summary["global_validation"] = {
        "maximum_trace_error": max(float(row["max_trace_error"]) for row in rows),
        "minimum_state_eigenvalue": min(
            float(row["minimum_eigenvalue"]) for row in rows
        ),
        "maximum_trace_preservation_error": max(
            float(row["max_trace_preservation_error"]) for row in rows
        ),
        "minimum_choi_eigenvalue": min(
            float(row["minimum_choi_eigenvalue"]) for row in rows
        ),
        "maximum_multipole_decay_error": max(
            float(row["multipole_decay_error"]) for row in rows
        ),
    }
    by_configuration = {
        label: {
            int(row["seed"]): row
            for row in rows
            if row["configuration"] == label
        }
        for label in OBJECTIVE_CONFIGURATIONS
    }
    baseline = by_configuration["frobenius"]
    paired_metrics = [
        "frobenius_error",
        "trace_distance",
        "infidelity",
        "q_js_divergence",
        "high_rank_multipole_loss",
    ]
    paired_comparisons = {}
    for label, selected in by_configuration.items():
        if label == "frobenius":
            continue
        comparison = {}
        for metric in paired_metrics:
            differences = [
                float(selected[seed][metric]) - float(baseline[seed][metric])
                for seed in seeds
            ]
            mean_difference = statistics.fmean(differences)
            if len(differences) > 1:
                standard_deviation = statistics.stdev(differences)
                margin = float(student_t.ppf(0.975, len(seeds) - 1)) * (
                    standard_deviation / math.sqrt(len(seeds))
                )
            else:
                margin = 0.0
            comparison[metric] = {
                "mean_candidate_minus_frobenius": mean_difference,
                "paired_95_percent_t_interval": [
                    mean_difference - margin,
                    mean_difference + margin,
                ],
                "lower_is_better_wins": sum(
                    difference < 0.0 for difference in differences
                ),
                "paired_run_count": len(differences),
            }
        paired_comparisons[label] = comparison
    summary["paired_vs_frobenius"] = paired_comparisons
    return summary


def direction_vector(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def significant_q_modes(
    thetas: np.ndarray,
    phis: np.ndarray,
    Q: np.ndarray,
    relative_threshold: float = 0.5,
    minimum_separation: float = 0.35,
) -> list[dict[str, float]]:
    """Find separated grid-local maxima above a declared relative threshold."""
    neighborhood_max = maximum_filter(
        Q,
        size=(5, 5),
        mode=("nearest", "wrap"),
    )
    candidates = np.argwhere(
        (Q >= neighborhood_max - 1e-14)
        & (Q >= relative_threshold * float(Q.max()))
    )
    ordered = sorted(
        ((float(Q[i, k]), int(i), int(k)) for i, k in candidates),
        reverse=True,
    )
    modes: list[dict[str, float]] = []
    vectors: list[np.ndarray] = []
    for value, theta_index, phi_index in ordered:
        theta = float(thetas[theta_index])
        phi = float(phis[phi_index])
        vector = direction_vector(theta, phi)
        if any(
            math.acos(float(np.clip(vector @ existing, -1.0, 1.0)))
            < minimum_separation
            for existing in vectors
        ):
            continue
        modes.append(
            {
                "theta": theta,
                "phi": phi,
                "q_value": value,
                "relative_height": value / float(Q.max()),
            }
        )
        vectors.append(vector)
    return modes


def run_representation_sweep(
    output_root: Path,
    j_values: list[float],
) -> list[dict]:
    points = toy.make_direction_dataset(n_samples=2000, sigma=0.22, seed=7)
    rows = []
    selected_maps = {}
    for j in j_values:
        rho = toy.empirical_density(points, j)
        thetas, phis, Q, _ = toy.husimi_q_grid(
            rho,
            j,
            n_theta=80,
            n_phi=160,
        )
        modes = significant_q_modes(thetas, phis, Q)
        separation = None
        if len(modes) >= 2:
            first = direction_vector(modes[0]["theta"], modes[0]["phi"])
            second = direction_vector(modes[1]["theta"], modes[1]["phi"])
            separation = math.acos(float(np.clip(first @ second, -1.0, 1.0)))
        rows.append(
            {
                "j": j,
                "dimension": int(round(2 * j + 1)),
                "significant_mode_count": len(modes),
                "top_two_separation_radians": separation,
                "modes": modes,
            }
        )
        if j in {2.0, 2.5}:
            selected_maps[j] = (thetas, phis, Q)

    for j, (thetas, phis, Q) in selected_maps.items():
        toy.plot_q_map(
            thetas,
            phis,
            Q,
            f"Encoded target Husimi Q (j={j:g})",
            output_root / f"encoded_target_Q_j{str(j).replace('.', 'p')}.png",
        )
    return rows


def plot_baseline_reproducibility(rows: list[dict], output_file: Path) -> None:
    selected = [row for row in rows if row["configuration"] == "frobenius"]
    seeds = [int(row["seed"]) for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    frobenius_values = [row["frobenius_error"] for row in selected]
    q_js_values = [row["q_js_divergence"] for row in selected]
    axes[0].scatter(seeds, frobenius_values, color="#2F6B9A", s=60)
    axes[0].axhline(
        statistics.fmean(frobenius_values),
        color="#3D3D3D",
        linestyle="--",
        linewidth=1.2,
        label="five-seed mean",
    )
    axes[0].set_ylabel("final Frobenius error")
    axes[1].scatter(seeds, q_js_values, color="#2F6B9A", s=60)
    axes[1].axhline(
        statistics.fmean(q_js_values),
        color="#3D3D3D",
        linestyle="--",
        linewidth=1.2,
        label="five-seed mean",
    )
    axes[1].set_ylabel("Q-grid Jensen–Shannon divergence")
    for axis in axes:
        axis.set_xlabel("seed")
        axis.set_xticks(seeds)
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.suptitle("Five-seed reproducibility of the validated baseline")
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_objective_comparison(rows: list[dict], output_file: Path) -> None:
    labels = list(OBJECTIVE_CONFIGURATIONS)
    display_labels = [
        "Frobenius",
        "Multipole\nrank-balanced",
        "Multipole\nhigh-rank",
        "Fidelity",
        "Husimi JS",
    ]
    metrics = [
        ("frobenius_error", "Frobenius error"),
        ("trace_distance", "Trace distance"),
        ("infidelity", "Infidelity"),
        ("q_js_divergence", "Q-grid JS divergence"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for axis, (metric, title) in zip(axes.reshape(-1), metrics, strict=True):
        grouped = [
            [float(row[metric]) for row in rows if row["configuration"] == label]
            for label in labels
        ]
        means = [statistics.fmean(values) for values in grouped]
        deviations = [
            statistics.stdev(values) if len(values) > 1 else 0.0
            for values in grouped
        ]
        positions = np.arange(len(labels))
        axis.bar(
            positions,
            means,
            yerr=deviations,
            capsize=4,
            color="#2F6B9A",
            alpha=0.82,
        )
        for position, values in zip(positions, grouped, strict=True):
            axis.scatter(
                np.full(len(values), position),
                values,
                color="#2F2F2F",
                s=18,
                zorder=3,
            )
        axis.set_xticks(range(len(labels)), display_labels, fontsize=8)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Objective comparison across identical five-seed runs\n(mean ± sample SD; lower is better)")
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representation_sweep(rows: list[dict], output_file: Path) -> None:
    j_values = [float(row["j"]) for row in rows]
    counts = [int(row["significant_mode_count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(j_values, counts, color="#2F6B9A", s=65)
    ax.axvline(2.0, color="tab:red", linestyle="--", label="current default")
    ax.set_xlabel("spin j")
    ax.set_ylabel("significant encoded Q maxima")
    ax.set_xticks(j_values)
    ax.set_yticks(sorted(set(counts)))
    ax.set_title("Resolution of the two-mode target under finite-j encoding")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(
        0.5,
        0.01,
        "Grid criterion: local maximum ≥50% of global peak; separation ≥0.35 rad",
        ha="center",
        fontsize=8,
        color="#3D3D3D",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", default="3,5,7,11,13")
    parser.add_argument("--j-values", default="0.5,1,1.5,2,2.5,3,4,5")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "experiments" / "validation_study",
    )
    args = parser.parse_args()

    seeds = parse_number_list(args.seeds, int)
    j_values = parse_number_list(args.j_values, float)
    output_root = args.outdir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for label, (objective, weighting) in OBJECTIVE_CONFIGURATIONS.items():
            for seed in seeds:
                jobs.append(
                    executor.submit(
                        run_training_job,
                        output_root,
                        label,
                        objective,
                        weighting,
                        seed,
                        args.epochs,
                        args.force,
                    )
                )
        for completed in as_completed(jobs):
            completed.result()

    rows = load_training_rows(output_root, seeds)
    write_classical_comparison(
        [output_root / "runs" / label / f"seed_{seed}"
         for label in OBJECTIVE_CONFIGURATIONS for seed in seeds],
        output_root / "classical_comparison.json")
    write_csv(output_root / "training_runs.csv", rows)
    summary = aggregate_rows(rows, seeds)

    representation_rows = run_representation_sweep(output_root, j_values)
    write_csv(
        output_root / "representation_sweep.csv",
        [
            {
                "j": row["j"],
                "dimension": row["dimension"],
                "significant_mode_count": row["significant_mode_count"],
                "top_two_separation_radians": row[
                    "top_two_separation_radians"
                ],
            }
            for row in representation_rows
        ],
    )
    summary["representation_sweep"] = representation_rows
    (output_root / "study_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_baseline_reproducibility(
        rows,
        output_root / "five_seed_reproducibility.png",
    )
    plot_objective_comparison(
        rows,
        output_root / "objective_comparison.png",
    )
    plot_representation_sweep(
        representation_rows,
        output_root / "representation_resolution_sweep.png",
    )
    print(f"validation study written to {output_root}")


if __name__ == "__main__":
    main()
