#!/usr/bin/env python3
"""Compare deterministic reverse channels with sampled instrument trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spin_diffusion_toy as toy  # noqa: E402
from spin_reverse import (  # noqa: E402
    collision_kraus_operators,
    parameter_schedule,
    reverse_collision_unitary,
)


def trace_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    difference = 0.5 * (first - second + (first - second).conj().T)
    return 0.5 * float(torch.linalg.eigvalsh(difference).abs().sum())


def squared_frobenius(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.sum(torch.abs(first - second) ** 2))


def cached_kraus_schedule(
    parameters: torch.Tensor,
    generators: list[torch.Tensor],
    system_dimension: int,
    n_steps: int,
) -> list[list[torch.Tensor]]:
    schedule = []
    for theta in parameter_schedule(parameters, n_steps=n_steps):
        unitary = reverse_collision_unitary(theta, generators)
        schedule.append(collision_kraus_operators(unitary, system_dimension))
    return schedule


def sample_trajectories(
    prior: torch.Tensor,
    kraus_schedule: list[list[torch.Tensor]],
    n_trajectories: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[list[int]], list[float]]:
    random_generator = torch.Generator().manual_seed(seed)
    final_states = []
    all_outcomes = []
    branch_entropies = []
    for _ in range(n_trajectories):
        rho = prior
        outcomes = []
        for kraus in kraus_schedule:
            branches = [operator @ rho @ operator.conj().T for operator in kraus]
            probabilities = torch.stack(
                [torch.real(torch.trace(branch)) for branch in branches]
            )
            if float(probabilities.min()) < -1e-10:
                raise AssertionError("negative instrument probability")
            if abs(float(probabilities.sum()) - 1.0) > 1e-10:
                raise AssertionError("instrument probabilities do not sum to one")
            normalized_probabilities = probabilities / probabilities.sum()
            positive = normalized_probabilities > 0
            branch_entropies.append(
                -float(
                    torch.sum(
                        normalized_probabilities[positive]
                        * torch.log(normalized_probabilities[positive])
                    )
                )
            )
            outcome = int(
                torch.multinomial(
                    normalized_probabilities,
                    num_samples=1,
                    generator=random_generator,
                ).item()
            )
            probability = probabilities[outcome]
            if float(probability) <= 1e-12:
                raise AssertionError("sampled a numerically zero-probability branch")
            rho = branches[outcome] / probability
            toy.assert_physical_density(rho, "sampled instrument state")
            outcomes.append(outcome)
        final_states.append(rho)
        all_outcomes.append(outcomes)
    return final_states, all_outcomes, branch_entropies


def analyze_run(
    run_directory: Path,
    n_trajectories: int,
    study_seed: int,
) -> tuple[dict, list[dict]]:
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    j = float(config["j"])
    n_steps = int(config["diffusion_steps"])
    Jx, Jy, Jz, identity = toy.spin_operators(j)
    generators = toy.build_reverse_generators(
        Jx,
        Jy,
        Jz,
        identity,
        ancilla_count=int(config["ancilla_count"]),
        generator_set=str(config["generator_set"]),
    )
    parameters = torch.load(
        run_directory / "reverse_parameters.pt",
        map_location="cpu",
        weights_only=True,
    ).to(toy.RDTYPE)
    prior = identity / identity.shape[0]
    target = torch.tensor(np.load(run_directory / "rho_data.npy"), dtype=toy.CDTYPE)
    saved_deterministic = torch.tensor(
        np.load(run_directory / "rho_generated.npy"), dtype=toy.CDTYPE
    )
    replayed_deterministic = toy.generate_density(
        prior,
        parameters,
        generators,
        n_steps=n_steps,
    )
    replay_error = squared_frobenius(replayed_deterministic, saved_deterministic)
    if replay_error > 1e-18:
        raise AssertionError(f"saved deterministic state replay failed: {replay_error}")

    kraus_schedule = cached_kraus_schedule(
        parameters,
        generators,
        system_dimension=identity.shape[0],
        n_steps=n_steps,
    )
    final_states, outcomes, branch_entropies = sample_trajectories(
        prior,
        kraus_schedule,
        n_trajectories=n_trajectories,
        seed=study_seed + seed,
    )
    stacked = torch.stack(final_states)
    empirical_average = stacked.mean(dim=0)
    trajectory_fidelities = [
        float(toy.quantum_fidelity(state, target)) for state in final_states
    ]
    trajectory_trace_distances = [
        trace_distance(state, target) for state in final_states
    ]
    diversity = [
        squared_frobenius(state, empirical_average) for state in final_states
    ]

    convergence_rows = []
    sample_sizes = [
        size
        for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        if size <= n_trajectories
    ]
    if sample_sizes[-1] != n_trajectories:
        sample_sizes.append(n_trajectories)
    cumulative = torch.zeros_like(prior)
    sample_size_set = set(sample_sizes)
    for index, state in enumerate(final_states, start=1):
        cumulative = cumulative + state
        if index in sample_size_set:
            estimate = cumulative / index
            convergence_rows.append(
                {
                    "seed": seed,
                    "trajectory_count": index,
                    "frobenius_error_to_deterministic": squared_frobenius(
                        estimate, replayed_deterministic
                    ),
                    "trace_distance_to_deterministic": trace_distance(
                        estimate, replayed_deterministic
                    ),
                }
            )

    flattened_outcomes = np.asarray(outcomes, dtype=int).reshape(-1)
    outcome_count = len(kraus_schedule[0])
    outcome_frequencies = np.bincount(
        flattened_outcomes,
        minlength=outcome_count,
    ) / len(flattened_outcomes)
    row = {
        "seed": seed,
        "trajectory_count": n_trajectories,
        "instrument_outcome_count": outcome_count,
        "deterministic_replay_frobenius_error": replay_error,
        "ensemble_frobenius_error_to_deterministic": squared_frobenius(
            empirical_average, replayed_deterministic
        ),
        "ensemble_trace_distance_to_deterministic": trace_distance(
            empirical_average, replayed_deterministic
        ),
        "deterministic_fidelity_to_target": float(
            toy.quantum_fidelity(replayed_deterministic, target)
        ),
        "mean_trajectory_fidelity_to_target": statistics.mean(
            trajectory_fidelities
        ),
        "trajectory_fidelity_sample_sd": statistics.stdev(
            trajectory_fidelities
        ),
        "deterministic_trace_distance_to_target": trace_distance(
            replayed_deterministic, target
        ),
        "mean_trajectory_trace_distance_to_target": statistics.mean(
            trajectory_trace_distances
        ),
        "trajectory_trace_distance_sample_sd": statistics.stdev(
            trajectory_trace_distances
        ),
        "mean_trajectory_diversity_squared_frobenius": statistics.mean(
            diversity
        ),
        "mean_branch_entropy_nats": statistics.mean(branch_entropies),
        "minimum_trajectory_eigenvalue": min(
            float(torch.linalg.eigvalsh(state).min()) for state in final_states
        ),
        "maximum_trajectory_trace_error": max(
            abs(float(torch.real(torch.trace(state))) - 1.0)
            for state in final_states
        ),
        "outcome_frequencies": outcome_frequencies.tolist(),
    }
    return row, convergence_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    metric_names = [
        "ensemble_frobenius_error_to_deterministic",
        "ensemble_trace_distance_to_deterministic",
        "deterministic_fidelity_to_target",
        "mean_trajectory_fidelity_to_target",
        "deterministic_trace_distance_to_target",
        "mean_trajectory_trace_distance_to_target",
        "mean_trajectory_diversity_squared_frobenius",
        "mean_branch_entropy_nats",
    ]
    result = {
        "run_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "trajectory_count_per_run": rows[0]["trajectory_count"],
        "metrics": {},
        "physicality": {
            "minimum_trajectory_eigenvalue": min(
                row["minimum_trajectory_eigenvalue"] for row in rows
            ),
            "maximum_trajectory_trace_error": max(
                row["maximum_trajectory_trace_error"] for row in rows
            ),
            "maximum_deterministic_replay_frobenius_error": max(
                row["deterministic_replay_frobenius_error"] for row in rows
            ),
        },
    }
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        result["metrics"][name] = {
            "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return result


def plot_convergence(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    colors = ["#2F5D8A", "#C17C20", "#6B7D3E", "#A14B74", "#52616B"]
    for seed, color in zip(sorted({row["seed"] for row in rows}), colors, strict=True):
        selected = sorted(
            [row for row in rows if row["seed"] == seed],
            key=lambda row: row["trajectory_count"],
        )
        counts = [row["trajectory_count"] for row in selected]
        axes[0].loglog(
            counts,
            [row["frobenius_error_to_deterministic"] for row in selected],
            marker="o",
            label=f"seed {seed}",
            color=color,
        )
        axes[1].loglog(
            counts,
            [row["trace_distance_to_deterministic"] for row in selected],
            marker="o",
            label=f"seed {seed}",
            color=color,
        )
    axes[0].set_title("Squared Frobenius error")
    axes[1].set_title("Trace distance")
    for axis in axes:
        axis.set_xlabel("sampled trajectories")
        axis.set_ylabel("empirical average vs deterministic channel")
        axis.grid(True, which="both", alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Quantum-instrument ensemble convergence")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "j25_fidelity_study" / "runs",
    )
    parser.add_argument("--trajectories", type=int, default=256)
    parser.add_argument("--study-seed", type=int, default=20261006)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "instrument_study",
    )
    args = parser.parse_args()
    if args.trajectories < 2:
        raise ValueError("at least two trajectories are required")
    run_directories = sorted(args.runs_root.resolve().glob("seed_*"))
    if not run_directories:
        raise FileNotFoundError(f"no trained runs found under {args.runs_root}")
    output_root = args.outdir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    convergence_rows = []
    for run_directory in run_directories:
        row, convergence = analyze_run(
            run_directory,
            n_trajectories=args.trajectories,
            study_seed=args.study_seed,
        )
        rows.append(row)
        convergence_rows.extend(convergence)
    rows.sort(key=lambda row: row["seed"])
    convergence_rows.sort(
        key=lambda row: (row["seed"], row["trajectory_count"])
    )

    csv_rows = [
        {
            **row,
            "outcome_frequencies": json.dumps(row["outcome_frequencies"]),
        }
        for row in rows
    ]
    write_csv(output_root / "runs.csv", csv_rows)
    write_csv(output_root / "ensemble_convergence.csv", convergence_rows)
    summary = aggregate(rows)
    summary["configuration"] = {
        "runs_root": str(args.runs_root.resolve()),
        "trajectory_count_per_run": args.trajectories,
        "study_seed": args.study_seed,
        "instrument_basis": "computational basis of the collision ancilla",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_convergence(
        convergence_rows,
        output_root / "ensemble_convergence.png",
    )
    print(f"instrument study written to {output_root}")


if __name__ == "__main__":
    main()
