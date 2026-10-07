#!/usr/bin/env python3
"""Five-seed fidelity study at the first mode-resolving spin, j=2.5."""

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
import torch
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from spin_quantum_diffusion.quantum import single_spin as toy  # noqa: E402
from spin_quantum_diffusion.classical.benchmark import (  # noqa: E402
    require_classical_benchmark,
    write_classical_comparison,
)
from tasks.validation_study import (  # noqa: E402
    independently_recompute_state_metrics,
    significant_q_modes,
)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_training(
    output_root: Path,
    seed: int,
    epochs: int,
    force: bool,
) -> Path:
    run_directory = output_root / "runs" / f"seed_{seed}"
    metrics_path = run_directory / "validation_metrics.json"
    if metrics_path.exists() and not force:
        require_classical_benchmark(run_directory)
        return run_directory
    run_directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "spin_quantum_diffusion.quantum.single_spin",
        "--j",
        "2.5",
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--objective",
        "fidelity",
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
            f"training failed for seed {seed}; see {run_directory / 'run.log'}"
        )
    return run_directory


def mode_vector(mode: dict[str, float]) -> np.ndarray:
    theta = mode["theta"]
    phi = mode["phi"]
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def match_modes(
    target_modes: list[dict[str, float]],
    generated_modes: list[dict[str, float]],
    tolerance: float,
) -> dict[str, float | int | bool | None]:
    if not target_modes or not generated_modes:
        return {
            "matched_mode_count": 0,
            "recovered_target_mode_count": 0,
            "mean_matched_angle_radians": None,
            "maximum_matched_angle_radians": None,
            "all_target_modes_recovered": False,
        }
    target_vectors = [mode_vector(mode) for mode in target_modes]
    generated_vectors = [mode_vector(mode) for mode in generated_modes]
    costs = np.array(
        [
            [
                math.acos(float(np.clip(target @ generated, -1.0, 1.0)))
                for generated in generated_vectors
            ]
            for target in target_vectors
        ]
    )
    target_indices, generated_indices = linear_sum_assignment(costs)
    matched = costs[target_indices, generated_indices]
    recovered = int(np.sum(matched <= tolerance))
    return {
        "matched_mode_count": int(len(matched)),
        "recovered_target_mode_count": recovered,
        "mean_matched_angle_radians": float(np.mean(matched)),
        "maximum_matched_angle_radians": float(np.max(matched)),
        "all_target_modes_recovered": (
            recovered == len(target_modes)
            and len(generated_modes) >= len(target_modes)
        ),
    }


def analyze_run(run_directory: Path, mode_tolerance: float) -> dict:
    metrics = json.loads(
        (run_directory / "validation_metrics.json").read_text(encoding="utf-8")
    )
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    independent = independently_recompute_state_metrics(
        run_directory / "rho_data.npy",
        run_directory / "rho_generated.npy",
        j=2.5,
    )
    reverse = metrics["reverse"]
    declared = {
        "frobenius_error": reverse["final_frobenius_state_error"],
        "trace_distance": reverse["final_trace_distance"],
        "infidelity": 1.0 - reverse["final_quantum_fidelity"],
        "q_js_divergence": reverse["grid_jensen_shannon_divergence"],
        "q_correlation": reverse["q_grid_correlation"],
    }
    for name, value in declared.items():
        if not math.isclose(
            float(value),
            independent[name],
            rel_tol=2e-9,
            abs_tol=2e-11,
        ):
            raise AssertionError(
                f"independent {name} check failed for seed {seed}: "
                f"{value} vs {independent[name]}"
            )

    target = torch.tensor(np.load(run_directory / "rho_data.npy"), dtype=toy.CDTYPE)
    generated = torch.tensor(
        np.load(run_directory / "rho_generated.npy"), dtype=toy.CDTYPE
    )
    thetas, phis, target_q, _ = toy.husimi_q_grid(target, 2.5)
    _, _, generated_q, _ = toy.husimi_q_grid(generated, 2.5)
    target_modes = significant_q_modes(thetas, phis, target_q)
    generated_modes = significant_q_modes(thetas, phis, generated_q)
    matching = match_modes(target_modes, generated_modes, mode_tolerance)

    classical = {}
    if "classical" in metrics:
        classical_path = run_directory / "classical" / "rho_generated.npy"
        classical_state = torch.tensor(np.load(classical_path), dtype=toy.CDTYPE)
        _, _, classical_q, _ = toy.husimi_q_grid(classical_state, 2.5)
        classical_modes = significant_q_modes(thetas, phis, classical_q)
        classical = {
            **independently_recompute_state_metrics(
                run_directory / "rho_data.npy", classical_path, j=2.5),
            "generated_mode_count": len(classical_modes),
            **match_modes(target_modes, classical_modes, mode_tolerance),
        }

    return {
        "seed": seed,
        **independent,
        "target_mode_count": len(target_modes),
        "generated_mode_count": len(generated_modes),
        **matching,
        **{f"classical_{key}": value for key, value in classical.items()},
        "initial_training_loss": reverse["initial_training_loss"],
        "final_training_loss": reverse["final_training_loss"],
        "max_trace_error": reverse["max_trace_error"],
        "minimum_eigenvalue": reverse["minimum_eigenvalue"],
        "max_trace_preservation_error": reverse["max_trace_preservation_error"],
        "minimum_choi_eigenvalue": reverse["minimum_choi_eigenvalue"],
        "max_multipole_decay_error": metrics["forward"][
            "max_multipole_decay_error"
        ],
    }


def aggregate(rows: list[dict]) -> dict:
    metric_names = [
        "frobenius_error",
        "trace_distance",
        "infidelity",
        "q_js_divergence",
        "q_correlation",
        "mean_matched_angle_radians",
        "maximum_matched_angle_radians",
    ]
    if all("classical_infidelity" in row for row in rows):
        metric_names += ["classical_" + name for name in metric_names.copy()]
    summary: dict[str, object] = {
        "run_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "all_target_modes_recovered_runs": sum(
            bool(row["all_target_modes_recovered"]) for row in rows
        ),
        "metrics": {},
        "physicality": {
            "maximum_trace_error": max(row["max_trace_error"] for row in rows),
            "minimum_state_eigenvalue": min(
                row["minimum_eigenvalue"] for row in rows
            ),
            "maximum_kraus_completeness_error": max(
                row["max_trace_preservation_error"] for row in rows
            ),
            "minimum_choi_eigenvalue": min(
                row["minimum_choi_eigenvalue"] for row in rows
            ),
            "maximum_multipole_decay_error": max(
                row["max_multipole_decay_error"] for row in rows
            ),
        },
    }
    for name in metric_names:
        values = [float(row[name]) for row in rows if row[name] is not None]
        summary["metrics"][name] = {
            "mean": statistics.mean(values) if values else None,
            "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    if all("classical_all_target_modes_recovered" in row for row in rows):
        summary["classical_all_target_modes_recovered_runs"] = sum(
            bool(row["classical_all_target_modes_recovered"]) for row in rows)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["seed"])
    seeds = [str(row["seed"]) for row in rows]
    panels = [
        ("infidelity", "Infidelity", False),
        ("q_js_divergence", "Q-grid JS divergence", False),
        ("maximum_matched_angle_radians", "Worst matched-mode angle [rad]", True),
        ("generated_mode_count", "Significant generated modes", True),
    ]
    colors = ["#2F5D8A", "#C17C20", "#6B7D3E", "#A14B74"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for axis, (name, title, show_threshold), color in zip(
        axes.reshape(-1), panels, colors, strict=True
    ):
        values = np.asarray([row[name] for row in rows], dtype=float)
        axis.scatter(seeds, values, s=45, color=color, zorder=3)
        axis.axhline(
            float(np.mean(values)),
            color="#30343B",
            linestyle="--",
            linewidth=1.2,
            label="five-seed mean",
        )
        if show_threshold and name == "maximum_matched_angle_radians":
            axis.axhline(
                0.35,
                color="#8A8F98",
                linestyle=":",
                linewidth=1.2,
                label="recovery tolerance",
            )
        if name == "generated_mode_count":
            axis.axhline(
                2.0,
                color="#8A8F98",
                linestyle=":",
                linewidth=1.2,
                label="target count",
            )
            axis.set_ylim(0.5, max(2.5, float(values.max()) + 0.5))
        axis.set_title(title)
        axis.set_xlabel("seed")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("j=2.5 fidelity objective: five-seed validation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", default="3,5,7,11,13")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--mode-tolerance", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "experiments" / "j25_fidelity_study",
    )
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    output_root = args.outdir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    directories: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        jobs = [
            executor.submit(
                run_training,
                output_root,
                seed,
                args.epochs,
                args.force,
            )
            for seed in seeds
        ]
        for completed in as_completed(jobs):
            directories.append(completed.result())

    rows = sorted(
        [analyze_run(directory, args.mode_tolerance) for directory in directories],
        key=lambda row: row["seed"],
    )
    summary = aggregate(rows)
    write_classical_comparison(directories, output_root / "classical_comparison.json")
    summary["configuration"] = {
        "j": 2.5,
        "objective": "fidelity",
        "epochs": args.epochs,
        "mode_relative_height_threshold": 0.5,
        "mode_minimum_separation_radians": 0.35,
        "mode_recovery_tolerance_radians": args.mode_tolerance,
    }
    write_csv(output_root / "runs.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_results(rows, output_root / "five_seed_summary.png")
    print(f"j=2.5 fidelity study written to {output_root}")


if __name__ == "__main__":
    main()
