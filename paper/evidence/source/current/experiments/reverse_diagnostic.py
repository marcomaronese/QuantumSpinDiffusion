#!/usr/bin/env python3
"""Paired fixed-architecture j=2.5 diagnostic, with audited saved states.

The manifest is frozen before training. Resume requires exact config and source
hash agreement. Each run is a separate process to isolate random generators.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import spin_diffusion_toy as toy
from spin_multipoles import irreducible_spherical_tensors
from j25_fidelity_study import analyze_run, mode_vector
from validation_study import significant_q_modes, independently_recompute_state_metrics


def match_each_mode(target_modes, generated_modes, tolerance=0.35):
    """One-to-one angular assignment; unmatched target modes remain explicit."""
    assignments = {}
    if target_modes and generated_modes:
        costs = np.array([[np.arccos(np.clip(mode_vector(t) @ mode_vector(g), -1, 1))
                           for g in generated_modes] for t in target_modes])
        ti, gi = linear_sum_assignment(costs)
        assignments = {int(t): (int(g), float(costs[t, g])) for t, g in zip(ti, gi)}
    rows = []
    for index, target in enumerate(target_modes):
        match = assignments.get(index)
        recovered = match is not None and match[1] <= tolerance
        rows.append({"target_index": index, "target": target,
                     "generated_index": None if match is None else match[0],
                     "angle_radians": None if match is None else match[1],
                     "recovered": recovered,
                     "recovered_relative_height": (generated_modes[match[0]]["relative_height"]
                                                   if recovered else None)})
    return rows


def rank_diagnostics(target, generated, tensors):
    """Coefficient errors include phase/orientation, unlike power ratios alone."""
    rows = {}
    for ell in sorted({ell for ell, _ in tensors}):
        basis = [tensor.numpy() for (rank, _), tensor in tensors.items() if rank == ell]
        a = np.array([np.trace(target @ tensor.conj().T) for tensor in basis])
        b = np.array([np.trace(generated @ tensor.conj().T) for tensor in basis])
        power = float(np.vdot(a, a).real)
        predicted_power = float(np.vdot(b, b).real)
        error = float(np.vdot(b-a, b-a).real)
        rows[str(ell)] = {"squared_coefficient_error": error, "target_power": power,
                          "generated_power": predicted_power,
                          "relative_coefficient_error": float(np.sqrt(error / power)) if power > 1e-24 else None,
                          "power_ratio": predicted_power / power if power > 1e-24 else None}
    return rows


def audit_run(directory):
    row = analyze_run(directory, 0.35)
    config = json.loads((directory / "config.json").read_text())
    metrics = json.loads((directory / "validation_metrics.json").read_text())
    target = np.load(directory / "rho_data.npy")
    generated = np.load(directory / "rho_generated.npy")
    jx, jy, _, _ = toy.spin_operators(2.5)
    tensors = irreducible_spherical_tensors(2.5, jx, jy)
    ranks = rank_diagnostics(target, generated, tensors)
    np.testing.assert_allclose(sum(x["squared_coefficient_error"] for x in ranks.values()),
                               row["frobenius_error"], rtol=1e-10, atol=1e-12)
    th, ph, tq, _ = toy.husimi_q_grid(torch.tensor(target), 2.5)
    _, _, gq, _ = toy.husimi_q_grid(torch.tensor(generated), 2.5)
    target_modes = significant_q_modes(th, ph, tq)
    generated_modes = significant_q_modes(th, ph, gq)
    matching = match_each_mode(target_modes, generated_modes)
    weak = matching[-1] if len(matching) >= 2 else None
    row.update({"starting_state": config["starting_state"], "supervision": config["supervision"],
                "epochs": config["epochs"], "objective": config["objective"],
                "hybrid_lambda": config["hybrid_lambda"], "ranks": ranks,
                "target_modes": target_modes, "generated_modes": generated_modes,
                "per_target_mode": matching,
                "weaker_target_relative_height": None if weak is None else weak["target"]["relative_height"],
                "weaker_recovered_relative_height": None if weak is None else weak["recovered_relative_height"],
                "training_seconds": metrics["reverse"]["training_seconds"],
                "generation_seconds": metrics["reverse"]["generation_seconds"],
                "parameter_count": metrics["architecture"]["trainable_parameter_count"],
                "mixed_prior_evaluation": independently_recompute_state_metrics(
                    directory / "rho_data.npy", directory / "rho_generated_mixed.npy", 2.5),
                "run_directory": str(directory)})
    # Recheck physicality from the saved trajectories, independently of saved metrics.
    for filename in ("forward_states.npy", "reverse_states.npy", "rho_generated_mixed.npy"):
        states = np.load(directory / filename).reshape(-1, 6, 6)
        for state in states:
            assert abs(np.trace(state)-1) < 1e-10
            assert np.linalg.norm(state-state.conj().T) < 1e-10
            assert np.linalg.eigvalsh(state).min() >= -1e-10
    (directory / "diagnostics.json").write_text(json.dumps(row, indent=2, allow_nan=False))
    return row


def source_digest():
    paths = source_paths()
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def source_paths():
    return sorted(ROOT.glob("spin_*.py")) + [Path(__file__), ROOT / "experiments/j25_fidelity_study.py",
                                             ROOT / "experiments/validation_study.py"]


def run_job(output, job):
    label = f'{job["starting_state"]}_{job["supervision"]}_e{job["epochs"]}_{job["objective"]}_l{job["hybrid_lambda"]}'
    directory = output / "runs" / label / f'seed_{job["seed"]}'
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "validation_metrics.json").exists():
        command = [sys.executable, str(ROOT / "spin_diffusion_toy.py"), "--j", "2.5", "--outdir", str(directory)]
        for key, value in job.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MPLCONFIGDIR=str(output / ".matplotlib"))
        with (directory / "run.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    config = json.loads((directory / "config.json").read_text())
    if any(config[key] != value for key, value in job.items()):
        raise ValueError(f"stale run configuration: {directory}")
    return audit_run(directory)


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "ci95": None}
    half = float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95": [mean-half, mean+half]}


def summarize(rows):
    groups = {}
    for row in rows:
        key = f'{row["starting_state"]}_{row["supervision"]}_e{row["epochs"]}_{row["objective"]}_l{row["hybrid_lambda"]}'
        groups.setdefault(key, []).append(row)
    metrics = ("infidelity", "trace_distance", "q_js_divergence", "training_seconds")
    summary = {key: {"seeds": [r["seed"] for r in group],
                     "both_modes_recovered": sum(r["target_mode_count"] == 2 and r["all_target_modes_recovered"] for r in group),
                     **{metric: mean_interval([r[metric] for r in group]) for metric in metrics}}
               for key, group in groups.items()}
    for key, group in groups.items():
        if all("classical_infidelity" in row for row in group):
            summary[key]["classical"] = {
                "both_modes_recovered": sum(r["target_mode_count"] == 2 and
                                            r["classical_all_target_modes_recovered"] for r in group),
                **{metric: mean_interval([r["classical_" + metric] for r in group])
                   for metric in ("infidelity", "trace_distance", "q_js_divergence")},
            }
    # Only one-factor contrasts, always paired by seed.
    comparisons = {}
    factors = ("starting_state", "supervision", "epochs", "hybrid_lambda")
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            ga, gb = groups[a], groups[b]
            if sum(ga[0][f] != gb[0][f] for f in factors) != 1:
                continue
            da, db = {r["seed"]: r for r in ga}, {r["seed"]: r for r in gb}
            seeds = sorted(da.keys() & db.keys())
            if not seeds:
                continue
            comparisons[f"{b} minus {a}"] = {metric: mean_interval([db[s][metric]-da[s][metric] for s in seeds])
                                                for metric in metrics}
    return {"groups": summary, "paired_differences": comparisons}


def execute(output, jobs, protocol, workers=2):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": source_digest(), "jobs": jobs, "protocol": protocol}
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Manifest differs; use a fresh output directory to avoid mixing experiments")
    path.write_text(json.dumps(manifest, indent=2))
    with ZipFile(output / "source_snapshot.zip", "w") as archive:
        for source in source_paths():
            archive.write(source, source.relative_to(ROOT))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_job, output, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f'Completed {len(rows)}/{len(jobs)}: {row["starting_state"]}/{row["supervision"]} '
                  f'epochs={row["epochs"]} seed={row["seed"]} modes={row["generated_mode_count"]}', flush=True)
    rows.sort(key=lambda r: (r["epochs"], r["starting_state"], r["supervision"], r["hybrid_lambda"], r["seed"]))
    (output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output / "summary.json").write_text(json.dumps(summarize(rows), indent=2, allow_nan=False))
    toy.write_classical_comparison([r["run_directory"] for r in rows],
                                   output / "classical_comparison.json")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="3,5,7,11,13")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--larger-epochs", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--outdir", type=Path, default=ROOT / "diagnostic_study")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or not 0 < args.epochs < args.larger_epochs or args.workers < 1:
        parser.error("require unique seeds, 0 < epochs < larger-epochs, and positive workers")
    jobs = [dict(seed=seed, epochs=epochs, starting_state=start, supervision=supervision,
                 objective="fidelity", hybrid_lambda=0.0)
            for epochs in (args.epochs, args.larger_epochs)
            for start in ("mixed", "forward") for supervision in ("path", "final") for seed in seeds]
    execute(args.outdir, jobs, {"j": 2.5, "architecture": "unchanged: 3 layers, 1 ancilla, full, independent",
             "mode_grid": [80, 160], "relative_peak_threshold": .5, "minimum_peak_separation_radians": .35,
             "matching_tolerance_radians": .35, "primary_outcome": "both encoded modes recovered",
             "interpretation": "Finite-budget failures cannot prove insufficient expressivity."}, args.workers)


if __name__ == "__main__":
    main()
