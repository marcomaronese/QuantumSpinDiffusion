"""Frozen unconditional Euclidean benchmarks through the spin-coherent model."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from spin_quantum_diffusion.classical.diffusion import (
    ClassicalDiffusionConfig, save_classical_model, train_classical_diffusion,
)
from spin_quantum_diffusion.classical.synthetic import TARGETS, sample_discrepancies, sample_target
from spin_quantum_diffusion.quantum.coherent_readout import coherent_kets, coherent_povm
from spin_quantum_diffusion.quantum.euclidean import EuclideanSpinChart
from spin_quantum_diffusion.quantum.losses import trace_distance
from spin_quantum_diffusion.quantum.multipoles import (
    irreducible_spherical_tensors, max_multipole_decay_error, multipole_trajectory,
)
from spin_quantum_diffusion.quantum.reverse import (
    build_reverse_generators, collision_kraus_operators, reverse_collision_unitary,
)
from spin_quantum_diffusion.quantum.single_spin import (
    assert_physical_density, build_forward_trajectory, generate_trajectory,
    spin_operators, train_reverse,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "notes/protocols/EUCLIDEAN_BENCHMARK_V1.md"
TWO_J, STEPS, LAYERS, UPDATES = 5, 6, 3, 512
GRID, FINE_GRID = (24, 48), (48, 96)
TOLERANCE = 1e-10
CONDITIONS = ("model", "encoded_train", "encoded_test", "mixed", "classical_common",
              "classical_direct", "bootstrap", "clipped_bootstrap", "real_reference")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def encoded_density(directions):
    kets = coherent_kets(torch.as_tensor(directions, dtype=torch.float64), TWO_J)
    return kets.T @ kets.conj() / len(kets)


def angles(directions):
    return np.column_stack((np.arccos(np.clip(directions[:, 2], -1, 1)),
                            np.arctan2(directions[:, 1], directions[:, 0]) % (2*np.pi)))


def directions_from_angles(values):
    theta, phi = values.T
    return np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))


def probabilities(measurement, density):
    values = measurement.probabilities(density).detach().numpy()
    if values.min() < -TOLERANCE or abs(values.sum() - 1) > TOLERANCE:
        raise ValueError("invalid Born probabilities; no normalization repair allowed")
    # Purely numerical negatives below machine precision are not expected here.
    if values.min() < 0:
        raise ValueError("negative Born probability")
    return values


def histogram(values, dimensions, weights=None):
    bins = 24 if dimensions == 1 else 12
    edges = np.r_[-np.inf, np.linspace(-3, 3, bins+1), np.inf]
    hist = np.histogramdd(values, bins=[edges]*dimensions, weights=weights)[0]
    return hist / len(values) if weights is None else hist


def snapshot(output, config):
    manifest = json.loads((ROOT / "materials/paper/evidence/manifest.json").read_text())
    for entry in manifest["files"]:
        if digest(ROOT / "materials/paper" / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"frozen evidence mismatch: {entry['path']}")
    paths = sorted((ROOT / "src").rglob("*.py")) + [
        Path(__file__), ROOT / "tasks/validate_synthetic_benchmark.py", PROTOCOL,
        ROOT / "pyproject.toml", ROOT / "tests/test_euclidean_benchmark.py",
    ]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    write_json(output / "metadata.json", {
        "protocol": str(PROTOCOL.relative_to(ROOT)), "config": config,
        "source_hashes": hashes, "source_fingerprint": fingerprint,
        "frozen_evidence_hashes_verified": len(manifest["files"]),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "torch": torch.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    return fingerprint


def prepare(output, config):
    rows = []
    for name in config["targets"]:
        target_index = list(TARGETS).index(name)
        for seed in config["seeds"]:
            base = seed + 100000*target_index
            directory = output / "runs" / name / str(seed)
            directory.mkdir(parents=True)
            data = {split: sample_target(name, config["n_train"] if split == "train" else config["n_eval"],
                                         base + offset)
                    for split, offset in (("train", 0), ("validation", 10000),
                                          ("test", 20000), ("reference", 30000))}
            chart = EuclideanSpinChart.fit(data["train"])
            np.savez_compressed(directory / "data.npz", **data)
            write_json(directory / "chart.json", chart.to_dict())
            rows.append({"target": name, "seed": seed, "base": base,
                         "directory": str(directory.relative_to(output)),
                         "target_specification": TARGETS[name],
                         "data_sha256": digest(directory / "data.npz"),
                         "chart_sha256": digest(directory / "chart.json")})
    write_json(output / "prepared.json", rows)
    return rows


def fit(directory, row, config, generators, identity, operators):
    # The fitting phase accesses training observations only. All test scoring
    # is performed in a second phase, after every checkpoint has been saved.
    training = np.load(directory / "data.npz")["train"]
    description = json.loads((directory / "chart.json").read_text())
    chart = EuclideanSpinChart(description["location"], description["scale"], description["radius"])
    directions, _ = chart.encode(training)
    target = encoded_density(directions)
    forward = build_forward_trajectory(target, *operators, D=1., dt=.35, T=STEPS)
    for index, state in enumerate(forward):
        assert_physical_density(state, f"forward {index}")
    tensors = irreducible_spherical_tensors(TWO_J/2, *operators[:2])
    decay_error = max_multipole_decay_error(
        multipole_trajectory(forward, tensors), np.arange(STEPS+1)*.35, 1.,
    )
    if decay_error > TOLERANCE:
        raise ValueError("forward multipole decay failed")
    rng = torch.Generator().manual_seed(row["base"] + 40000)
    initial = .03 * torch.randn((STEPS, LAYERS, len(generators)), generator=rng, dtype=torch.float64)
    started = time.perf_counter()
    with (directory / "training.log").open("w") as log, redirect_stdout(log):
        parameters, history = train_reverse(
            forward, generators, LAYERS, config["updates"], .04, row["base"] + 40000,
            objective="frobenius", supervision="final", initial_parameters=initial,
        )
    quantum_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reverse = generate_trajectory(identity / len(identity), parameters, generators)
    generation_seconds = time.perf_counter() - started
    states = {"forward": torch.stack(forward).numpy(), "reverse": torch.stack(reverse).numpy(),
              "parameters": parameters.numpy(), "initial_parameters": initial.numpy(),
              "generators": torch.stack(generators).numpy(), "loss_history": np.asarray(history)}
    physicality = [assert_physical_density(state, f"reverse {index}") for index, state in enumerate(reverse)]
    kraus_residual = 0.
    for controls in parameters:
        unitary = reverse_collision_unitary(controls, generators)
        kraus = collision_kraus_operators(unitary, len(identity))
        residual = float(torch.linalg.norm(sum(k.mH @ k for k in kraus) - identity))
        kraus_residual = max(kraus_residual, residual)
    if kraus_residual > TOLERANCE:
        raise ValueError("reverse channel is not trace preserving")
    classical_config = ClassicalDiffusionConfig(
        steps=64, epochs=config["classical_epochs"], batch_size=config["classical_batch"],
        hidden_width=64, learning_rate=.001,
    )
    started = time.perf_counter()
    classical, losses = train_classical_diffusion(angles(directions), classical_config, row["base"] + 50000)
    classical_seconds = time.perf_counter() - started
    save_classical_model(classical, directory / "classical.pt")
    started = time.perf_counter()
    classical_directions = directions_from_angles(classical.sample(config["n_eval"], row["base"] + 60005))
    classical_generation_seconds = time.perf_counter() - started
    states["classical_directions"] = classical_directions
    states["classical_density"] = encoded_density(classical_directions).numpy()
    states["classical_loss_history"] = np.asarray(losses)
    np.savez_compressed(directory / "fit.npz", **states)
    write_json(directory / "fit.json", {
        "quantum_training_seconds": quantum_seconds, "quantum_generation_seconds": generation_seconds,
        "classical_training_seconds": classical_seconds,
        "classical_generation_seconds": classical_generation_seconds,
        "quantum_parameters": parameters.numel(), "quantum_updates": config["updates"],
        "classical_parameters": sum(p.numel() for p in classical.parameters()),
        "classical_updates": classical_config.epochs * int(np.ceil(len(training) / classical_config.batch_size)),
        "classical_config": asdict(classical_config), "physicality": physicality,
        "kraus_completeness_residual": kraus_residual,
        "forward_multipole_decay_error": decay_error,
        "fit_sha256": digest(directory / "fit.npz"),
        "classical_checkpoint_sha256": digest(directory / "classical.pt"),
        "test_accessed_during_fit": False,
    })


def evaluate(directory, row, config, measurement, fine):
    data = dict(np.load(directory / "data.npz"))
    fitted = dict(np.load(directory / "fit.npz"))
    spec = json.loads((directory / "chart.json").read_text())
    chart = EuclideanSpinChart(spec["location"], spec["scale"], spec["radius"])
    states = {split: encoded_density(chart.encode(data[split])[0])
              for split in ("train", "validation", "test")}
    densities = {"model": torch.from_numpy(fitted["reverse"][-1]), "encoded_train": states["train"],
                 "encoded_test": states["test"], "mixed": torch.eye(TWO_J+1, dtype=torch.complex128)/(TWO_J+1),
                 "classical_common": torch.from_numpy(fitted["classical_density"])}
    decoded = chart.decode(measurement.directions.numpy())
    fine_decoded = chart.decode(fine.directions.numpy())
    standardized_grid = chart.standardize(decoded)
    samples, born, outcomes, metrics, refinements = {}, {}, {}, {}, {}
    dimensions = data["train"].shape[1]
    reference = chart.standardize(data["test"])
    raw_histogram = histogram(reference, dimensions)
    for index, name in enumerate(CONDITIONS):
        rng = np.random.default_rng(row["base"] + 60000 + index)
        if name in densities:
            assert_physical_density(densities[name], name)
            probability = probabilities(measurement, densities[name])
            outcomes[name] = rng.choice(len(probability), config["n_eval"], p=probability)
            samples[name] = decoded[outcomes[name]]
            born[name] = probability
            fine_probability = probabilities(fine, densities[name])
            h = histogram(standardized_grid, dimensions, probability)
            fine_h = histogram(chart.standardize(fine_decoded), dimensions, fine_probability)
            refinements[name] = {
                "binned_tv": float(.5*np.abs(h-fine_h).sum()),
                "mean_max_abs": float(np.max(np.abs(probability @ decoded - fine_probability @ fine_decoded))),
                "second_moment_max_abs": float(np.max(np.abs(
                    decoded.T @ (probability[:, None]*decoded)
                    - fine_decoded.T @ (fine_probability[:, None]*fine_decoded)))),
            }
            raw_bin_tv = float(.5*np.abs(raw_histogram-h).sum())
        elif name == "classical_direct":
            samples[name] = chart.decode(fitted["classical_directions"])
            raw_bin_tv = float(.5*np.abs(raw_histogram-histogram(chart.standardize(samples[name]), dimensions)).sum())
        elif name in ("bootstrap", "clipped_bootstrap"):
            indices = rng.integers(len(data["train"]), size=config["n_eval"])
            values = data["train"][indices]
            samples[name] = values if name == "bootstrap" else chart.decode(chart.encode(values)[0])
            raw_bin_tv = float(.5*np.abs(raw_histogram-histogram(chart.standardize(samples[name]), dimensions)).sum())
        else:
            samples[name] = data["reference"]
            raw_bin_tv = float(.5*np.abs(raw_histogram-histogram(chart.standardize(samples[name]), dimensions)).sum())
        metrics[name] = {**sample_discrepancies(reference, chart.standardize(samples[name]), config["energy_count"]),
                         "raw_binned_tv": raw_bin_tv,
                         "decoded_mean": samples[name].mean(axis=0).tolist(),
                         "decoded_covariance": np.atleast_2d(np.cov(samples[name], rowvar=False)).tolist()}
    state_metrics = {}
    for split in ("validation", "test"):
        target_probability = probabilities(measurement, states[split])
        for name in ("model", "encoded_train", "mixed", "classical_common"):
            state_metrics[f"{name}_{split}"] = {
                "trace_distance": float(trace_distance(densities[name], states[split])),
                "born_tv": float(.5*np.abs(born[name]-target_probability).sum()),
            }
    report = {
        "target": row["target"], "seed": row["seed"], "metrics": metrics,
        "state_metrics": state_metrics,
        "clipped_fractions": {name: float(chart.encode(values)[1].mean()) for name, values in data.items()},
        "grid_refinement": refinements,
        "prior_to_terminal_trace_distance": float(trace_distance(densities["mixed"], torch.from_numpy(fitted["forward"][-1]))),
        "train_model_trace_distance": float(trace_distance(densities["model"], states["train"])),
        "resources": json.loads((directory / "fit.json").read_text()),
        "heldout_test_consumed": True,
    }
    np.savez_compressed(directory / "evaluation.npz", **{f"samples_{key}": value for key, value in samples.items()},
                        **{f"born_{key}": value for key, value in born.items()},
                        **{f"outcomes_{key}": value for key, value in outcomes.items()},
                        rho_train=states["train"].numpy(), rho_validation=states["validation"].numpy(),
                        rho_test=states["test"].numpy())
    write_json(directory / "metrics.json", report)
    return report


def plots(output, rows, destination=None):
    destination = output if destination is None else destination
    for dimensions in (1, 2):
        selected = [row for row in rows if TARGETS[row["target"]]["dimensions"] == dimensions
                    and row["seed"] == rows[0]["seed"]]
        if not selected:
            continue
        if dimensions == 1:
            figure, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
            for axis, row in zip(axes.flat, selected):
                directory = output / row["directory"]
                data = np.load(directory / "data.npz")["test"][:, 0]
                samples = np.load(directory / "evaluation.npz")
                chart = json.loads((directory / "chart.json").read_text())
                low, high = chart["location"][0] + np.array([-3, 3])*chart["scale"][0]
                bins = np.linspace(low, high, 36)
                for values, color, label in ((data, "#202c39", "Raw held-out"),
                    (samples["samples_encoded_test"][:, 0], "#ce8b20", "Encoded oracle"),
                    (samples["samples_model"][:, 0], "#2166ac", "SpinDiffusion")):
                    counts, _ = np.histogram(values, bins)
                    axis.stairs(counts / len(values) / np.diff(bins), bins, color=color, label=label)
                omitted = np.mean((data < low) | (data > high))
                axis.set_title(f"{row['target']} | raw outside chart: {omitted:.1%}", fontsize=10)
                axis.set_xlabel("Original coordinate")
                axis.set_ylabel("Density (not renormalized)")
            axes[0, 0].legend(fontsize=8)
            for axis in list(axes.flat)[len(selected):]:
                axis.set_visible(False)
        else:
            figure, axes = plt.subplots(len(selected), 3, figsize=(11, 3*len(selected)), squeeze=False)
            for axes_row, row in zip(axes, selected):
                directory = output / row["directory"]
                data = np.load(directory / "data.npz")["test"]
                samples = np.load(directory / "evaluation.npz")
                spec = json.loads((directory / "chart.json").read_text())
                for axis, values, title, color in zip(axes_row,
                    (data, samples["samples_encoded_test"], samples["samples_model"]),
                    ("Raw held-out", "Encoded oracle", "SpinDiffusion"),
                    ("#202c39", "#ce8b20", "#2166ac"), strict=True):
                    axis.scatter(*values[:1000].T, s=3, alpha=.3, color=color, rasterized=True)
                    axis.set_xlim(spec["location"][0]-3*spec["scale"][0], spec["location"][0]+3*spec["scale"][0])
                    axis.set_ylim(spec["location"][1]-3*spec["scale"][1], spec["location"][1]+3*spec["scale"][1])
                    axis.set_title(f"{row['target']} | {title}", fontsize=9)
                    axis.set_xlabel("x1")
                    axis.set_ylabel("x2")
        figure.suptitle(f"Fixed j=5/2 | first declared seed {rows[0]['seed']} | bounded chart", fontsize=13, y=.995)
        figure.tight_layout(rect=(0, 0, 1, .97))
        figure.savefig(destination / f"distributions_{dimensions}d.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
    labels = list(dict.fromkeys(row["target"] for row in rows))
    figure, axis = plt.subplots(figsize=(13, 5))
    for offset, condition, color, label in ((-.3, "real_reference", "#707070", "Real vs real"),
        (-.1, "encoded_test", "#ce8b20", "Encoded oracle"),
        (.1, "model", "#2166ac", "SpinDiffusion"),
        (.3, "classical_common", "#26866b", "DDPM + common POVM")):
        groups = [[json.loads((output / row["directory"] / "metrics.json").read_text())["metrics"][condition]["sliced_w1"]
                   for row in rows if row["target"] == name] for name in labels]
        axis.bar(np.arange(len(labels))+offset, [np.mean(group) for group in groups], .19,
                 yerr=[np.std(group, ddof=1) if len(group)>1 else 0 for group in groups],
                 color=color, label=label, capsize=2)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_ylabel("W1 / sliced W1 (training-standardized units)\nMean +/- SD", fontsize=10)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(destination / "discrepancy_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.outdir.resolve()
    if not output.is_relative_to(ROOT / "outputs/experiments") or output.exists():
        raise ValueError("use a fresh directory under outputs/experiments")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    config = {"targets": ["gaussian", "correlated_gaussian"] if args.smoke else list(TARGETS),
              "seeds": [991] if args.smoke else [1031, 2081, 4093],
              "n_train": 64 if args.smoke else 2048, "n_eval": 96 if args.smoke else 2048,
              "updates": 2 if args.smoke else UPDATES, "classical_epochs": 2 if args.smoke else 64,
              "classical_batch": 64 if args.smoke else 256, "energy_count": 96 if args.smoke else 1024,
              "smoke": args.smoke, "two_j": TWO_J, "steps": STEPS, "layers": LAYERS,
              "grid": GRID, "fine_grid": FINE_GRID, "conditions": CONDITIONS}
    snapshot(output, config)
    rows = prepare(output, config)
    *operators, identity = spin_operators(TWO_J/2)
    generators = build_reverse_generators(*operators, identity)
    measurement, fine = coherent_povm(TWO_J, *GRID), coherent_povm(TWO_J, *FINE_GRID)
    for grid in (measurement, fine):
        residual = torch.linalg.norm(grid.frame.T @ grid.frame.conj() - identity)
        if residual > TOLERANCE:
            raise ValueError("incomplete POVM")
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        fit(output / row["directory"], row, config, generators, identity, operators)
        print(f"Fit {index}/{len(rows)}: {row['target']} seed {row['seed']} ({time.perf_counter()-started:.1f}s)", flush=True)
    results = []
    for row in rows:
        report = evaluate(output / row["directory"], row, config, measurement, fine)
        results.append(report)
        print(f"Evaluated {row['target']} seed {row['seed']}", flush=True)
    summary = {}
    for name in config["targets"]:
        group = [report for report in results if report["target"] == name]
        summary[name] = {condition: {metric: {
            "mean": float(np.mean([row["metrics"][condition][metric] for row in group])),
            "sd": float(np.std([row["metrics"][condition][metric] for row in group], ddof=1)) if len(group)>1 else 0.,
        } for metric in ("sliced_w1", "energy_v", "raw_binned_tv")} for condition in CONDITIONS}
    write_json(output / "summary.json", {"targets": summary, "fits": len(rows),
        "scope": "exploratory fixed-model benchmark; no architecture selection or evidence promotion",
        "elapsed_seconds": time.perf_counter()-started})
    plots(output, rows)
    files = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "manifest.json", files)
    print(f"Saved benchmark: {output}", flush=True)


if __name__ == "__main__":
    main()
