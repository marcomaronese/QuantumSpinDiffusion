"""Measure directional reconstruction across spin counts and common local j."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.linalg import expm
from scipy.stats import vonmises_fisher
import torch

from spin_quantum_diffusion.quantum.coherent_readout import coherent_povm
from spin_quantum_diffusion.quantum.config import ExperimentConfig, reproducibility_metadata
from spin_quantum_diffusion.quantum.losses import trace_distance
from spin_quantum_diffusion.quantum.reverse import build_reverse_generators
from spin_quantum_diffusion.quantum.single_spin import liouvillian_matrix, spin_operators
from spin_quantum_diffusion.quantum.spin_register import (
    encode_product_coherent,
    independent_spin_channel,
    product_born_probabilities,
    shared_ancilla_kraus,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "notes/protocols/SPIN_GRID_V1.md"
CENTERS_DEG = ((40, 15), (61, 50), (102, 153), (127, 271))
ROTATIONS_DEG = (0, 65, -110)
CONCENTRATIONS = (55, 85, 24, 40)
SHARED_WEIGHT = 0.55
SHARED_MODES = (0.36, 0.29, 0.22, 0.13)
STEPS, LAYERS = 3, 2
DIFFUSION_RATE, TIME_STEP = 1.0, 0.75
LEARNING_RATE, INIT_SCALE, PATH_WEIGHT = 0.055, 0.6, 0.15
READOUT_GRID, FINER_GRID = (32, 64), (48, 96)
TOLERANCE = 1e-10


def mode_centers():
    theta, phi = np.deg2rad(CENTERS_DEG).T
    azimuth = phi[None, :] + np.deg2rad(ROTATIONS_DEG)[:, None]
    return np.stack((np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth),
                     np.broadcast_to(np.cos(theta), azimuth.shape)), axis=-1)


def sample_directions(count, seed, centers):
    rng = np.random.default_rng(seed)
    shared = rng.random(count) < SHARED_WEIGHT
    first_two = rng.integers(len(CONCENTRATIONS), size=(count, 2))
    labels = np.column_stack((first_two, first_two.sum(axis=1) % len(CONCENTRATIONS)))
    labels[shared] = rng.choice(len(CONCENTRATIONS), size=(shared.sum(), 1), p=SHARED_MODES)
    directions = np.empty((count, len(ROTATIONS_DEG), 3))
    for spin, site in enumerate(centers):
        for mode, (center, kappa) in enumerate(zip(site, CONCENTRATIONS, strict=True)):
            selected = labels[:, spin] == mode
            if selected.any():
                directions[selected, spin] = vonmises_fisher.rvs(
                    center, kappa, size=selected.sum(), random_state=rng,
                )
    return directions, labels


def region_histogram(directions, centers):
    labels = np.einsum("nsc,skc->nsk", directions, centers).argmax(axis=-1)
    shape = (centers.shape[1],) * centers.shape[0]
    bins = np.ravel_multi_index(labels.T, shape)
    return np.bincount(bins, minlength=np.prod(shape)).reshape(shape) / len(labels)


def region_effects(two_j, centers, grid):
    povm = coherent_povm(two_j, *grid)
    effects = []
    for site in centers:
        labels = (povm.directions @ torch.from_numpy(site).T).argmax(dim=1)
        frames = [povm.frame[labels == mode] for mode in range(len(site))]
        effects.append(torch.stack([frame.T @ frame.conj() for frame in frames]))
    identity = torch.eye(two_j + 1, dtype=povm.frame.dtype)
    residual = max(float(torch.linalg.norm(site.sum(0) - identity)) for site in effects)
    if residual > TOLERANCE:
        raise RuntimeError(f"incomplete coherent POVM: {residual}")
    return effects, residual


def prefix(probabilities, spins):
    axes = tuple(range(spins, probabilities.ndim))
    return probabilities.sum(axis=axes) if axes else probabilities


def tv(left, right):
    return float(0.5 * np.abs(left - right).sum())


def connected_pair(p):
    pair = prefix(p, 2)
    return pair - np.outer(pair.sum(axis=1), pair.sum(axis=0))


def connected_triple(p):
    first, second, third = p.sum(axis=(1, 2)), p.sum(axis=(0, 2)), p.sum(axis=(0, 1))
    return (p - p.sum(axis=2)[:, :, None] * third
            - p.sum(axis=1)[:, None, :] * second[None, :, None]
            - p.sum(axis=0)[None, :, :] * first[:, None, None]
            + 2 * np.einsum("i,j,k->ijk", first, second, third))


def state_checks(states):
    residuals = {
        "trace_residual": max(float((state.trace() - 1).abs()) for state in states),
        "hermiticity_residual": max(float(torch.linalg.norm(state - state.mH)) for state in states),
        "min_eigenvalue": min(float(torch.linalg.eigvalsh(state).min()) for state in states),
    }
    if max(residuals["trace_residual"], residuals["hermiticity_residual"],
           -residuals["min_eigenvalue"]) > TOLERANCE:
        raise RuntimeError(f"state physicality failed: {residuals}")
    return residuals


def born(density, effects):
    p = product_born_probabilities(density, effects).numpy()
    if p.min() < -TOLERANCE or abs(p.sum() - 1) > TOLERANCE:
        raise RuntimeError("Born probabilities are negative or do not sum to one")
    return p


def fit(train_state, operators, identity, spins, seed, updates):
    local_channel = torch.from_numpy(expm(TIME_STEP * liouvillian_matrix(
        operators, DIFFUSION_RATE,
    ).numpy()))
    forward = [train_state]
    for _ in range(STEPS):
        forward.append(independent_spin_channel(forward[-1], local_channel, spins))
    size = len(train_state)
    register_identity = torch.eye(size, dtype=train_state.dtype)
    prior = register_identity / size
    generators = build_reverse_generators(*operators, identity, generator_set="no-twist")
    local_generators, ancilla_generators = generators[:-2], generators[-2:]
    width = len(local_generators)
    rng = torch.Generator().manual_seed(seed)
    local = torch.randn((STEPS, LAYERS, len(ROTATIONS_DEG), width),
                        dtype=torch.float64, generator=rng)
    ancilla = torch.randn((STEPS, LAYERS, len(ancilla_generators)),
                          dtype=torch.float64, generator=rng)
    angles = torch.nn.Parameter(INIT_SCALE * torch.cat((local[:, :, :spins].flatten(2), ancilla), dim=2))
    initial = angles.detach().clone()
    optimizer = torch.optim.Adam([angles], lr=LEARNING_RATE)
    purity = train_state.abs().square().sum()

    def generate():
        state = prior
        trajectory, kraus_steps = [prior], []
        for step in angles:
            kraus = shared_ancilla_kraus(step, local_generators, ancilla_generators, spins)
            state = (kraus @ state @ kraus.mH).sum(dim=0)
            trajectory.append(state)
            kraus_steps.append(kraus)
        return trajectory, kraus_steps

    started = time.perf_counter()
    losses = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        reverse, _ = generate()
        errors = [(prediction - target).abs().square().sum()
                  for prediction, target in zip(reverse[1:], reversed(forward[:-1]), strict=True)]
        loss = (errors[-1] + PATH_WEIGHT * torch.stack(errors[:-1]).mean()) / purity
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite reverse objective")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    training_seconds = time.perf_counter() - started
    with torch.no_grad():
        started = time.perf_counter()
        reverse, kraus_steps = generate()
        generation_seconds = time.perf_counter() - started
        kraus_residual = max(float(torch.linalg.norm((k.mH @ k).sum(0) - register_identity))
                             for k in kraus_steps)
    if kraus_residual > TOLERANCE:
        raise RuntimeError(f"Kraus completeness failed: {kraus_residual}")
    return forward, reverse, initial, angles.detach(), losses, {
        "parameters": angles.numel(), "updates": updates, "training_seconds": training_seconds,
        "generation_seconds": generation_seconds, "kraus_residual": kraus_residual,
        "forward_final_to_prior_trace": float(trace_distance(forward[-1], prior)),
    }


def evaluate_cell(directory, data, centers, raw_train, raw_validation, spins, two_j, seed, updates):
    train, validation = (torch.from_numpy(split[:, :spins]) for split in data)
    train_state = encode_product_coherent(train, two_j)
    target = encode_product_coherent(validation, two_j)
    factorized = encode_product_coherent(train[:, :1], two_j)
    for spin in range(1, spins):
        marginal = encode_product_coherent(train[:, spin:spin + 1], two_j)
        factorized = torch.kron(factorized.contiguous(), marginal.contiguous())
    *operators, identity = spin_operators(two_j / 2)
    forward, reverse, initial, angles, losses, costs = fit(
        train_state, operators, identity, spins, seed, updates,
    )
    generated = reverse[-1]
    states = {"encoded": target, "generated": generated,
              "factorized": factorized, "train": train_state}
    checks = state_checks([*forward, *reverse, target, factorized])
    effects, povm_residual = region_effects(two_j, centers[:spins], READOUT_GRID)
    finer, finer_residual = region_effects(two_j, centers[:spins], FINER_GRID)
    probabilities = {name: born(state, effects) for name, state in states.items()}
    fine_p = {name: born(state, finer) for name, state in states.items()}
    encoded, model = probabilities["encoded"], probabilities["generated"]
    metrics = {
        "raw_generated_tv": tv(raw_validation, model),
        "raw_encoded_tv": tv(raw_validation, encoded),
        "encoded_generated_tv": tv(encoded, model),
        "raw_factorized_tv": tv(raw_validation, probabilities["factorized"]),
        "raw_train_validation_tv": tv(raw_train, raw_validation),
        "encoded_train_validation_tv": tv(probabilities["train"], encoded),
        "state_trace_distance": float(trace_distance(generated, target)),
        "state_factorized_trace_distance": float(trace_distance(factorized, target)),
        "state_train_validation_trace_distance": float(trace_distance(train_state, target)),
        "first_spin_raw_generated_tv": tv(prefix(raw_validation, 1), prefix(model, 1)),
        "first_spin_raw_encoded_tv": tv(prefix(raw_validation, 1), prefix(encoded, 1)),
        "first_pair_connected_error": None,
        "first_pair_encoding_error": None,
        "first_pair_source_strength": None,
        "triple_connected_error": None,
        "triple_encoding_error": None,
        "triple_source_strength": None,
        "finer_grid_max_tv": max(tv(p, fine_p[name]) for name, p in probabilities.items()),
        "povm_residual": max(povm_residual, finer_residual),
    }
    if spins >= 2:
        raw_pair = connected_pair(raw_validation)
        metrics.update(first_pair_connected_error=tv(raw_pair, connected_pair(model)),
                       first_pair_encoding_error=tv(raw_pair, connected_pair(encoded)),
                       first_pair_source_strength=tv(raw_pair, 0))
    if spins == 3:
        raw_triple = connected_triple(raw_validation)
        metrics.update(triple_connected_error=tv(raw_triple, connected_triple(model)),
                       triple_encoding_error=tv(raw_triple, connected_triple(encoded)),
                       triple_source_strength=tv(raw_triple, 0))
    np.savez_compressed(directory / "states.npz", forward=torch.stack(forward).numpy(),
                        reverse=torch.stack(reverse).numpy(), validation=target.numpy(),
                        factorized=factorized.numpy(), initial_angles=initial.numpy(),
                        angles=angles.numpy(), losses=losses)
    np.savez_compressed(directory / "readout.npz", raw_train=raw_train, raw_validation=raw_validation,
                        effects=torch.stack(effects).numpy(), finer_effects=torch.stack(finer).numpy(),
                        **probabilities, **{f"finer_{name}": p for name, p in fine_p.items()})
    return {"spins": spins, "two_j": two_j, "j": two_j / 2, "dimension": (two_j + 1)**spins,
            **metrics, **checks, **costs}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(directory, protocol):
    paper = ROOT / "materials/paper"
    manifest = paper / "evidence/manifest.json"
    entries = json.loads(manifest.read_text())["files"]
    for entry in entries:
        if sha256(paper / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"frozen evidence hash mismatch: {entry['path']}")
    protocol["frozen_evidence"] = {"manifest_sha256": sha256(manifest), "files_checked": len(entries)}
    write_json(directory / "protocol.json", protocol)
    metadata = reproducibility_metadata(ExperimentConfig(), ROOT)
    metadata.pop("seed")
    metadata["protocol_sha256"] = sha256(directory / "protocol.json")
    write_json(directory / "reproducibility.json", metadata)
    sources = [PROTOCOL, ROOT / "pyproject.toml", Path(__file__).resolve(),
               ROOT / "tasks/plot_spin_grid.py", *sorted((ROOT / "src").rglob("*.py"))]
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        copy = directory / "source" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
        hashes[str(relative)] = sha256(copy)
    write_json(directory / "source_manifest.json", hashes)


def main():
    from tasks.plot_spin_grid import plot_grid

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One update, engineering data only")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (args.outdir or ROOT / f"outputs/experiments/spin_grid_v1_{timestamp}").resolve()
    if not directory.is_relative_to(ROOT / "outputs/experiments"):
        parser.error("--outdir must stay under outputs/experiments")
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spins_grid = (1, 2, 3)
    two_j_grid = (1, 5) if args.smoke else (1, 2, 3, 4, 5)
    seeds = (991,) if args.smoke else (107, 109, 113)
    n_train, n_validation, updates = (32, 48, 1) if args.smoke else (1536, 8192, 160)
    seed_offsets = {"train": 0, "validation": 10_000, "initialization": 20_000, "closed_test": 30_000}
    snapshot(directory, {
        "id": "spin_grid_v1_smoke" if args.smoke else "spin_grid_v1",
        "spins": spins_grid, "two_j": two_j_grid, "seeds": seeds, "seed_offsets": seed_offsets,
        "n_train": n_train, "n_validation": n_validation, "test_generated": False,
        "dataset": {"centers_theta_phi_deg": CENTERS_DEG, "site_rotations_deg": ROTATIONS_DEG,
                    "von_mises_fisher_kappa": CONCENTRATIONS, "shared_weight": SHARED_WEIGHT,
                    "shared_modes": SHARED_MODES, "other_branch": "U,V,(U+V) mod 4"},
        "optimization": {"steps": STEPS, "layers": LAYERS, "adam_lr": LEARNING_RATE,
                         "init_scale": INIT_SCALE, "intermediate_loss_weight": PATH_WEIGHT,
                         "updates": updates, "threads": 1, "checkpoint": "last"},
        "forward": {"diffusion_rate": DIFFUSION_RATE, "time_step": TIME_STEP},
        "readout_grid": READOUT_GRID, "finer_grid": FINER_GRID, "physicality_tolerance": TOLERANCE,
        "selection": "none; validation-only descriptive grid", "stopping": "fixed updates, no retries",
    })
    centers = mode_centers()
    rows = []
    for seed in seeds:
        seed_dir = directory / f"seed_{seed}"
        seed_dir.mkdir()
        train, train_labels = sample_directions(n_train, seed + seed_offsets["train"], centers)
        validation, validation_labels = sample_directions(
            n_validation, seed + seed_offsets["validation"], centers,
        )
        np.savez_compressed(seed_dir / "splits.npz", train=train, validation=validation, centers=centers,
                            train_latent_labels=train_labels, validation_latent_labels=validation_labels)
        raw_train = region_histogram(train, centers)
        raw_validation = region_histogram(validation, centers)
        for spins, two_j in itertools.product(spins_grid, two_j_grid):
            cell = seed_dir / f"n{spins}_two_j{two_j}"
            cell.mkdir()
            row = evaluate_cell(cell, (train, validation), centers, prefix(raw_train, spins),
                                prefix(raw_validation, spins), spins, two_j,
                                seed + seed_offsets["initialization"], updates)
            row["seed"] = seed
            write_json(cell / "metrics.json", row)
            rows.append(row)
            print(f"seed={seed}, N={spins}, j={two_j / 2:g}, "
                  f"TV={row['raw_generated_tv']:.3f}, {row['training_seconds']:.1f}s", flush=True)
    with (directory / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_grid(directory, rows, spins_grid, two_j_grid, len(seeds), updates)
    write_json(directory / "artifact_manifest.json", {
        str(path.relative_to(directory)): sha256(path) for path in sorted(directory.rglob("*"))
        if path.is_file()
    })
    print(directory)


if __name__ == "__main__":
    main()
