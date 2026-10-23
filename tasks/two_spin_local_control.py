"""Frozen E5 local/shared-ancilla diagnosis; never select on confirmation data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from spin_quantum_diffusion.quantum import single_spin as toy
from spin_quantum_diffusion.quantum.config import ExperimentConfig, reproducibility_metadata
from spin_quantum_diffusion.quantum.correlated import (
    build_independent_two_spin_reverse_generators, build_two_spin_reverse_generators,
    build_two_spin_forward_trajectory, make_correlated_direction_pairs,
    empirical_joint_density, factorized_density, clustered_product_mixture_density,
    sample_riemannian_heat_kernel, partial_trace_two_spin, connected_spin_correlation,
    quantum_mutual_information, joint_multipole_coefficients,
)
from spin_quantum_diffusion.quantum.multipoles import irreducible_spherical_tensors
from spin_quantum_diffusion.classical.diffusion import ClassicalDiffusionConfig
from spin_quantum_diffusion.classical.benchmark import run_classical_benchmark
from tasks.two_spin_correlated import (
    density_metrics, reverse_cptp_residuals, train_neural_baseline, sample_neural_pairs,
)
from tasks.two_spin_supervision import ROOT, METRICS, interval, write_json
from tasks.validate_two_spin import metrics as independent_metrics
from tasks.validate_two_spin_supervision import verify_manifest


MODELS = ("independent_local", "shared_no_direct", "shared_direct")
BASELINES = ("factorized", "tensor_mixture", "heat_kernel", "neural", "classical_ddpm")
GUARDRAILS = {"infidelity": .005, "marginal_trace_distance_mean": .01}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(root):
    write_json(root / "artifact_manifest.json", {
        str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"
    })


def snapshot(root, protocol):
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(root / "protocol.json", protocol)
    sources = [ROOT / "pyproject.toml", ROOT / "notes/protocols/TWO_SPIN_LOCAL_CONTROL_V1.md"]
    for folder in ("src/spin_quantum_diffusion", "tasks", "tests"):
        sources.extend(sorted((ROOT / folder).rglob("*.py")))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[str(relative)] = sha256(source)
    write_json(root / "source_manifest.json", hashes)
    metadata = reproducibility_metadata(ExperimentConfig(), ROOT)
    metadata.pop("seed", None)
    metadata["study_seeds"] = protocol["seeds"]
    metadata["protocol_sha256"] = protocol["protocol_sha256"]
    write_json(root / "reproducibility.json", metadata)


def summary(rows, split="validation", selected=None):
    models = sorted({r["model"] for r in rows})
    means = {m: {key: interval([r[split][key] for r in rows if r["model"] == m])
                 for key in METRICS} for m in models}
    reference = {r["seed"]: r for r in rows if r["model"] == MODELS[0]}
    paired, eligible = {}, []
    for model in MODELS[1:]:
        if model not in means:
            continue
        candidate = [r for r in rows if r["model"] == model]
        paired[model] = {k: interval([r[split][k] - reference[r["seed"]][split][k]
                                     for r in candidate]) for k in METRICS}
        if all(r["physicality_passed"] for r in candidate) and all(
            means[model][k]["mean"] <= means[MODELS[0]][k]["mean"] + margin
            for k, margin in GUARDRAILS.items()
        ):
            eligible.append(model)
    physical = all(r["physicality_passed"] for r in rows)
    # For test data the candidate identity is supplied from sealed validation;
    # there is no fallback or ranking by test performance.
    winner = selected if split == "test" else (
        min(eligible, key=lambda m: (means[m]["trace_distance"]["mean"], MODELS.index(m)))
        if eligible and physical else None)
    wins = sum(r[split]["trace_distance"] <= reference[r["seed"]][split]["trace_distance"]
               for r in rows if r["model"] == winner)
    deltas = paired.get(winner, {})
    trace_improvement = -deltas["trace_distance"]["mean"] if deltas else 0.
    corr_improvement = -deltas["connected_correlation_error"]["mean"] if deltas else 0.
    passed = bool(physical and winner in eligible and trace_improvement >= .01
                  and corr_improvement >= .01 and wins >= np.ceil(.8 * len(reference)))
    return {"models": means, "paired_minus_independent_local": paired,
            "eligible": eligible, "validation_candidate": winner,
            "absolute_mean_trace_improvement": trace_improvement,
            "absolute_mean_correlation_improvement": corr_improvement,
            "nonworse_seed_count": wins, "correlation_gate_passed": passed,
            "go_to_confirmation": passed if split == "validation" else False,
            "test_accessed": split == "test", "model_selection_on_test": False,
            "scope": f"{len(reference)} paired training realizations, one family, exact encoded-state diagnosis; unequal resources.",
            "intervals": "Descriptive paired Student-t 95%; no population or adjusted-significance claim."}


def initial_angles(seed, model, names, shared_names):
    if model == "independent_local":
        return torch.cat([.03 * torch.randn(3, 1, 8, dtype=toy.RDTYPE,
                          generator=torch.Generator().manual_seed(seed + offset))
                          for offset in (220000, 320000)], dim=2)
    paired = .03 * torch.randn(3, 1, len(shared_names), dtype=toy.RDTYPE,
                              generator=torch.Generator().manual_seed(seed + 120000))
    return paired[:, :, [shared_names.index(name) for name in names]]


def evaluate(directory, seed, model, state, targets, basis, operators, identity,
             training_seconds=0., generation_seconds=0., parameters=0):
    values = {}
    coefficients = joint_multipole_coefficients(state, basis)
    keys = list(coefficients)
    discrepancy = 0.
    for split, target in targets.items():
        result = density_metrics(state, target, quantum_mutual_information(target, 3),
                                 connected_spin_correlation(target, operators, identity), operators, identity)
        replay = independent_metrics(state.numpy(), target.numpy())
        discrepancy = max(discrepancy, *(abs(result[k] - v) for k, v in replay.items()))
        if discrepancy >= 2e-9:
            raise AssertionError("independent metric mismatch")
        marginal_errors = [float(toy.trace_distance(partial_trace_two_spin(state, 3, keep),
                            partial_trace_two_spin(target, 3, keep))) for keep in (0, 1)]
        result["marginal_trace_distances"] = marginal_errors
        result["marginal_trace_distance_mean"] = float(np.mean(marginal_errors))
        target_coeffs = joint_multipole_coefficients(target, basis)
        rank_errors = {}
        for k in keys:
            rank = f"{k[0]},{k[2]}"
            rank_errors[rank] = rank_errors.get(rank, 0.) + float(abs(coefficients[k] - target_coeffs[k])**2)
        if abs(sum(rank_errors.values()) - result["frobenius_error"]) >= 1e-12:
            raise AssertionError("joint multipole Parseval check failed")
        result["multipole_rank_squared_errors"] = rank_errors
        np.savez(directory / f"{model}_{split}_multipoles.npz", keys=np.asarray(keys),
                 predicted=np.asarray([complex(coefficients[k]) for k in keys]),
                 target=np.asarray([complex(target_coeffs[k]) for k in keys]))
        values[split] = result
    np.save(directory / f"{model}_rho.npy", state.numpy())
    np.save(directory / f"{model}_connected.npy", connected_spin_correlation(state, operators, identity).numpy())
    row = {"seed": seed, "model": model, "trainable_parameters": parameters,
           "training_seconds": training_seconds, "generation_seconds": generation_seconds,
           "physicality_passed": True, "independent_metric_maximum_discrepancy": discrepancy, **values}
    write_json(directory / f"{model}_metrics.json", row)
    return row


def fit_baselines(pairs, target, directory, seed, args):
    started = time.perf_counter()
    yield "factorized", factorized_density(target, 3), 0., time.perf_counter() - started, 0
    started = time.perf_counter()
    state = clustered_product_mixture_density(pairs, 1., 2, seed + 170000)
    yield "tensor_mixture", state, time.perf_counter() - started, 0., 0
    started = time.perf_counter()
    samples = sample_riemannian_heat_kernel(pairs, args.baseline_samples, .1, seed + 160000)
    np.save(directory / "heat_kernel_pairs.npy", np.asarray(samples))
    state = empirical_joint_density(samples, 1.)
    yield "heat_kernel", state, 0., time.perf_counter() - started, 0
    started = time.perf_counter()
    model, history = train_neural_baseline(pairs, 39, args.baseline_epochs, seed + 140000)
    training = time.perf_counter() - started
    torch.save(model.state_dict(), directory / "neural_parameters.pt")
    write_json(directory / "neural_history.json", history)
    started = time.perf_counter()
    samples = sample_neural_pairs(model, args.baseline_samples, seed + 180000)
    np.save(directory / "neural_pairs.npy", np.asarray(samples))
    state = empirical_joint_density(samples, 1.)
    yield "neural", state, training, time.perf_counter() - started, 39
    samples, meta = run_classical_benchmark(pairs, ClassicalDiffusionConfig(epochs=args.baseline_epochs),
        seed + 150000, args.baseline_samples, directory / "classical")
    state = empirical_joint_density(samples, 1.)
    yield "classical_ddpm", state, meta["training_seconds"], meta["sampling_seconds"], meta["trainable_parameter_count"]


def screen(root, args, operators, identity, basis):
    configurations = {MODELS[0]: build_independent_two_spin_reverse_generators(operators, identity),
                      MODELS[1]: build_two_spin_reverse_generators(operators, identity, False),
                      MODELS[2]: build_two_spin_reverse_generators(operators, identity, True)}
    protocol = {
        "version": "two_spin_local_control_v1", "seeds": args.seeds,
        "n_train": args.n_data, "n_validation": args.n_data, "closed_test_samples": 1200,
        "test_generated": False, "conditions": list(MODELS),
        "split_seed_offsets": {"train": 0, "validation": 110000, "closed_test": 130000},
        "initialization_seed_offsets": {"shared": 120000, "local_first": 220000, "local_second": 320000},
        "dataset": {"family": "rotated normalized-Gaussian pairs", "sigma": .18, "rotation_angle": .55},
        "architecture": {"j": 1., "steps": 3, "layers": 1, "prior": "I9/9", "D": 1., "dt": .5,
            "models": {m: {"parameters": 3 * len(g), "collision_dimension": g[0].shape[0],
                           "ancilla_dimension": g[0].shape[0] // 9, "names": names}
                       for m, (g, names) in configurations.items()}},
        "optimization": {"supervision": "final", "objective": "infidelity", "optimizer": "Adam",
                         "lr": .04, "betas": [.9, .999], "eps": 1e-8, "updates": args.epochs,
                         "channel_evaluations_per_update": 3, "early_stopping": False, "restarts": 0},
        "selection": {"reference": MODELS[0], "primary_metric": "validation trace distance",
                      "guardrails": GUARDRAILS, "trace_improvement": .01, "correlation_improvement": .01,
                      "nonworse_fraction": .8, "tie_order": list(MODELS[1:]),
                      "physicality_tolerance": 1e-10, "replay_tolerance": 2e-9},
        "baselines": {"conditions": list(BASELINES), "neural_parameters": 39,
                      "neural_epochs": args.baseline_epochs, "neural_lr": .03,
                      "ddpm": asdict(ClassicalDiffusionConfig(epochs=args.baseline_epochs)),
                      "heat_bandwidth": .1, "tensor_bond": 2, "samples": args.baseline_samples,
                      "seed_offsets": {"neural": 140000, "ddpm": 150000, "heat": 160000,
                                       "tensor": 170000, "neural_sampling": 180000}},
        "limits": ["One family; unequal parameters, ancilla ranks, generator scales and compute.",
                   "Exact quantum states versus finite sampled classical states; no physical POVM.",
                   "Quantum MI is not classical source MI; source MI not estimated."]}
    snapshot(root, protocol)
    rows = []
    for seed in args.seeds:
        directory = root / f"seed_{seed}"
        directory.mkdir()
        pairs = make_correlated_direction_pairs(args.n_data, .18, .55, seed)
        validation = make_correlated_direction_pairs(args.n_data, .18, .55, seed + 110000)
        targets = {"train": empirical_joint_density(pairs, 1.),
                   "validation": empirical_joint_density(validation, 1.)}
        np.save(directory / "train_pairs.npy", np.asarray(pairs))
        np.save(directory / "validation_pairs.npy", np.asarray(validation))
        for split, target in targets.items():
            toy.assert_physical_density(target, split)
            np.save(directory / f"rho_{split}.npy", target.numpy())
        forward = build_two_spin_forward_trajectory(targets["train"], operators, identity, 1., .5, 3)
        np.save(directory / "forward_states.npy", torch.stack(forward).numpy())
        write_json(directory / "dataset_manifest.json", {p.name: sha256(p) for p in sorted(directory.glob("*.npy"))})
        for model, (generators, names) in configurations.items():
            initial = initial_angles(seed, model, names, configurations[MODELS[2]][1])
            np.save(directory / f"{model}_initial.npy", initial.numpy())
            np.save(directory / f"{model}_generators.npy", torch.stack(generators).numpy())
            diagnostics = []
            started = time.perf_counter()
            parameters, history = toy.train_reverse(forward, generators, 1, args.epochs, .04,
                seed + 120000, supervision="final", objective="fidelity", initial_parameters=initial,
                training_diagnostics=diagnostics)
            training = time.perf_counter() - started
            started = time.perf_counter()
            trajectory = toy.generate_trajectory(torch.eye(9, dtype=toy.CDTYPE)/9, parameters, generators)
            generation = time.perf_counter() - started
            physical = [toy.assert_physical_density(state, model) for state in trajectory]
            completeness, minimum_choi = reverse_cptp_residuals(parameters, generators, 9, 3)
            if completeness >= 1e-10 or minimum_choi < -1e-10:
                raise AssertionError("channel physicality failed")
            product_error = max(float(torch.linalg.norm(s - factorized_density(s, 3))) for s in trajectory)
            if model == MODELS[0] and (product_error >= 1e-10 or any(
                abs(quantum_mutual_information(s, 3)) >= 1e-10 or
                float(torch.linalg.norm(connected_spin_correlation(s, operators, identity))) >= 1e-10
                for s in trajectory)):
                raise AssertionError("independent local channel generated correlations")
            np.save(directory / f"{model}_parameters.npy", parameters.numpy())
            np.save(directory / f"{model}_reverse_states.npy", torch.stack(trajectory).numpy())
            write_json(directory / f"{model}_training.json", {
                "loss_history_before_updates": history, "diagnostics": diagnostics,
                "optimizer_updates": args.epochs, "channel_evaluations": 3*args.epochs,
                "state_residuals": physical, "kraus_completeness_error": completeness,
                "minimum_choi_eigenvalue": minimum_choi, "product_residual": product_error,
                "generator_names": names})
            row = evaluate(directory, seed, model, trajectory[-1], targets, basis, operators,
                           identity, training, generation, parameters.numel())
            rows.append(row)
            print(f"seed={seed} {model} validation_trace={row['validation']['trace_distance']:.6f}", flush=True)
        for model, state, training, generation, count in fit_baselines(pairs, targets["train"], directory, seed, args):
            rows.append(evaluate(directory, seed, model, state, targets, basis, operators,
                                 identity, training, generation, count))
        write_json(root / "results.json", rows)
    write_json(root / "selection.json", summary(rows))
    seal(root)
    return summary(rows)


def confirm(root, parent, operators, identity, basis):
    from tasks.validate_two_spin_local_control import validate_local_control
    audit = validate_local_control(parent)
    selection = json.loads((parent / "selection.json").read_text())
    if not selection["go_to_confirmation"]:
        raise ValueError("screen did not pass: confirmation must remain closed")
    selected = selection["validation_candidate"]
    original = json.loads((parent / "protocol.json").read_text())
    models = [MODELS[0], selected, *BASELINES]
    protocol = {"version": "two_spin_local_control_confirmation_v1", "seeds": original["seeds"],
        "test_generated": True, "new_training_updates": 0, "model_selection_on_test": False,
        "selected": selected, "conditions": models, "n_test": original["closed_test_samples"],
        "test_offset": original["split_seed_offsets"]["closed_test"],
        "parent_path": str(parent), "parent_audit": audit,
        "parent_hashes": {name: sha256(parent / name) for name in
                          ("protocol.json", "selection.json", "artifact_manifest.json")},
        "frozen_state_hashes": {str(Path(f"seed_{s}") / f"{m}_rho.npy"):
            sha256(parent / f"seed_{s}" / f"{m}_rho.npy") for s in original["seeds"] for m in models},
        "decision": original["selection"], "scope": original["limits"]}
    snapshot(root, protocol)  # All selected identities and state hashes precede test creation.
    rows = []
    for seed in protocol["seeds"]:
        directory = root / f"seed_{seed}"
        directory.mkdir()
        pairs = make_correlated_direction_pairs(protocol["n_test"], .18, .55, seed + protocol["test_offset"])
        np.save(directory / "test_pairs.npy", np.asarray(pairs))
        target = empirical_joint_density(pairs, 1.)
        np.save(directory / "rho_test.npy", target.numpy())
        for model in models:
            source = parent / f"seed_{seed}" / f"{model}_rho.npy"
            state = torch.tensor(np.load(source))
            row = evaluate(directory, seed, model, state, {"test": target}, basis, operators, identity)
            if sha256(directory / source.name) != protocol["frozen_state_hashes"][str(source.relative_to(parent))]:
                raise AssertionError("frozen state changed")
            rows.append(row)
    result = summary(rows, "test", selected)
    write_json(root / "results.json", rows)
    write_json(root / "confirmation.json", result)
    seal(root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/experiments/two_spin_local_control_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 103, 107, 109, 113])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--n-data", type=int, default=1200)
    parser.add_argument("--baseline-epochs", type=int, default=300)
    parser.add_argument("--baseline-samples", type=int, default=2000)
    parser.add_argument("--confirm-from", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.baseline_epochs, args.baseline_samples) < 1 or args.n_data < 2:
        raise ValueError("positive budgets and at least two data points required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("distinct training seeds required")
    root = args.outdir.resolve()
    if not root.is_relative_to(ROOT / "outputs/experiments"):
        raise ValueError("outputs must remain under outputs/experiments")
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    jx, jy, jz, identity = toy.spin_operators(1.)
    basis = irreducible_spherical_tensors(1., jx, jy)
    try:
        result = (confirm(root, args.confirm_from.resolve(), [jx, jy, jz], identity, basis)
                  if args.confirm_from else screen(root, args, [jx, jy, jz], identity, basis))
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        write_json(root / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
