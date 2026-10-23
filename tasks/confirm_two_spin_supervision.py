"""Evaluate the frozen supervision choice on previously unopened observations."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from spin_quantum_diffusion.quantum import single_spin as toy
from spin_quantum_diffusion.quantum.correlated import (
    connected_spin_correlation, empirical_joint_density,
    make_correlated_direction_pairs, quantum_mutual_information,
)
from tasks.two_spin_correlated import density_metrics
from tasks.two_spin_supervision import interval, write_json, METRICS
from tasks.validate_two_spin import marginal, metrics as independent_metrics
from tasks.validate_two_spin_supervision import validate_study


ROOT = Path(__file__).resolve().parents[1]
PARENT_PROTOCOL = "387e73320d41964f5a5df7d9edfdf1ab3b123a1ab3a653276a91788e8f508810"
PARENT_ARTIFACT_MANIFEST = "8ac461d7c3b523d1cfefa958a16b84338b66b60570746fa46bc1c0f0c01ad7a5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path,
                        default=ROOT / "outputs/experiments/two_spin_supervision_v1")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "outputs/experiments/two_spin_supervision_confirmation_v1")
    args = parser.parse_args()
    study, output = args.study_dir.resolve(), args.outdir.resolve()
    if not output.is_relative_to(ROOT / "outputs/experiments"):
        raise ValueError("confirmation outputs must remain under outputs/experiments")
    # Check identity and complete validation audit before generating test data.
    audit = validate_study(study)
    protocol = json.loads((study / "protocol.json").read_text())
    selection = json.loads((study / "selection.json").read_text())
    if protocol["protocol_sha256"] != PARENT_PROTOCOL:
        raise ValueError("this confirmation protocol names a different parent study")
    if sha256(study / "artifact_manifest.json") != PARENT_ARTIFACT_MANIFEST:
        raise ValueError("parent artifact identity differs from the frozen confirmation source")
    if selection["validation_candidate"] != "final" or not selection["go_to_confirmation"]:
        raise ValueError("predeclared final-versus-path confirmation is not authorized by validation")
    output.mkdir(parents=True, exist_ok=False)
    frozen = {
        "parent_protocol_sha256": PARENT_PROTOCOL,
        "parent_artifact_manifest_sha256": PARENT_ARTIFACT_MANIFEST,
        "parent_selection_sha256": sha256(study / "selection.json"),
        "selected_mode": "final", "reference_mode": "path",
        "seeds": protocol["seeds"], "n_test": 1200, "test_seed_offset": 30000,
        "new_training_updates": 0, "model_selection_on_test": False,
        "decision": {"minimum_mean_trace_improvement": 0.01,
                     "minimum_nonworse_seed_count": 4,
                     "guardrails": protocol["selection"]["guardrails_vs_path_mean"]},
    }
    write_json(output / "protocol.json", frozen)
    write_json(output / "parent_independent_validation.json", audit)
    source_hashes = {}
    source_paths = [ROOT / "notes/protocols/TWO_SPIN_CONFIRMATION_V1.md", ROOT / "pyproject.toml"]
    for folder in (ROOT / "src/spin_quantum_diffusion", ROOT / "tasks"):
        source_paths.extend(sorted(folder.rglob("*.py")))
    for source in source_paths:
        relative = source.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[str(relative)] = sha256(source)
    write_json(output / "source_manifest.json", source_hashes)
    jx, jy, jz, identity = toy.spin_operators(1.0)
    operators = [jx, jy, jz]
    models = ["path", "final", "factorized", "heat_kernel", "tensor_mixture", "neural", "classical_ddpm"]
    rows = []
    maximum_discrepancy = 0.0
    try:
        for seed in protocol["seeds"]:
            directory = output / f"seed_{seed}"
            directory.mkdir()
            started = time.perf_counter()
            pairs = make_correlated_direction_pairs(1200, 0.18, 0.55, seed + 30000)
            target = empirical_joint_density(pairs, 1.0)
            encoding_seconds = time.perf_counter() - started
            np.save(directory / "test_pairs.npy", np.asarray(pairs))
            np.save(directory / "rho_test.npy", target.numpy())
            toy.assert_physical_density(target, "confirmation target")
            parent_models = {}
            for model in models:
                source = study / f"seed_{seed}" / f"{model}_rho.npy"
                shutil.copyfile(source, directory / source.name)
                parent_models[model] = {"state_sha256": sha256(source)}
                for suffix in ("parameters.pt", "training.json", "metrics.json"):
                    extra = study / f"seed_{seed}" / f"{model}_{suffix}"
                    if extra.exists():
                        shutil.copyfile(extra, directory / extra.name)
                        parent_models[model][suffix + "_sha256"] = sha256(extra)
                if model == "classical_ddpm":
                    for name in ("model.pt", "metadata.json"):
                        extra = study / f"seed_{seed}" / "classical" / name
                        shutil.copyfile(extra, directory / f"classical_{name}")
                        parent_models[model][name + "_sha256"] = sha256(extra)
                state = np.load(directory / source.name)
                toy.assert_physical_density(torch.tensor(state), model)
                # Independent calculation is performed on files saved in this confirmation.
                saved_target = np.load(directory / "rho_test.npy")
                values = independent_metrics(state, saved_target)
                marginal_errors = [0.5 * float(np.abs(np.linalg.eigvalsh(
                    marginal(state, 3, keep) - marginal(saved_target, 3, keep))).sum())
                    for keep in (0, 1)]
                replay = density_metrics(torch.tensor(state), target,
                    quantum_mutual_information(target, 3),
                    connected_spin_correlation(target, operators, identity), operators, identity)
                maximum_discrepancy = max(maximum_discrepancy, max(
                    abs(replay[key] - value) for key, value in values.items()))
                if maximum_discrepancy >= 2e-9:
                    raise AssertionError("confirmation independent metric replay failed")
                values["marginal_trace_distance_mean"] = float(np.mean(marginal_errors))
                values["marginal_trace_distances"] = marginal_errors
                rows.append({"seed": seed, "model": model, "test": values,
                             "test_generation_and_encoding_seconds": encoding_seconds})
            write_json(directory / "parent_model_hashes.json", parent_models)
        means = {model: {key: interval([r["test"][key] for r in rows if r["model"] == model])
                         for key in METRICS} for model in models}
        reference = {r["seed"]: r for r in rows if r["model"] == "path"}
        selected = [r for r in rows if r["model"] == "final"]
        paired = {key: interval([r["test"][key] - reference[r["seed"]]["test"][key]
                                 for r in selected]) for key in METRICS}
        wins = sum(r["test"]["trace_distance"] <= reference[r["seed"]]["test"]["trace_distance"]
                   for r in selected)
        guardrails = all(means["final"][key]["mean"] <= means["path"][key]["mean"] + margin
                         for key, margin in frozen["decision"]["guardrails"].items())
        improvement = means["path"]["trace_distance"]["mean"] - means["final"]["trace_distance"]["mean"]
        report = {
            "selected_mode_frozen_before_test": "final", "test_model_selection_performed": False,
            "fitted_model_realizations": len(protocol["seeds"]), "models": means, "paired_final_minus_path": paired,
            "nonworse_seed_count": wins, "absolute_mean_trace_improvement": improvement,
            "guardrails_passed": guardrails,
            "supervision_confirmation_passed": bool(guardrails and wins >= 4 and improvement >= 0.01),
            "maximum_independent_metric_discrepancy": maximum_discrepancy,
            "correlated_competitiveness_established": False,
            "mean_trace_distance_below_all_baselines": all(
                means["final"]["trace_distance"]["mean"] <= means[model]["trace_distance"]["mean"]
                for model in models if model not in {"final", "path"}),
            "scope": "Five previously fitted models, independent test observations, one data family; no physical-POVM sample claim.",
        }
        write_json(output / "results.json", rows)
        write_json(output / "confirmation.json", report)
        write_json(output / "artifact_manifest.json", {
            str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
        })
        print(json.dumps(report, indent=2))
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                             "completed_rows": len(rows)})
        raise


if __name__ == "__main__":
    main()
