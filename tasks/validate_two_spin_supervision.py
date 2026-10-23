"""Replay primary metrics from saved supervision-study states using NumPy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tasks.validate_two_spin import marginal, metrics


ROOT = Path(__file__).resolve().parents[1]


def verify_manifest(directory: Path) -> dict:
    """Verify each sealed artifact before trusting recorded metrics."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    for relative, expected in manifest.items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory):
            raise ValueError(f"artifact path escapes study: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"artifact changed: {relative}")
    return manifest


def validate_study(directory: Path) -> dict:
    """Check sealed artifacts and independently replay all state metrics."""
    directory = Path(directory).resolve()
    manifest = verify_manifest(directory)
    maximum = 0.0
    trace = hermiticity = 0.0
    minimum = 1.0
    rows = json.loads((directory / "results.json").read_text())
    for row in rows:
        folder = directory / f"seed_{row['seed']}"
        density = np.load(folder / f"{row['model']}_rho.npy")
        for split in ("train", "validation"):
            target = np.load(folder / f"rho_{split}.npy")
            recomputed = metrics(density, target)
            differences = [abs(value - row[split][key]) for key, value in recomputed.items()]
            marginal_errors = [0.5 * np.abs(np.linalg.eigvalsh(
                marginal(density, 3, keep) - marginal(target, 3, keep))).sum()
                for keep in (0, 1)]
            differences.append(abs(float(np.mean(marginal_errors))
                                   - row[split]["marginal_trace_distance_mean"]))
            maximum = max(maximum, *differences)
        trajectories = [density]
        trajectory_path = folder / f"{row['model']}_reverse_states.npy"
        if trajectory_path.exists():
            trajectories.extend(np.load(trajectory_path))
            np.testing.assert_allclose(trajectories[1], np.eye(9) / 9, atol=1e-14, rtol=0)
            np.testing.assert_allclose(trajectories[-1], density, atol=1e-14, rtol=0)
        for state in trajectories:
            trace = max(trace, float(abs(np.trace(state) - 1)))
            hermiticity = max(hermiticity, float(np.linalg.norm(state - state.conj().T)))
            minimum = min(minimum, float(np.linalg.eigvalsh((state + state.conj().T) / 2).min()))
    if maximum >= 2e-9 or max(trace, hermiticity) >= 1e-10 or minimum < -1e-10:
        raise AssertionError("independent state-metric/physicality audit failed")
    selection = json.loads((directory / "selection.json").read_text())
    assert selection["test_accessed"] is False
    assert json.loads((directory / "protocol.json").read_text())["test_generated"] is False
    return {"passed": True, "model_rows": len(rows), "split_evaluations": 2 * len(rows),
            "hashed_artifacts": len(manifest), "maximum_metric_discrepancy": maximum,
            "maximum_trace_error": trace, "maximum_hermiticity_error": hermiticity,
            "minimum_eigenvalue": minimum, "test_accessed": False,
            "method": "NumPy/SciPy metrics and physicality from saved density matrices",
            "artifact_manifest_sha256": hashlib.sha256((directory / "artifact_manifest.json").read_bytes()).hexdigest(),
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def validate_confirmation(directory: Path) -> dict:
    """Recompute confirmation metrics from saved test targets and frozen states."""
    directory = Path(directory).resolve()
    manifest = verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    rows = json.loads((directory / "results.json").read_text())
    maximum = 0.0
    expected_models = {"final", "path", "factorized", "heat_kernel",
                       "tensor_mixture", "neural", "classical_ddpm"}
    expected = {(seed, model) for seed in protocol["seeds"] for model in expected_models}
    assert len(rows) == len(expected)
    assert {(row["seed"], row["model"]) for row in rows} == expected
    assert protocol["new_training_updates"] == 0
    assert protocol["model_selection_on_test"] is False
    for row in rows:
        folder = directory / f"seed_{row['seed']}"
        density = np.load(folder / f"{row['model']}_rho.npy")
        target = np.load(folder / "rho_test.npy")
        recomputed = metrics(density, target)
        maximum = max(maximum, *(abs(row["test"][key] - value)
                                 for key, value in recomputed.items()))
        marginal_mean = np.mean([0.5 * np.abs(np.linalg.eigvalsh(
            marginal(density, 3, keep) - marginal(target, 3, keep))).sum()
            for keep in (0, 1)])
        maximum = max(maximum, abs(float(marginal_mean) - row["test"]["marginal_trace_distance_mean"]))
        parent = json.loads((folder / "parent_model_hashes.json").read_text())[row["model"]]
        assert hashlib.sha256((folder / f"{row['model']}_rho.npy").read_bytes()).hexdigest() == parent["state_sha256"]
        for state in (density, target):
            assert abs(np.trace(state) - 1) < 1e-10
            assert np.linalg.norm(state - state.conj().T) < 1e-10
            assert np.linalg.eigvalsh((state + state.conj().T) / 2).min() >= -1e-10
    assert maximum < 2e-9
    return {"passed": True, "model_rows": len(rows), "hashed_artifacts": len(manifest),
            "maximum_metric_discrepancy": maximum, "frozen_output_hashes_match": True,
            "test_model_selection_performed": False,
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "artifact_manifest_sha256": hashlib.sha256((directory / "artifact_manifest.json").read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path,
                        default=ROOT / "outputs/experiments/two_spin_supervision_v1")
    parser.add_argument("--confirmation", action="store_true",
                        help="interpret --study-dir as a confirmation output directory")
    args = parser.parse_args()
    report = validate_confirmation(args.study_dir) if args.confirmation else validate_study(args.study_dir)
    (args.study_dir / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
