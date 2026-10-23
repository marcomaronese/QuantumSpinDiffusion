"""Independent NumPy/SciPy replay of local-control state and channel outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from tasks.two_spin_supervision import ROOT, write_json
from tasks.validate_two_spin import marginal, metrics, connected_correlation, mutual_information
from tasks.validate_two_spin_supervision import validate_study, verify_manifest


def validate_local_control(directory):
    from tasks.two_spin_local_control import MODELS, BASELINES, summary, sha256
    directory = Path(directory).resolve()
    manifest = verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    rows = json.loads((directory / "results.json").read_text())
    # All source copies are sealed too; ensure their declared fingerprints match.
    for relative, digest in json.loads((directory / "source_manifest.json").read_text()).items():
        if sha256(directory / "source" / relative) != digest:
            raise AssertionError("source snapshot fingerprint mismatch")
    declared = protocol.pop("protocol_sha256")
    import hashlib
    if hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != declared:
        raise AssertionError("protocol hash mismatch")
    is_test = protocol["test_generated"]
    expected_models = protocol["conditions"] if is_test else [*MODELS, *BASELINES]
    expected = {(seed, model) for seed in protocol["seeds"] for model in expected_models}
    if len(rows) != len(expected) or {(r["seed"], r["model"]) for r in rows} != expected:
        raise AssertionError("missing or duplicate model/seed identities")
    maximum_metric = maximum_channel = completeness = product_residual = 0.
    minimum_choi = 1.
    if not is_test:
        base_audit = validate_study(directory)
        maximum_metric = base_audit["maximum_metric_discrepancy"]
        for seed in protocol["seeds"]:
            folder = directory / f"seed_{seed}"
            for model in MODELS:
                generators = np.load(folder / f"{model}_generators.npy")
                parameters = np.load(folder / f"{model}_parameters.npy")
                saved = np.load(folder / f"{model}_reverse_states.npy")
                for gen in generators:
                    np.testing.assert_allclose(gen, gen.conj().T, atol=1e-13, rtol=0)
                    np.testing.assert_allclose(np.linalg.norm(gen), 1., atol=1e-13, rtol=0)
                current = np.eye(9, dtype=complex) / 9
                for index, theta in enumerate(parameters):
                    unitary = np.eye(generators.shape[-1], dtype=complex)
                    for layer in theta:
                        for angle, generator in zip(layer, generators, strict=True):
                            unitary = expm(-1j * angle * generator) @ unitary
                    ancillas = unitary.shape[0] // 9
                    blocks = unitary.reshape(9, ancillas, 9, ancillas)
                    kraus = [blocks[:, a, :, 0] for a in range(ancillas)]
                    completeness = max(completeness, float(np.linalg.norm(
                        sum(k.conj().T @ k for k in kraus) - np.eye(9))))
                    vectors = [k.T.reshape(-1) for k in kraus]
                    choi = sum(np.outer(v, v.conj()) for v in vectors)
                    minimum_choi = min(minimum_choi, float(np.linalg.eigvalsh(choi).min()))
                    current = sum(k @ current @ k.conj().T for k in kraus)
                    maximum_channel = max(maximum_channel, float(np.linalg.norm(current - saved[index+1])))
                    if model == MODELS[0]:
                        product_residual = max(product_residual,
                            float(np.linalg.norm(current - np.kron(marginal(current, 3, 0), marginal(current, 3, 1)))),
                            float(np.linalg.norm(connected_correlation(current))),
                            abs(mutual_information(current, 3)))
        recorded = json.loads((directory / "selection.json").read_text())
        if summary(rows) != recorded:
            raise AssertionError("validation decision replay mismatch")
    else:
        if protocol["new_training_updates"] != 0 or protocol["model_selection_on_test"]:
            raise AssertionError("confirmation contract violated")
        for row in rows:
            folder = directory / f"seed_{row['seed']}"
            path = folder / f"{row['model']}_rho.npy"
            if sha256(path) != protocol["frozen_state_hashes"][str(path.relative_to(directory))]:
                raise AssertionError("frozen output changed")
            density, target = np.load(path), np.load(folder / "rho_test.npy")
            for state in (density, target):
                np.testing.assert_allclose(state, state.conj().T, atol=1e-10, rtol=0)
                if abs(np.trace(state)-1) >= 1e-10 or np.linalg.eigvalsh(state).min() < -1e-10:
                    raise AssertionError("confirmation state physicality failed")
            replay = metrics(density, target)
            maximum_metric = max(maximum_metric, *(abs(row["test"][k]-v) for k, v in replay.items()))
            marginal_mean = np.mean([.5*np.abs(np.linalg.eigvalsh(
                marginal(density, 3, k)-marginal(target, 3, k))).sum() for k in (0, 1)])
            maximum_metric = max(maximum_metric, abs(marginal_mean-row["test"]["marginal_trace_distance_mean"]))
        recorded = json.loads((directory / "confirmation.json").read_text())
        if summary(rows, "test", protocol["selected"]) != recorded:
            raise AssertionError("frozen confirmation decision mismatch")
    if maximum_metric >= 2e-9 or max(maximum_channel, completeness, product_residual) >= 1e-10 or minimum_choi < -1e-10:
        raise AssertionError("independent local-control audit failed")
    return {"passed": True, "model_rows": len(rows), "hashed_artifacts": len(manifest),
            "maximum_metric_discrepancy": maximum_metric,
            "maximum_channel_replay_discrepancy": maximum_channel if not is_test else None,
            "maximum_kraus_completeness_error": completeness if not is_test else None,
            "minimum_choi_eigenvalue": minimum_choi if not is_test else None,
            "maximum_product_correlation_or_mi_residual": product_residual if not is_test else None,
            "test_accessed": is_test, "decision_replayed": True,
            "method": "NumPy/SciPy metrics; screen channels replayed with scipy.linalg.expm, no Torch channel code",
            "artifact_manifest_sha256": sha256(directory / "artifact_manifest.json"),
            "audit_source_sha256": sha256(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "outputs/experiments/two_spin_local_control_v1")
    args = parser.parse_args()
    report = validate_local_control(args.study_dir)
    write_json(args.study_dir / "independent_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
