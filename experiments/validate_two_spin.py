#!/usr/bin/env python3
"""Independent NumPy/SciPy audit of the saved two-spin pilot metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm


ROOT = Path(__file__).resolve().parents[1]


def marginal(density: np.ndarray, dimension: int, keep: int) -> np.ndarray:
    blocks = density.reshape(dimension, dimension, dimension, dimension)
    if keep == 0:
        return np.einsum("abcb->ac", blocks)
    if keep == 1:
        return np.einsum("abad->bd", blocks)
    raise ValueError("keep must be 0 or 1")


def entropy(density: np.ndarray) -> float:
    hermitian = 0.5 * (density + density.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    positive = eigenvalues[eigenvalues > 1e-12]
    return -float(np.sum(positive * np.log(positive)))


def mutual_information(density: np.ndarray, dimension: int) -> float:
    return (
        entropy(marginal(density, dimension, keep=0))
        + entropy(marginal(density, dimension, keep=1))
        - entropy(density)
    )


def spin_operators_j1() -> list[np.ndarray]:
    scale = 1 / np.sqrt(2)
    jx = scale * np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex
    )
    jy = scale * np.array(
        [[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex
    )
    jz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    return [jx, jy, jz]


def connected_correlation(density: np.ndarray) -> np.ndarray:
    operators = spin_operators_j1()
    identity = np.eye(3)
    first_means = [
        np.trace(density @ np.kron(operator, identity)).real
        for operator in operators
    ]
    second_means = [
        np.trace(density @ np.kron(identity, operator)).real
        for operator in operators
    ]
    result = np.zeros((3, 3))
    for first_index, first in enumerate(operators):
        for second_index, second in enumerate(operators):
            result[first_index, second_index] = (
                np.trace(density @ np.kron(first, second)).real
                - first_means[first_index] * second_means[second_index]
            )
    return result


def fidelity(first: np.ndarray, second: np.ndarray) -> float:
    root = sqrtm(first)
    sandwich = root @ second @ root
    sandwich = 0.5 * (sandwich + sandwich.conj().T)
    eigenvalues = np.linalg.eigvalsh(sandwich)
    return float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))) ** 2)


def metrics(density: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = density - target
    difference = 0.5 * (difference + difference.conj().T)
    target_information = mutual_information(target, 3)
    information = mutual_information(density, 3)
    target_correlation = connected_correlation(target)
    return {
        "frobenius_error": float(np.sum(np.abs(density - target) ** 2).real),
        "trace_distance": 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference)))),
        "infidelity": 1.0 - fidelity(density, target),
        "quantum_mutual_information": information,
        "mutual_information_absolute_error": abs(information - target_information),
        "connected_correlation_error": float(
            np.linalg.norm(connected_correlation(density) - target_correlation)
        ),
        "minimum_eigenvalue": float(
            np.linalg.eigvalsh(0.5 * (density + density.conj().T)).min()
        ),
        "trace_error": abs(float(np.trace(density).real) - 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=ROOT / "two_spin_pilot",
    )
    args = parser.parse_args()
    pilot_directory = args.pilot_dir.resolve()
    saved = np.load(pilot_directory / "density_matrices.npz")
    target = saved["target_test"]
    state_names = {
        "Factorized marginals": "factorized",
        "Riemannian heat-kernel KDE": "riemannian_heat_kernel",
        "Separable tensor mixture (chi=2)": "tensor_mixture",
        "Quantum reverse, no direct J1-J2": "quantum_no_direct",
        "Quantum reverse, direct J1-J2": "quantum_direct",
        "Matched-parameter neural generator": "neural",
    }
    with (pilot_directory / "comparison.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        declared_rows = {row["model"]: row for row in csv.DictReader(handle)}

    fields = [
        "frobenius_error",
        "trace_distance",
        "infidelity",
        "quantum_mutual_information",
        "mutual_information_absolute_error",
        "connected_correlation_error",
        "minimum_eigenvalue",
        "trace_error",
    ]
    audits = []
    maximum_absolute_discrepancy = 0.0
    for model, state_name in state_names.items():
        independent = metrics(saved[state_name], target)
        discrepancies = {
            field: abs(float(declared_rows[model][field]) - independent[field])
            for field in fields
        }
        maximum_absolute_discrepancy = max(
            maximum_absolute_discrepancy,
            max(discrepancies.values()),
        )
        audits.append(
            {
                "model": model,
                "independent_metrics": independent,
                "absolute_discrepancies": discrepancies,
            }
        )
    if maximum_absolute_discrepancy > 2e-9:
        raise AssertionError(
            f"two-spin metric audit failed: discrepancy={maximum_absolute_discrepancy}"
        )
    result = {
        "method": "independent NumPy/SciPy recomputation from density_matrices.npz",
        "maximum_absolute_discrepancy": maximum_absolute_discrepancy,
        "models": audits,
    }
    (pilot_directory / "independent_validation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(
        "two-spin independent validation passed; "
        f"maximum discrepancy={maximum_absolute_discrepancy:.3e}"
    )


if __name__ == "__main__":
    main()
