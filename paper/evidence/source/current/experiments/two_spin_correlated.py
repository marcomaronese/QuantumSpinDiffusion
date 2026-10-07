#!/usr/bin/env python3
"""Separate two-spin correlated-generation pilot with explicit baselines.

This is a small exact experiment, not an advantage benchmark.  Quantum models
are compared with a factorized density, a spherical heat-kernel KDE, a
physical low-bond separable tensor mixture, and a matched-parameter neural
generator.  The neural and quantum models use the same trainable parameter
count for the direct-interaction configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spin_diffusion_toy as toy  # noqa: E402
from spin_classical_diffusion import add_classical_arguments, config_from_arguments
from spin_classical_benchmark import run_classical_benchmark
from spin_correlated import (  # noqa: E402
    angles_to_vector,
    build_two_spin_forward_trajectory,
    build_two_spin_reverse_generators,
    clustered_product_mixture_density,
    connected_spin_correlation,
    empirical_joint_density,
    factorized_density,
    make_correlated_direction_pairs,
    quantum_mutual_information,
    sample_riemannian_heat_kernel,
)
from spin_reverse import (  # noqa: E402
    choi_matrix_from_kraus,
    collision_kraus_operators,
    parameter_schedule,
    reverse_collision_unitary,
)


class DirectionPairGenerator(torch.nn.Module):
    """Small generic MLP generator with normalized 3D direction outputs."""

    def __init__(self, noise_dimension: int, hidden_dimension: int):
        super().__init__()
        self.first = torch.nn.Linear(noise_dimension, hidden_dimension)
        self.second = torch.nn.Linear(hidden_dimension, 6)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        raw = self.second(torch.tanh(self.first(noise)))
        first = raw[:, :3] / torch.linalg.norm(
            raw[:, :3], dim=1, keepdim=True
        ).clamp_min(1e-12)
        second = raw[:, 3:] / torch.linalg.norm(
            raw[:, 3:], dim=1, keepdim=True
        ).clamp_min(1e-12)
        return torch.cat([first, second], dim=1)


def pair_features(pairs) -> torch.Tensor:
    return torch.tensor(
        np.asarray(
            [
                np.concatenate(
                    [angles_to_vector(*first), angles_to_vector(*second)]
                )
                for first, second in pairs
            ]
        ),
        dtype=toy.RDTYPE,
    )


def moment_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match first, within-spin second, and cross-spin moments."""
    generated_mean = generated.mean(dim=0)
    target_mean = target.mean(dim=0)
    generated_second = generated.T @ generated / generated.shape[0]
    target_second = target.T @ target / target.shape[0]
    return torch.sum((generated_mean - target_mean) ** 2) + torch.sum(
        (generated_second - target_second) ** 2
    )


def train_neural_baseline(
    train_pairs,
    parameter_target: int,
    epochs: int,
    seed: int,
) -> tuple[DirectionPairGenerator, list[float]]:
    noise_dimension = 4
    hidden_dimension = max(1, round((parameter_target - 6) / 11))
    torch.manual_seed(seed)
    model = DirectionPairGenerator(noise_dimension, hidden_dimension)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != parameter_target:
        raise ValueError(
            f"cannot exactly match {parameter_target} parameters; got {parameter_count}"
        )
    target = pair_features(train_pairs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        noise = torch.randn(len(target), noise_dimension, dtype=toy.RDTYPE)
        generated = model(noise)
        loss = moment_loss(generated, target)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


@torch.no_grad()
def sample_neural_pairs(
    model: DirectionPairGenerator,
    n_samples: int,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        n_samples,
        model.first.in_features,
        generator=generator,
        dtype=toy.RDTYPE,
    )
    samples = model(noise).numpy()
    return [
        (toy.xyz_to_angles(row[:3]), toy.xyz_to_angles(row[3:]))
        for row in samples
    ]


def density_metrics(
    density: torch.Tensor,
    target: torch.Tensor,
    target_mutual_information: float,
    target_correlation: torch.Tensor,
    operators,
    identity,
) -> dict[str, float]:
    physicality = toy.assert_physical_density(density, "correlated model density")
    difference = density - target
    hermitian_difference = 0.5 * (difference + difference.conj().T)
    mutual_information = quantum_mutual_information(density, identity.shape[0])
    correlation = connected_spin_correlation(
        density,
        operators,
        identity,
    )
    return {
        "frobenius_error": float(torch.sum(torch.abs(difference) ** 2)),
        "trace_distance": 0.5
        * float(torch.linalg.eigvalsh(hermitian_difference).abs().sum()),
        "infidelity": 1.0 - float(toy.quantum_fidelity(density, target)),
        "quantum_mutual_information": mutual_information,
        "mutual_information_absolute_error": abs(
            mutual_information - target_mutual_information
        ),
        "connected_correlation_error": float(
            torch.linalg.norm(correlation - target_correlation)
        ),
        "minimum_eigenvalue": physicality["minimum_eigenvalue"],
        "trace_error": abs(physicality["trace_real"] - 1.0),
    }


def reverse_cptp_residuals(
    parameters: torch.Tensor,
    generators: list[torch.Tensor],
    system_dimension: int,
    n_steps: int,
) -> tuple[float, float]:
    maximum_completeness = 0.0
    minimum_choi = math.inf
    identity = torch.eye(system_dimension, dtype=toy.CDTYPE)
    for theta in parameter_schedule(parameters, n_steps=n_steps):
        unitary = reverse_collision_unitary(theta, generators)
        kraus = collision_kraus_operators(unitary, system_dimension)
        completeness = sum(operator.conj().T @ operator for operator in kraus)
        maximum_completeness = max(
            maximum_completeness,
            float(torch.linalg.norm(completeness - identity)),
        )
        minimum_choi = min(
            minimum_choi,
            float(torch.linalg.eigvalsh(choi_matrix_from_kraus(kraus)).min()),
        )
    return maximum_completeness, minimum_choi


def add_result(
    rows: list[dict],
    *,
    name: str,
    family: str,
    density: torch.Tensor,
    target: torch.Tensor,
    target_mutual_information: float,
    target_correlation: torch.Tensor,
    operators,
    identity,
    trainable_parameters: int,
    effective_parameters: int,
    training_seconds: float,
    generation_seconds: float,
    sampling_seconds_2000: float | None,
    tensor_bond_dimension: int | None = None,
    max_kraus_completeness_error: float | None = None,
    minimum_choi_eigenvalue: float | None = None,
) -> None:
    rows.append(
        {
            "model": name,
            "family": family,
            "trainable_parameters": trainable_parameters,
            "effective_parameters": effective_parameters,
            "tensor_bond_dimension": tensor_bond_dimension,
            "training_seconds": training_seconds,
            "generation_seconds": generation_seconds,
            "sampling_seconds_2000": sampling_seconds_2000,
            "max_kraus_completeness_error": max_kraus_completeness_error,
            "minimum_choi_eigenvalue": minimum_choi_eigenvalue,
            **density_metrics(
                density,
                target,
                target_mutual_information,
                target_correlation,
                operators,
                identity,
            ),
        }
    )


def plot_comparison(rows: list[dict], path: Path) -> None:
    names = [row["model"] for row in rows]
    positions = np.arange(len(rows))
    panels = [
        ("trace_distance", "Trace distance"),
        ("mutual_information_absolute_error", "Mutual-information error [nats]"),
        ("connected_correlation_error", "Connected-correlation error"),
    ]
    colors = ["#2F5D8A", "#C17C20", "#6B7D3E"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.5))
    for axis, (metric, title), color in zip(axes, panels, colors, strict=True):
        values = [row[metric] for row in rows]
        axis.barh(positions, values, color=color, alpha=0.9)
        axis.set_yticks(positions, labels=names if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlim(left=0)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Two-spin correlated pilot (single seed; lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_classical_arguments(parser)
    parser.add_argument("--j", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=1200)
    parser.add_argument("--sigma", type=float, default=0.18)
    parser.add_argument("--rotation-angle", type=float, default=0.55)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--D", type=float, default=1.0)
    parser.add_argument("--quantum-epochs", type=int, default=80)
    parser.add_argument("--neural-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "two_spin_pilot",
    )
    args = parser.parse_args()
    classical_config = config_from_arguments(args)
    if args.n_train < 1 or args.n_test < 1:
        raise ValueError("n-train and n-test must be positive")
    if args.j != 1.0:
        raise ValueError("this first controlled pilot is validated only at j=1")
    output_root = args.outdir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["outdir"] = str(output_root)
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    train_pairs = make_correlated_direction_pairs(
        args.n_train,
        args.sigma,
        args.rotation_angle,
        args.seed,
    )
    test_pairs = make_correlated_direction_pairs(
        args.n_test,
        args.sigma,
        args.rotation_angle,
        args.seed + 1000,
    )
    train_density = empirical_joint_density(train_pairs, args.j)
    test_density = empirical_joint_density(test_pairs, args.j)
    toy.assert_physical_density(train_density, "two-spin training density")
    toy.assert_physical_density(test_density, "two-spin test density")
    Jx, Jy, Jz, identity = toy.spin_operators(args.j)
    operators = [Jx, Jy, Jz]
    target_mutual_information = quantum_mutual_information(
        test_density, identity.shape[0]
    )
    target_correlation = connected_spin_correlation(
        test_density,
        operators,
        identity,
    )
    rows: list[dict] = []
    states: dict[str, torch.Tensor] = {"target_test": test_density}

    start = time.perf_counter()
    factorized = factorized_density(train_density, identity.shape[0])
    factorized_seconds = time.perf_counter() - start
    sampling_start = time.perf_counter()
    rng = np.random.default_rng(args.seed + 1)
    rng.integers(0, len(train_pairs), size=(2000, 2))
    factorized_sampling = time.perf_counter() - sampling_start
    add_result(
        rows,
        name="Factorized marginals",
        family="independent spherical",
        density=factorized,
        target=test_density,
        target_mutual_information=target_mutual_information,
        target_correlation=target_correlation,
        operators=operators,
        identity=identity,
        trainable_parameters=0,
        effective_parameters=2 * (identity.numel() - 1),
        training_seconds=0.0,
        generation_seconds=factorized_seconds,
        sampling_seconds_2000=factorized_sampling,
    )
    states["factorized"] = factorized

    start = time.perf_counter()
    heat_pairs = sample_riemannian_heat_kernel(
        train_pairs,
        args.n_test,
        bandwidth=0.10,
        seed=args.seed + 2,
    )
    heat_sampling = time.perf_counter() - start
    start = time.perf_counter()
    heat_density = empirical_joint_density(heat_pairs, args.j)
    heat_generation = time.perf_counter() - start
    add_result(
        rows,
        name="Riemannian heat-kernel KDE",
        family="classical spherical diffusion",
        density=heat_density,
        target=test_density,
        target_mutual_information=target_mutual_information,
        target_correlation=target_correlation,
        operators=operators,
        identity=identity,
        trainable_parameters=0,
        effective_parameters=0,
        training_seconds=0.0,
        generation_seconds=heat_generation,
        sampling_seconds_2000=heat_sampling * 2000 / args.n_test,
    )
    states["riemannian_heat_kernel"] = heat_density

    diffusion_pairs, diffusion_metadata = run_classical_benchmark(
        train_pairs, classical_config, args.seed, args.n_test, output_root / "classical")
    start = time.perf_counter()
    diffusion_density = empirical_joint_density(diffusion_pairs, args.j)
    diffusion_encoding = time.perf_counter() - start
    diffusion_parameters = diffusion_metadata["trainable_parameter_count"]
    add_result(
        rows, name="Classical DDPM", family="learned Cartesian diffusion",
        density=diffusion_density, target=test_density,
        target_mutual_information=target_mutual_information,
        target_correlation=target_correlation, operators=operators, identity=identity,
        trainable_parameters=diffusion_parameters, effective_parameters=diffusion_parameters,
        training_seconds=diffusion_metadata["training_seconds"],
        generation_seconds=diffusion_encoding,
        sampling_seconds_2000=diffusion_metadata["sampling_seconds"] * 2000 / args.n_test,
    )
    states["classical_ddpm"] = diffusion_density

    start = time.perf_counter()
    tensor_density = clustered_product_mixture_density(
        train_pairs,
        args.j,
        bond_dimension=2,
        seed=args.seed + 3,
    )
    tensor_training = time.perf_counter() - start
    add_result(
        rows,
        name="Separable tensor mixture (chi=2)",
        family="tensor network",
        density=tensor_density,
        target=test_density,
        target_mutual_information=target_mutual_information,
        target_correlation=target_correlation,
        operators=operators,
        identity=identity,
        trainable_parameters=0,
        effective_parameters=2 * 2 * (identity.numel() - 1) + 1,
        training_seconds=tensor_training,
        generation_seconds=0.0,
        sampling_seconds_2000=None,
        tensor_bond_dimension=2,
    )
    states["tensor_mixture"] = tensor_density

    forward = build_two_spin_forward_trajectory(
        train_density,
        operators,
        identity,
        diffusion_rate=args.D,
        time_step=args.dt,
        steps=args.T,
    )
    prior = torch.eye(identity.shape[0] ** 2, dtype=toy.CDTYPE) / (
        identity.shape[0] ** 2
    )
    quantum_parameter_target = None
    for include_direct, label in [
        (False, "Quantum reverse, no direct J1-J2"),
        (True, "Quantum reverse, direct J1-J2"),
    ]:
        generators, names = build_two_spin_reverse_generators(
            operators,
            identity,
            include_direct_interactions=include_direct,
        )
        start = time.perf_counter()
        parameters, history = toy.train_reverse(
            forward,
            generators,
            layers=1,
            epochs=args.quantum_epochs,
            lr=0.04,
            seed=args.seed,
            objective="fidelity",
        )
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        generated = toy.generate_density(prior, parameters, generators)
        generation_seconds = time.perf_counter() - start
        completeness, minimum_choi = reverse_cptp_residuals(
            parameters,
            generators,
            system_dimension=prior.shape[0],
            n_steps=args.T,
        )
        add_result(
            rows,
            name=label,
            family="quantum collision channel",
            density=generated,
            target=test_density,
            target_mutual_information=target_mutual_information,
            target_correlation=target_correlation,
            operators=operators,
            identity=identity,
            trainable_parameters=parameters.numel(),
            effective_parameters=parameters.numel(),
            training_seconds=training_seconds,
            generation_seconds=generation_seconds,
            sampling_seconds_2000=None,
            max_kraus_completeness_error=completeness,
            minimum_choi_eigenvalue=minimum_choi,
        )
        safe_label = "quantum_direct" if include_direct else "quantum_no_direct"
        states[safe_label] = generated
        torch.save(parameters, output_root / f"{safe_label}_parameters.pt")
        (output_root / f"{safe_label}_training.json").write_text(
            json.dumps(
                {
                    "generator_names": names,
                    "initial_loss": history[0],
                    "final_loss": history[-1],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if include_direct:
            quantum_parameter_target = parameters.numel()

    if quantum_parameter_target is None:
        raise AssertionError("direct quantum model was not trained")
    start = time.perf_counter()
    neural_model, neural_history = train_neural_baseline(
        train_pairs,
        parameter_target=quantum_parameter_target,
        epochs=args.neural_epochs,
        seed=args.seed,
    )
    neural_training = time.perf_counter() - start
    start = time.perf_counter()
    neural_pairs = sample_neural_pairs(
        neural_model,
        args.n_test,
        seed=args.seed + 4,
    )
    neural_sampling = time.perf_counter() - start
    start = time.perf_counter()
    neural_density = empirical_joint_density(neural_pairs, args.j)
    neural_generation = time.perf_counter() - start
    neural_parameters = sum(
        parameter.numel() for parameter in neural_model.parameters()
    )
    add_result(
        rows,
        name="Matched-parameter neural generator",
        family="generic neural",
        density=neural_density,
        target=test_density,
        target_mutual_information=target_mutual_information,
        target_correlation=target_correlation,
        operators=operators,
        identity=identity,
        trainable_parameters=neural_parameters,
        effective_parameters=neural_parameters,
        training_seconds=neural_training,
        generation_seconds=neural_generation,
        sampling_seconds_2000=neural_sampling * 2000 / args.n_test,
    )
    states["neural"] = neural_density
    torch.save(neural_model.state_dict(), output_root / "neural_parameters.pt")
    (output_root / "neural_training.json").write_text(
        json.dumps(
            {
                "initial_loss": neural_history[0],
                "final_loss": neural_history[-1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = list(rows[0])
    with (output_root / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scope": "single-seed two-spin pilot; not an advantage benchmark",
        "target_quantum_mutual_information": target_mutual_information,
        "target_connected_correlation_norm": float(
            torch.linalg.norm(target_correlation)
        ),
        "models": rows,
        "limitations": [
            "Only one train/test seed is evaluated.",
            "The learned DDPM uses Cartesian diffusion with final sphere projection and a separate fixed budget; its parameter count is not matched.",
            "The heat-kernel baseline is a KDE/one-step diffusion baseline, not a learned reverse SDE.",
            "Quantum direction-sampling cost is not yet implemented for the joint Husimi-Q readout.",
            "The shared collision ancilla can mediate correlations even in the no-direct-interaction ablation.",
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez(
        output_root / "density_matrices.npz",
        **{name: state.detach().cpu().numpy() for name, state in states.items()},
    )
    plot_comparison(rows, output_root / "comparison.png")
    print(f"two-spin correlated pilot written to {output_root}")


if __name__ == "__main__":
    main()
