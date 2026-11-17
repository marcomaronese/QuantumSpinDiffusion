"""Render the validation heatmaps for the directional spin grid."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises_fisher
import torch

from spin_quantum_diffusion.quantum import sphere_plots
from spin_quantum_diffusion.quantum.spin_register import reduced_spin_density


ROOT = Path(__file__).resolve().parents[1]


PANELS = (
    ("raw_generated_tv", "Raw target to model · joint TV"),
    ("raw_encoded_tv", "Raw target to encoding · joint TV"),
    ("encoded_generated_tv", "Encoding to model · joint TV"),
    ("first_pair_connected_error", "First pair · connected error"),
)


def plot_grid(directory, rows, spins_grid, two_j_grid, count, updates):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.3), layout="constrained")
    fig.suptitle("Directional reconstruction across spin counts and local spin sizes",
                 fontsize=16, fontweight="bold")
    bound = max(row[key] for row in rows for key, _ in PANELS if row[key] is not None)
    vmax = max(0.1, np.ceil(bound * 10) / 10)
    cmap = plt.colormaps["Blues"].with_extremes(bad="#eceef0")
    for ax, (key, title) in zip(axes.flat, PANELS, strict=True):
        shape = (len(spins_grid), len(two_j_grid))
        means, spreads = np.full(shape, np.nan), np.full(shape, np.nan)
        for y, spins in enumerate(spins_grid):
            for x, two_j in enumerate(two_j_grid):
                values = [row[key] for row in rows if row["spins"] == spins
                          and row["two_j"] == two_j and row[key] is not None]
                if values:
                    means[y, x] = np.mean(values)
                    if count > 1:
                        spreads[y, x] = np.std(values, ddof=1)
        image = ax.imshow(means, origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for (y, x), value in np.ndenumerate(means):
            label = f"{value:.3f}" if np.isfinite(value) else "—"
            if np.isfinite(spreads[y, x]):
                label += f"\n±{spreads[y, x]:.3f}"
            color = "white" if value / vmax > 0.67 else "#182633"
            ax.text(x, y, label, ha="center", va="center", color=color, fontsize=9)
        ax.set_xticks(range(len(two_j_grid)), [f"{two_j / 2:g}" for two_j in two_j_grid])
        ax.set_yticks(range(len(spins_grid)), spins_grid)
        ax.set(xlabel="Common local spin j", ylabel="Number of spins N")
        ax.set_title(title, loc="left", fontsize=11, pad=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.036, pad=0.025)
    colorbar.set_label("Half L1 error · lower is better")
    detail = (f"{count} paired realizations, mean ± SD" if count > 1 else "Engineering check, one realization")
    fig.supxlabel(f"Validation · {detail} · {updates} updates per cell · four regions per spin",
                  fontsize=10, color="#344554")
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"performance_heatmap.{suffix}", dpi=170)
    plt.close(fig)


def plot_marginal_distributions(directory, theta, phi, fields, label):
    fig = plt.figure(figsize=(12.2, 7.4), layout="constrained")
    grid = fig.add_gridspec(2, 5, width_ratios=(1, 0.045, 1, 1, 0.045))
    raw_norm = matplotlib.colors.Normalize(0, fields[0].max())
    common_norm = matplotlib.colors.Normalize(0, max(fields[1].max(), fields[2].max()))
    columns = (0, 2, 3)
    titles = ("Target grezzo teorico", "Target codificato", "Ricostruita")
    for row, view in enumerate(sphere_plots.OPPOSITE_VIEWS):
        for index, (column, field, title) in enumerate(zip(columns, fields, titles, strict=True)):
            ax = fig.add_subplot(grid[row, column], projection="3d")
            norm = raw_norm if index == 0 else common_norm
            sphere_plots.plot_sphere_heatmap(ax, theta, phi, field, norm=norm,
                                            cmap=sphere_plots.DENSITY_CMAP, view=view)
            ax.set_title(title if row == 0 else "Vista opposta", fontsize=12)
    for column, norm, title in ((1, raw_norm, "Densità grezza [sr⁻¹]"),
                                (4, common_norm, "Densità codificata / ricostruita [sr⁻¹]")):
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=sphere_plots.DENSITY_CMAP),
                     cax=fig.add_subplot(grid[:, column]), label=title)
    fig.suptitle("Distribuzione marginale del primo spin\n" + label, fontsize=14)
    fig.supxlabel("Scale lineari · Scala comune per target codificato e ricostruzione", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"marginal_distributions.{suffix}", dpi=170)
    plt.close(fig)


def plot_joint_distributions(directory, probabilities, label):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.7), layout="constrained")
    maximum = 100 * max(p.max() for p in probabilities)
    titles = ("Target grezzo", "Target codificato", "Ricostruita")
    for row, (p, title) in enumerate(zip(probabilities, titles, strict=True)):
        for third in range(4):
            ax = axes[row, third]
            image = ax.imshow(100 * p[:, :, third], origin="lower", vmin=0, vmax=maximum,
                              cmap="Blues", aspect="equal")
            ax.set_xticks(range(4), range(1, 5))
            ax.set_yticks(range(4), range(1, 5))
            ax.tick_params(length=0)
            if row == 0:
                ax.set_title(f"Regione spin 3 = {third + 1}", fontsize=11)
            if row == 2:
                ax.set_xlabel("Regione spin 2")
            if third == 0:
                ax.set_ylabel(title + "\nRegione spin 1", fontsize=11)
            for (first, second), value in np.ndenumerate(100 * p[:, :, third]):
                ax.text(second, first, f"{value:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if value / maximum > 0.67 else "#182633")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015,
                 label="Probabilità congiunta p(R₁, R₂, R₃) [%]")
    fig.suptitle("Distribuzione congiunta sulle quattro regioni per spin\n" + label, fontsize=14)
    fig.supxlabel("Ogni riga contiene tutti i 64 esiti congiunti · Scala comune · Validazione", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"joint_distributions.{suffix}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize saved spin-grid target and reconstructed distributions")
    parser.add_argument("--study-dir", type=Path,
                        default=ROOT / "outputs/experiments/spin_grid_v1_20261111")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--two-j", type=int)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    protocol_path = study / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    seed = args.seed if args.seed is not None else protocol["seeds"][0]
    two_j = args.two_j if args.two_j is not None else max(protocol["two_j"])
    if seed not in protocol["seeds"] or two_j not in protocol["two_j"] or 3 not in protocol["spins"]:
        parser.error("choose a saved seed and local spin from a study containing three-spin runs")
    cell = study / f"seed_{seed}/n3_two_j{two_j}"
    inputs = [protocol_path, study / f"seed_{seed}/splits.npz", cell / "states.npz",
              cell / "readout.npz", cell / "metrics.json"]
    manifest_path = study / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {}
    for path in inputs:
        relative = str(path.relative_to(study))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest.get(relative) != digest:
            raise ValueError(f"saved artifact checksum mismatch: {relative}")
        hashes[relative] = digest
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (args.outdir or ROOT / f"outputs/experiments/spin_grid_distributions_{timestamp}").resolve()
    if not directory.is_relative_to(ROOT / "outputs/experiments"):
        parser.error("--outdir must stay under outputs/experiments")
    directory.mkdir(parents=True, exist_ok=False)
    with np.load(inputs[1]) as splits:
        centers = splits["centers"][0]
    with np.load(inputs[2]) as states:
        encoded_state = torch.from_numpy(states["validation"])
        generated_state = torch.from_numpy(states["reverse"][-1])
        marginals = [reduced_spin_density(state, two_j + 1, 3, 0).numpy()
                     for state in (encoded_state, generated_state)]
    with np.load(inputs[3]) as readout:
        probabilities = [readout[key] for key in ("raw_validation", "encoded", "generated")]
    theta, phi = sphere_plots.sphere_grid(73, 144)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    directions = np.stack((np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)), axis=-1)
    dataset = protocol["dataset"]
    weights = (dataset["shared_weight"] * np.array(dataset["shared_modes"])
               + (1 - dataset["shared_weight"]) / len(centers))

    def raw_density(points):
        return sum(weight * vonmises_fisher.pdf(points, center, kappa)
                   for weight, center, kappa in zip(weights, centers, dataset["von_mises_fisher_kappa"], strict=True))

    fields = [raw_density(directions)]
    fields.extend(sphere_plots.husimi_density(state, two_j / 2, theta, phi) for state in marginals)
    nodes, weights_gl = np.polynomial.legendre.leggauss(80)
    theta_gl = np.arccos(nodes[::-1])
    tg, pg = np.meshgrid(theta_gl, phi, indexing="ij")
    grid_directions = np.stack((np.sin(tg) * np.cos(pg), np.sin(tg) * np.sin(pg), np.cos(tg)), axis=-1)
    integrands = [raw_density(grid_directions)]
    integrands.extend(sphere_plots.husimi_density(state, two_j / 2, theta_gl, phi) for state in marginals)
    integrals = [float(np.sum(field * weights_gl[::-1, None]) * 2 * np.pi / len(phi)) for field in integrands]
    if max(abs(value - 1) for value in integrals) > 1e-10:
        raise RuntimeError(f"marginal normalization failed: {integrals}")
    if any(p.shape != (4, 4, 4) or p.min() < -1e-10 or abs(p.sum() - 1) > 1e-10
           for p in probabilities):
        raise RuntimeError("invalid joint readout probabilities")
    metrics = json.loads(inputs[4].read_text())
    replay_error = abs(float(np.abs(probabilities[0] - probabilities[2]).sum() / 2)
                       - metrics["raw_generated_tv"])
    if replay_error > 1e-12:
        raise RuntimeError("joint figure disagrees with the saved TV metric")
    label = f"N = 3 · j = {two_j / 2:g} · seed = {seed} · {metrics['updates']} aggiornamenti"
    plot_marginal_distributions(directory, theta, phi, fields, label)
    plot_joint_distributions(directory, probabilities, label)
    np.savez_compressed(directory / "fields.npz", theta=theta, phi=phi, raw_marginal=fields[0],
                        encoded_marginal=fields[1], generated_marginal=fields[2],
                        raw_joint=probabilities[0], encoded_joint=probabilities[1], generated_joint=probabilities[2])
    sources = [Path(__file__).resolve(), Path(sphere_plots.__file__).resolve(),
               ROOT / "src/spin_quantum_diffusion/quantum/spin_register.py"]
    provenance = {
        "created_utc": timestamp, "study": str(study), "seed": seed, "spins": 3, "two_j": two_j,
        "choice": "first protocol seed and largest saved j unless explicitly overridden",
        "input_sha256": hashes, "parent_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "marginals": "first site; raw theoretical mixture, encoded validation and generated continuous Husimi densities",
        "marginal_normalization_integrals": integrals, "marginal_grid": [len(theta), len(phi)],
        "marginal_color_scales": "raw linear scale; common linear scale for encoded and generated",
        "joint": "saved unconditional product POVM region probabilities with all 64 outcomes on a common percentage scale",
        "metric_replay_error": replay_error, "raw_validation_count": protocol["n_validation"],
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()},
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
