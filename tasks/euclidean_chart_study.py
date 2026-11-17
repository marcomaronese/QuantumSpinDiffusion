"""Screen and independently confirm Euclidean charts through continuous POVMs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
import torch

from spin_quantum_diffusion.classical.synthetic import TARGETS, sample_discrepancies, sample_target
from spin_quantum_diffusion.quantum.coherent_readout import (
    coherent_directions_from_uniforms, coherent_kets,
)
from spin_quantum_diffusion.quantum.euclidean import EuclideanSpinChart, RadialSpinChart
from spin_quantum_diffusion.quantum.single_spin import make_direction_dataset


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "notes/protocols/EUCLIDEAN_CHART_V1.md"
MAPS = [{"name": "bounded_R3", "kind": "bounded", "length": 3.}] + [
    {"name": f"{kind}_s{length:g}", "kind": kind, "length": length}
    for kind in ("stereographic", "sinh_stereographic") for length in (.5, 1., 2.)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def chart_for(description, calibration):
    if description["kind"] == "bounded":
        return EuclideanSpinChart.fit(calibration, radius=description["length"])
    return RadialSpinChart.fit(calibration, description["length"], description["kind"])


def source_snapshot(output):
    evidence = json.loads((ROOT / "materials/paper/evidence/manifest.json").read_text())
    for item in evidence["files"]:
        if digest(ROOT / "materials/paper" / item["path"]) != item["sha256"]:
            raise RuntimeError("frozen evidence mismatch")
    paths = list((ROOT / "src").rglob("*.py")) + [Path(__file__),
        ROOT / "tasks/validate_euclidean_chart.py", ROOT / "tests/test_radial_charts.py", PROTOCOL]
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = digest(path)
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"source_hashes": hashes, "source_fingerprint": fingerprint,
            "frozen_evidence_hashes": len(evidence["files"]), "numpy": np.__version__,
            "torch": torch.__version__, "python": platform.python_version(),
            "created_utc": datetime.now(timezone.utc).isoformat()}


def verify_manifest(directory):
    for name, expected in json.loads((directory / "manifest.json").read_text()).items():
        if digest(directory / name) != expected:
            raise AssertionError(f"changed artifact: {name}")


def metrics(calibration, reference, description, two_j, noise):
    chart = chart_for(description, calibration)
    encoded, clipped = chart.encode(calibration)
    incoming = torch.from_numpy(encoded[noise["indices"]])
    observed = coherent_directions_from_uniforms(incoming, two_j,
        torch.from_numpy(noise["radial"]), torch.from_numpy(noise["azimuth"]))
    decoded = chart.decode(observed.numpy().reshape(-1, 3)).reshape(*noise["indices"].shape, 2)
    standardized = chart.standardize(decoded.reshape(-1, 2)).reshape(decoded.shape)
    real = chart.standardize(reference)
    scores = [sample_discrepancies(real, batch, energy_count=min(1024, len(real))) for batch in standardized]
    radii = np.linalg.norm(standardized.reshape(-1, 2), axis=1)
    real_radii = np.linalg.norm(real, axis=1)
    quantiles = np.quantile(radii, [.5, .95, .995])
    roundtrip = chart.standardize(chart.decode(encoded))-chart.standardize(calibration)
    kets = coherent_kets(torch.from_numpy(encoded), two_j)
    rho = (kets.T @ kets.conj()/len(kets)).numpy()
    residual = max(abs(np.trace(rho)-1), np.max(np.abs(rho-rho.conj().T)),
                   -np.linalg.eigvalsh(rho).min())
    if residual>1e-10:
        raise ValueError("nonphysical encoded state")
    finite_variance = description["kind"] != "stereographic"
    stretch = chart.local_stretches(calibration) if isinstance(chart, RadialSpinChart) else None
    dot = (observed.numpy()*incoming.numpy()).sum(axis=-1)
    result = {"sliced_w1": float(np.mean([score["sliced_w1"] for score in scores])),
        "sliced_w1_batch_sd": float(np.std([score["sliced_w1"] for score in scores], ddof=1)),
        "batch_scores": scores, "energy_v": float(np.mean([score["energy_v"] for score in scores])),
        "radius_quantiles": quantiles.tolist(), "radius_max": float(radii.max()),
        "reference_radius_995": float(np.quantile(real_radii, .995)),
        "tail_ratio_995": float(quantiles[-1]/np.quantile(real_radii, .995)),
        "outlier_mass": float(np.mean(radii>np.quantile(real_radii, .995))),
        "calibration_clipped_fraction": float(clipped.mean()),
        "max_unclipped_roundtrip": float(np.abs(roundtrip[~clipped]).max()),
        "roundtrip_displacement_mean": float(np.linalg.norm(roundtrip, axis=1).mean()),
        "empirical_covariance": np.cov(standardized.reshape(-1, 2), rowvar=False).tolist() if finite_variance else None,
        "population_second_moment_finite": finite_variance,
        "pole_husimi_density": float((two_j+1)/(4*np.pi)*rho[-1, -1].real),
        "mean_angular_error": float(np.arccos(np.clip(dot, -1, 1)).mean()),
        "mean_dot": float(dot.mean()), "mean_dot_squared": float((dot*dot).mean()),
        "angular_stretch_median": np.median(stretch, axis=0).tolist() if stretch is not None else None,
        "state_physicality_residual": float(residual)}
    if result["max_unclipped_roundtrip"]>1e-10:
        raise ValueError("chart round-trip failed")
    return result, decoded, rho


def select(rows, descriptions):
    selections = {}
    for target in dict.fromkeys(row["target"] for row in rows):
        group = [row for row in rows if row["target"] == target and row["two_j"] == 5]
        means = {d["name"]: float(np.mean([r["sliced_w1"] for r in group if r["map"]["name"] == d["name"]])) for d in descriptions}
        chosen = {}
        for kind in ("stereographic", "sinh_stereographic"):
            choices = [d for d in descriptions if d["kind"] == kind]
            if kind == "sinh_stereographic":
                guarded = [d for d in choices if all(r["tail_ratio_995"]<=1.5 for r in group if r["map"]==d)]
                choices = guarded or choices
            chosen[kind] = min(choices, key=lambda d: (means[d["name"]], d["length"]))
        candidate = chosen["sinh_stereographic"]
        guard = all(r["tail_ratio_995"]<=1.5 for r in group if r["map"] == candidate)
        improvement = 1-means[candidate["name"]]/means["bounded_R3"]
        selections[target] = {"selected": [{"name": "bounded_R3", "kind": "bounded", "length": 3.},
                                          chosen["stereographic"], candidate],
            "validation_w1": means, "sinh_validation_improvement": improvement,
            "sinh_tail_guardrail": guard,
            "candidate_eligible": bool(improvement>=.1 and guard)}
    return selections


def angular_control(output):
    samples = np.array(make_direction_dataset(2048, .22, 71873))
    theta, phi = samples.T
    directions = np.c_[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    kets = coherent_kets(torch.from_numpy(directions), 5).numpy()
    rho = kets.T @ kets.conj()/len(kets)
    theta = np.linspace(1e-4, np.pi-1e-4, 65)
    phi = np.arange(128)*2*np.pi/128
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    grid = np.c_[(np.sin(th)*np.cos(ph)).ravel(), (np.sin(th)*np.sin(ph)).ravel(), np.cos(th).ravel()]
    vectors = coherent_kets(torch.from_numpy(grid), 5).numpy()
    q = (6/(4*np.pi)*np.einsum("ni,ij,nj->n", vectors.conj(), rho, vectors).real).reshape(th.shape)
    maxima = np.argwhere((q == maximum_filter(q, size=5, mode=("nearest", "wrap"))) & (q>=.25*q.max()))
    peaks = []
    for index in sorted(maxima, key=lambda ind: -q[tuple(ind)]):
        direction = grid[np.ravel_multi_index(tuple(index), q.shape)]
        if all(np.arccos(np.clip(direction @ p["direction"], -1, 1))>=np.deg2rad(20) for p in peaks):
            peaks.append({"direction": direction.tolist(), "height": float(q[tuple(index)])})
    np.savez_compressed(output / "angular_control.npz", input_directions=directions, rho=rho, grid=grid, q=q)
    write(output / "angular_control.json", {"peaks": peaks, "significant_modes": len(peaks), "two_j": 5})


def plot_results(output, rows, phase):
    targets = list(dict.fromkeys(row["target"] for row in rows))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for ax, target in zip(axes.flat, targets):
        group = [r for r in rows if r["target"] == target and r["two_j"] == 5]
        names = list(dict.fromkeys(r["map"]["name"] for r in group))
        for name in names:
            scores = [r["sliced_w1"] for r in group if r["map"]["name"] == name]
            index = names.index(name)
            ax.bar(index, np.mean(scores), yerr=np.std(scores, ddof=1) if len(scores)>1 else 0,
                   color="#2166ac" if "sinh" in name else ("#ce8b20" if name=="bounded_R3" else "#777777"), capsize=2)
        ax.set_xticks(range(len(names)), [name.replace("_stereographic", "") for name in names], rotation=40, ha="right", fontsize=8)
        ax.set_title(target)
        ax.set_ylabel("Validation SW1" if phase=="screen" else "Held-out SW1")
    for ax in list(axes.flat)[len(targets):]:
        ax.set_visible(False)
    fig.suptitle("Continuous coherent POVM | j=5/2 | mean and SD across realizations")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / "chart_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    if phase == "confirm":
        fig, axes = plt.subplots(len(targets), 4, figsize=(14, 3*len(targets)), squeeze=False)
        for axrow, target in zip(axes, targets):
            group = [r for r in rows if r["target"]==target and r["two_j"]==5 and r["seed"]==rows[0]["seed"]]
            directory = output / group[0]["data_directory"]
            raw = np.load(directory / "data.npz")["reference"]
            chart = EuclideanSpinChart.fit(np.load(directory / "data.npz")["calibration"])
            low, high = chart.location-3*chart.scale, chart.location+3*chart.scale
            values = [raw] + [np.load(output / r["artifact"])["decoded"][0] for r in group]
            labels = ["Raw test"] + [r["map"]["name"] for r in group]
            for ax, value, label in zip(axrow, values, labels):
                ax.scatter(*value[:1000].T, s=2, alpha=.25)
                ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1])
                ax.set_title(target+" | "+label, fontsize=8)
        fig.suptitle("First declared seed | fixed display window; tails included in metrics")
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(output / "decoded_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--screen-dir", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.outdir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "outputs/experiments"):
        raise ValueError("use a fresh outputs/experiments directory")
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    metadata = source_snapshot(output)
    config = {"phase": args.phase, "smoke": args.smoke, "n": 64 if args.smoke else 4096,
        "batches": 2 if args.smoke else 4, "two_js": [5] if args.smoke else [5,10,20],
        "targets": ["ring"] if args.smoke else [name for name, spec in TARGETS.items() if spec["dimensions"]==2],
        "seeds": [991] if args.smoke else [1871,4099,8089]}
    descriptions = [d for d in MAPS if d["length"]==1 or d["kind"]=="bounded"] if args.smoke else MAPS
    selection = None
    if args.phase == "confirm":
        if args.screen_dir is None:
            raise ValueError("confirmation requires a screen")
        screen = args.screen_dir.resolve()
        verify_manifest(screen)
        audit = json.loads((screen / "audit.json").read_text())
        previous = json.loads((screen / "metadata.json").read_text())
        if (not audit["passed"] or previous["source_fingerprint"] != metadata["source_fingerprint"]
                or previous["config"]["smoke"] != args.smoke):
            raise ValueError("screen must be audited and have identical source/smoke identity")
        selection = json.loads((screen / "selection.json").read_text())
        shutil.copy2(screen / "selection.json", output / "selection.json")
        metadata["parent_screen"] = str(screen.relative_to(ROOT))
        metadata["selection_sha256"] = digest(screen / "selection.json")
        metadata["parent_manifest_sha256"] = digest(screen / "manifest.json")
    metadata["config"] = config
    write(output / "metadata.json", metadata)
    rows = []
    for target in config["targets"]:
        for seed in config["seeds"]:
            base = seed+100000*list(TARGETS).index(target)
            directory = output / "data" / target / str(seed)
            directory.mkdir(parents=True)
            if selection is None:
                calibration = sample_target(target, config["n"], base)
                reference = sample_target(target, config["n"], base+10000)
                real_reference = sample_target(target, config["n"], base+20000)
                maps = descriptions
            else:
                calibration = np.load(screen / "data" / target / str(seed) / "data.npz")["calibration"]
                reference = sample_target(target, config["n"], base+50000)
                real_reference = sample_target(target, config["n"], base+60000)
                maps = selection[target]["selected"]
            np.savez_compressed(directory / "data.npz", calibration=calibration, reference=reference,
                                real_reference=real_reference)
            affine = EuclideanSpinChart.fit(calibration)
            floor = sample_discrepancies(affine.standardize(reference), affine.standardize(real_reference))
            for two_j in config["two_js"]:
                rng = np.random.default_rng(base+(30000 if selection is None else 70000)+100*two_j)
                shape = (config["batches"], config["n"])
                noise = {"indices": rng.integers(config["n"], size=shape),
                         "radial": rng.random(shape), "azimuth": rng.random(shape)}
                noise_path = directory / f"noise_j{two_j}.npz"
                np.savez_compressed(noise_path, **noise)
                for description in maps:
                    result, decoded, rho = metrics(calibration, reference, description, two_j, noise)
                    case = output / "cases" / target / str(seed) / f"j{two_j}"
                    case.mkdir(parents=True, exist_ok=True)
                    artifact = case / (description["name"]+".npz")
                    np.savez_compressed(artifact, decoded=decoded, rho=rho)
                    rows.append({"target": target, "seed": seed, "two_j": two_j, "map": description,
                        "artifact": str(artifact.relative_to(output)),
                        "data_directory": str(directory.relative_to(output)),
                        "noise": str(noise_path.relative_to(output)), "real_reference": floor, **result})
            print(f"{args.phase}: {target} seed {seed}; {len(rows)} cases", flush=True)
    write(output / "results.json", rows)
    if selection is None:
        write(output / "selection.json", select(rows, descriptions))
        angular_control(output)
    plot_results(output, rows, args.phase)
    write(output / "manifest.json", {str(path.relative_to(output)): digest(path)
          for path in sorted(output.rglob("*")) if path.is_file()})
    print(f"Saved {len(rows)} cases to {output}", flush=True)


if __name__ == "__main__":
    main()
