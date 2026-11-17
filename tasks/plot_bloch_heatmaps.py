"""Render spherical harmonics and saved single-spin Husimi states, without training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from spin_quantum_diffusion.quantum import sphere_plots


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "materials/paper"
DEFAULT_RUN = PAPER / "evidence/raw/diagnostic_study/runs/mixed_final_e500_fidelity_l0.0/seed_7"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_inputs(paths):
    """Check frozen inputs against their existing manifest; never modify it."""
    frozen = [path for path in paths if path.is_relative_to(PAPER / "evidence/raw")]
    if not frozen:
        return {}
    manifest = PAPER / "evidence/manifest.json"
    entries = {entry["path"]: entry["sha256"] for entry in json.loads(manifest.read_text())["files"]}
    for path in frozen:
        key = path.relative_to(PAPER).as_posix()
        if key not in entries or sha256(path) != entries[key]:
            raise ValueError(f"frozen evidence hash mismatch or missing entry: {key}")
    return {"manifest_sha256": sha256(manifest), "verified_input_count": len(frozen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                        help="Single-spin run containing config.json and saved density matrices")
    parser.add_argument("--reconstructed-state", default="rho_generated_mixed.npy",
                        help="Saved state filename; default uses generation from I/d")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New directory under outputs/experiments; never overwritten")
    parser.add_argument("--ell-max", type=int, default=3)
    parser.add_argument("--n-theta", type=int, default=49)
    parser.add_argument("--n-phi", type=int, default=96)
    args = parser.parse_args()
    if args.ell_max < 0:
        parser.error("--ell-max must be nonnegative")
    theta, phi = sphere_plots.sphere_grid(args.n_theta, args.n_phi)
    run = args.run_dir.resolve()
    inputs = [run / "config.json", run / "rho_data.npy", (run / args.reconstructed_state).resolve()]
    if not inputs[2].is_relative_to(run):
        parser.error("--reconstructed-state must be inside the selected run")
    evidence_check = verify_frozen_inputs(inputs)
    config = json.loads(inputs[0].read_text())
    j = config["j"]
    target_state, model_state = (np.load(path, allow_pickle=False) for path in inputs[1:])
    target = sphere_plots.husimi_density(target_state, j, theta, phi)
    model = sphere_plots.husimi_density(model_state, j, theta, phi)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (args.output_dir or ROOT / f"outputs/experiments/bloch_heatmaps_{timestamp}").resolve()
    if not destination.is_relative_to(ROOT / "outputs/experiments"):
        parser.error("--output-dir must stay under outputs/experiments")
    destination.mkdir(parents=True, exist_ok=False)
    title = f"Target and reconstruction | j = {j:g}, seed = {config.get('seed', 'unspecified')}"
    title += f", epochs = {config.get('epochs', 'unspecified')}"
    figure = sphere_plots.plot_husimi_comparison(theta, phi, target, model, title=title)
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"husimi_comparison.{suffix}", dpi=180)
    plt.close(figure)
    for label, view in zip(("front", "back"), sphere_plots.OPPOSITE_VIEWS, strict=True):
        figure = sphere_plots.plot_harmonic_atlas(theta, phi, args.ell_max, view=view)
        for suffix in ("png", "pdf"):
            figure.savefig(destination / f"harmonics_{label}.{suffix}", dpi=160)
        plt.close(figure)
    keys = [(ell, m) for ell in range(args.ell_max + 1) for m in range(-ell, ell + 1)]
    np.savez_compressed(destination / "fields.npz", theta=theta, phi=phi,
                        target_q=target, reconstructed_q=model, difference_q=model - target,
                        harmonic_keys=keys, real_harmonics=np.stack([
                            sphere_plots.real_spherical_harmonic(ell, m, theta, phi)
                            for ell, m in keys]))
    # Record exact fields and rendering provenance, without creating new study evidence.
    source_files = [Path(__file__).resolve(), Path(sphere_plots.__file__).resolve()]
    provenance = {
        "created_utc": timestamp,
        "operation": "Rendering of saved states; no training, selection, or new observations",
        "paper_evidence_promoted": False,
        "run_directory": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "inputs": {str(path): sha256(path) for path in inputs},
        "frozen_evidence_check": evidence_check,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "config": config,
        "grid": {"n_theta": args.n_theta, "n_phi": args.n_phi, "poles_included": True,
                 "phi_endpoint_included": False, "rendering_seam_closed": True},
        "readout": "Q(Omega) = (2j+1)/(4pi) <Omega|rho|Omega>, density per solid angle",
        "target": "Husimi readout of the empirical encoded target, not the raw source density",
        "normalization": "No state repair, sin(theta) factor, or grid renormalization",
        "sphere": "Unit coherent-direction sphere; for j>1/2 not a full quantum-state Bloch sphere",
        "views_elevation_azimuth_degrees": sphere_plots.OPPOSITE_VIEWS,
        "harmonics": {"ell_max": args.ell_max, "spin_accessible_ell_max": round(2 * j),
                      "convention": "Real orthonormal: m<0 sine, m>0 cosine; R_11 = sqrt(3/4pi) x",
                      "meaning": "Basis functions only, not fitted coefficients or multipole powers"},
        "density_color_limits": [0, float(max(target.max(), model.max()))],
        "residual_color_limits": [-max(float(np.abs(model - target).max()), np.finfo(float).eps),
                                  max(float(np.abs(model - target).max()), np.finfo(float).eps)],
        "outputs": {path.name: sha256(path) for path in sorted(destination.iterdir())},
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
