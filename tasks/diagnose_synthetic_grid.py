"""Post-hoc exact transport between two saved-benchmark readout grids.

Triggered by binned-TV sensitivity; this adds no fit, sample or selection.
The original study, manifest and predeclared results remain untouched.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import wasserstein_distance

from .validate_synthetic_benchmark import born, povm, sha256


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    study, output = args.study_dir.resolve(), args.outdir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "outputs/experiments"):
        raise ValueError("use a fresh output directory under outputs/experiments")
    manifest = json.loads((study / "manifest.json").read_text())
    for name, expected in manifest.items():
        if sha256(study / name) != expected:
            raise AssertionError(f"changed input artifact: {name}")
    original_audit = json.loads((study / "audit.json").read_text())
    if not original_audit["passed"] or original_audit["manifest_sha256"] != sha256(study / "manifest.json"):
        raise AssertionError("original study must pass its saved audit")
    output.mkdir(parents=True)
    config = json.loads((study / "metadata.json").read_text())["config"]
    coarse_directions, coarse_frame = povm(config["two_j"], *config["grid"])
    fine_directions, fine_frame = povm(config["two_j"], *config["fine_grid"])
    results = []
    for row in json.loads((study / "prepared.json").read_text()):
        directory = study / row["directory"]
        description = json.loads((directory / "chart.json").read_text())
        fit = np.load(directory / "fit.npz")
        evaluation = np.load(directory / "evaluation.npz")
        dimensions = description["dimensions"]
        if dimensions == 1:
            projections = np.ones((1, 1))
        else:
            angles = np.arange(32)*np.pi/32
            projections = np.c_[np.cos(angles), np.sin(angles)]
        # Train-standardized decoder is radius * n[:dimension].
        coarse = description["radius"]*coarse_directions[:, :dimensions]
        fine = description["radius"]*fine_directions[:, :dimensions]
        for name, state in (("model", fit["reverse"][-1]), ("encoded_test", evaluation["rho_test"]),
                            ("classical_common", fit["classical_density"])):
            coarse_probability, fine_probability = born(state, coarse_frame), born(state, fine_frame)
            distance = np.mean([wasserstein_distance(coarse @ axis, fine @ axis,
                                                     coarse_probability, fine_probability)
                                for axis in projections])
            results.append({"target": row["target"], "seed": row["seed"], "condition": name,
                            "coarse_to_fine_weighted_sliced_w1": float(distance)})
    for source in (Path(__file__), ROOT / "tasks/validate_synthetic_benchmark.py"):
        shutil.copy2(source, output / source.name)
    report = {"scope": "post-hoc grid diagnostic prompted by predeclared binned-TV sensitivity",
              "study": str(study.relative_to(ROOT)), "study_manifest_sha256": sha256(study / "manifest.json"),
              "coarse_grid": config["grid"], "fine_grid": config["fine_grid"],
              "new_fits_or_samples": 0, "rows": results,
              "maximum_weighted_sliced_w1": max(row["coarse_to_fine_weighted_sliced_w1"] for row in results)}
    (output / "grid_transport.json").write_text(json.dumps(report, indent=2)+"\n")
    hashes = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
