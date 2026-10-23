"""Render working F6 diagnostics only after an independent study audit."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tasks.two_spin_local_control import MODELS, BASELINES, ROOT, sha256, write_json
from tasks.validate_two_spin_local_control import validate_local_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "outputs/experiments/two_spin_local_control_v1")
    args = parser.parse_args()
    root = args.study_dir.resolve()
    if not root.is_relative_to(ROOT / "outputs/experiments"):
        raise ValueError("working figures must stay under outputs/experiments")
    audit = validate_local_control(root)
    protocol = json.loads((root / "protocol.json").read_text())
    split = "test" if protocol["test_generated"] else "validation"
    rows = json.loads((root / "results.json").read_text())
    models = [m for m in (*MODELS, *BASELINES) if any(r["model"] == m for r in rows)]
    labels = {"independent_local": "Independent local\n48 parameters, 2 ancillas",
              "shared_no_direct": "Shared ancilla\n33 parameters",
              "shared_direct": "Shared + direct\n39 parameters",
              "factorized": "Empirical\nfactorized", "tensor_mixture": "Tensor mixture",
              "heat_kernel": "Heat bootstrap", "neural": "Neural", "classical_ddpm": "Ambient DDPM"}
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    seeds = sorted({r["seed"] for r in rows})
    lookup = {(r["seed"], r["model"]): r for r in rows}
    for ax, key, ylabel in zip(axes,
        ("trace_distance", "connected_correlation_error", "marginal_trace_distance_mean"),
        ("Trace distance", "Connected-correlation error", "Mean marginal trace distance"), strict=True):
        quantum = [m for m in models if m in MODELS]
        for seed in seeds:
            ax.plot([models.index(m) for m in quantum],
                    [lookup[seed,m][split][key] for m in quantum], color="0.75", lw=.8, zorder=1)
        for x, model in enumerate(models):
            values = [lookup[s,model][split][key] for s in seeds]
            color = "#20689d" if model in MODELS else "#82714a"
            ax.scatter(x + np.linspace(-.08,.08,len(seeds)), values, s=20, color=color, alpha=.8, zorder=2)
            ax.scatter(x, np.mean(values), s=65, marker="_", color="black", linewidths=2, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    axes[-1].set_xticks(range(len(models)), [labels[m] for m in models], rotation=20, ha="right", fontsize=9)
    figure.suptitle(f"Working E5 {split}: {len(seeds)} paired training realizations, one data family", fontsize=13)
    figure.text(.5,.02,"Points: individual realizations; black marks: means. Unequal resources; exact quantum versus sampled classical states.\n"
                "No physical-POVM sampling or quantum-advantage claim. Not yet promoted to paper evidence.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0,.08,1,.96))
    destination = root / "figures"
    destination.mkdir(exist_ok=False)
    figure.savefig(destination / "local_control.pdf")
    figure.savefig(destination / "local_control.png", dpi=160)
    plt.close(figure)
    write_json(destination / "provenance.json", {
        "audit": audit, "split": split, "paper_evidence_promoted": False,
        "inputs": {name: sha256(root/name) for name in ("protocol.json", "results.json", "artifact_manifest.json")},
        "plot_source_sha256": sha256(__file__),
        "outputs": {p.name: sha256(p) for p in sorted(destination.glob("local_control.*"))}})
    print(destination / "local_control.pdf")


if __name__ == "__main__":
    main()
