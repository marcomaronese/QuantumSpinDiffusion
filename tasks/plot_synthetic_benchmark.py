"""Render audited synthetic observations without changing the sealed run."""

import argparse
import json
from pathlib import Path
import shutil

from .synthetic_benchmark import ROOT, digest, plots, write_json


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
        if digest(study / name) != expected:
            raise AssertionError(f"changed input artifact: {name}")
    audit = json.loads((study / "audit.json").read_text())
    if not audit["passed"] or audit["manifest_sha256"] != digest(study / "manifest.json"):
        raise AssertionError("study must pass its saved audit")
    output.mkdir(parents=True)
    plots(study, json.loads((study / "prepared.json").read_text()), destination=output)
    for source in (Path(__file__), ROOT / "tasks/synthetic_benchmark.py"):
        shutil.copy2(source, output / source.name)
    write_json(output / "rendering.json", {"study": str(study.relative_to(ROOT)),
        "study_manifest_sha256": digest(study / "manifest.json"),
        "sampling": "saved samples only; first declared seed, no new draws",
        "changes": "title and axis-label spacing; unchanged observations and metrics"})
    write_json(output / "manifest.json", {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    print(output)


if __name__ == "__main__":
    main()
