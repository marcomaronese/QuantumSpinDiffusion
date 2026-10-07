from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_has_declared_boundaries():
    required = (
        "src/spin_quantum_diffusion/quantum",
        "src/spin_quantum_diffusion/classical",
        "tasks",
        "tests",
        "outputs/experiments",
        "materials/paper",
        "materials/documents",
        "notes",
    )
    assert all((ROOT / path).is_dir() for path in required)


def test_legacy_flat_source_locations_are_absent():
    legacy = (
        "spin_diffusion_toy.py",
        "spin_classical_diffusion.py",
        "spin_classical_benchmark.py",
        "spin_config.py",
        "spin_correlated.py",
        "spin_losses.py",
        "spin_multipoles.py",
        "spin_reverse.py",
        "experiments",
        "paper",
        "output",
    )
    assert not any((ROOT / path).exists() for path in legacy)


def test_frozen_paper_evidence_is_separate_from_working_outputs():
    assert (ROOT / "materials/paper/evidence/manifest.json").is_file()
    assert (ROOT / "outputs/experiments/README.md").is_file()
