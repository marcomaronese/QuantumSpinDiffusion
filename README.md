# Spin phase-space quantum diffusion

This repository studies a finite-dimensional generative construction for
directional data on the sphere. Classical directions are encoded as spin
coherent-state density matrices, evolved through isotropic quantum diffusion,
and reconstructed with learned collision-model channels. A classical DDPM is
included as a required comparison.

The repository is organized so that reusable implementations, executable
studies, tests, generated outputs, research materials, and project history have
separate homes.

## Repository map

```text
src/spin_quantum_diffusion/
  quantum/                  Spin encoding, forward/reverse channels, losses,
                            multipoles, and correlated-spin utilities
  classical/                Classical DDPM and benchmark/report helpers
tasks/                      Reproducible experiment and validation entry points
tests/                      Unit, physicality, regression, and CLI tests
outputs/
  experiments/              Generated run artifacts; ignored by Git
  tmp/                      Disposable intermediate files; ignored by Git
materials/
  paper/                    LaTeX manuscript and immutable audited evidence
  documents/                Other project documents
notes/                      Agent-facing history, reports, and next steps
```

Start with [notes/NEXT_STEPS.md](notes/NEXT_STEPS.md) for the active research
plan and [materials/paper/CLAIMS.md](materials/paper/CLAIMS.md) for the exact
claim-to-evidence ledger. Historical notes may mention the former flat layout;
the paths in this README are authoritative for current work.

## Set up and test

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

The package uses a `src/` layout. For a one-off run without installing it, use
`PYTHONPATH=src` from the repository root.

## Run the single-spin model

```bash
.venv/bin/spin-diffusion --epochs 700
```

The equivalent module entry point is:

```bash
.venv/bin/python -m spin_quantum_diffusion.quantum.single_spin --epochs 700
```

The default output directory is `outputs/experiments/single_spin/`. A fast
sanity run can use `--epochs 100`. Every current run also trains a classical
DDPM on the same directions and writes its artifacts under `classical/` inside
the run directory.

The quantum training objective can be selected with `--objective`; supported
values are `frobenius`, `multipole`, `fidelity`, `hybrid`, `trace-distance`, and
`husimi-js`. Reverse-channel controls include `--generator-set`,
`--parameter-sharing`, and `--ancilla-count`.

The classical implementation is an ambient-space DDPM over Cartesian unit
vectors. Joint two-spin samples use six coordinates so the model can learn
cross-spin correlations. It is a comparison baseline, not an intrinsic
Riemannian diffusion model, and its resource budget is not matched to the
quantum model.

## Run research tasks

All tasks are modules and write new artifacts below `outputs/experiments/` by
default:

```bash
.venv/bin/python -m tasks.validation_study
.venv/bin/python -m tasks.j25_fidelity_study
.venv/bin/python -m tasks.reverse_diagnostic --workers 8
.venv/bin/python -m tasks.hybrid_objective_study --workers 8
.venv/bin/python -m tasks.instrument_study
.venv/bin/python -m tasks.two_spin_correlated
.venv/bin/python -m tasks.validate_two_spin
```

The tasks preserve configurations, metrics, trained parameters, source
fingerprints, and relevant intermediate states. Cached historical runs are not
silently upgraded when the source or required baseline changes; use a new output
directory for a new protocol.

## Current conclusions

The audited evidence currently supports these bounded conclusions:

- Spin-coherent encoding transfers spherical heat evolution to the isotropic
  spin channel, with finite-rank information loss.
- For the declared seed-7 dataset and grid criterion, two significant encoded
  modes first appear at `j=2.5` in the tested representation sweep.
- Forward multipole decay, state physicality, and reverse-channel completeness
  agree with theory to numerical precision in the audited small simulations.
- At 100 epochs, final-only supervision recovered both modes in 5/5 seeds,
  versus 1/5 for path supervision, with either tested initialization. At 500
  epochs, all four diagnostic conditions recovered both modes.
- Validation selected hybrid weight `lambda=0.75`; on five confirmation seeds
  it reduced mean top-rank squared error by 80.3% and trace distance by 51.1%,
  though trace distance did not improve for every seed.
- The sampled ancilla instrument agrees with the deterministic collision
  channel in ensemble averages for the archived aggregate study.
- The present two-spin quantum reverse models underperform the tested
  correlation-aware classical baselines in the one audited pilot.

These results do not establish quantum advantage, hardware efficiency,
diffusion necessity, lossless recovery of arbitrary spherical densities, or a
general minimum spin resolution. The single-spin studies also do not yet supply
independent train/test evaluation. See the claim ledger for evidence paths and
the precise limits of each statement.

## Paper and frozen evidence

The paper project is in `materials/paper/`. Its `evidence/raw/` and
`evidence/source/` trees are immutable snapshots whose hashes are recorded in
`evidence/manifest.json`; do not edit or replace them. Generated experiment
outputs under `outputs/` are working data and are separate from this frozen
evidence.

Rebuild and audit the manuscript from the repository root with:

```bash
make -C materials/paper PYTHON=../../.venv/bin/python all
```

This regenerates the audit, figures, tables, PDF, and distribution archives
without retraining the models. Author metadata, venue selection, broader
novelty review, and stronger multi-split experiments remain outstanding.

## Output discipline

Do not commit routine generated runs. Place them under
`outputs/experiments/<study-name>/`, declare the protocol before running, and
retain enough state to audit every reported metric. Evidence promoted into the
paper must be copied into a new named immutable snapshot with checksums; the
existing snapshot must remain unchanged.
