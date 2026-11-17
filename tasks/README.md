# Experiment and audit entry points

Read the [official plan](../notes/SPINDIFFUSION_PAPER_PLAN.md) for the next
experiment. These modules are retained because they implement or audit the
paper's construction, baselines, historical results, or current E5 work.
Their presence does not schedule another run.

## Current E5 work

| Module | Role |
|---|---|
| `two_spin_supervision` | Completed fixed-architecture four-condition screen; protocol/source snapshots and fixed classical baselines |
| `confirm_two_spin_supervision` | Separate confirmation of the frozen original final-versus-path choice; tied to the recorded parent artifact hashes |
| `validate_two_spin_supervision` | Independent replay of saved screen and confirmation states |
| `two_spin_local_control` | Frozen three-condition independent-local/shared-ancilla diagnosis; final-only supervision and fresh splits |
| `validate_two_spin_local_control` | Independent NumPy/SciPy state, channel, factorization and decision replay |
| `plot_two_spin_local_control` | Audited working F6 diagnostics; local outputs with plot provenance |

The current E5 control uses the [local-channel protocol](../notes/protocols/TWO_SPIN_LOCAL_CONTROL_V1.md)
with fresh selection/test identities. Its [audited no-go report](../notes/TWO_SPIN_LOCAL_CONTROL_REPORT.md)
records the completed validation and the still-closed confirmation. The original
confirmation observations are consumed and must not be reused for tuning.

Audit the existing working results from the repository root:

```bash
.venv/bin/python -m tasks.validate_two_spin_supervision
.venv/bin/python -m tasks.validate_two_spin_supervision --confirmation \
  --study-dir outputs/experiments/two_spin_supervision_confirmation_v1
```

The local outputs must exist; they are not part of a fresh clone. The
[supervision report](../notes/TWO_SPIN_SUPERVISION_REPORT.md) records their
hashes, numerical findings and limits.

## Historical studies and supporting controls

| Module | Paper or plan role |
|---|---|
| `validation_study` | E2 representation and objective sweeps |
| `j25_fidelity_study` | E3 five-seed mode-recovery diagnostic at j=2.5 |
| `reverse_diagnostic` | E3 initialization, supervision and budget grid |
| `hybrid_objective_study` | E3 validation-selected weighting and confirmation |
| `instrument_study` | E6 instrument ensemble consistency |
| `two_spin_correlated` | E5 correlated pilot and classical baselines |
| `validate_two_spin` | Independent audit of saved correlated-pilot metrics |

Invoke studies as `.venv/bin/python -m tasks.<module>` from the repository
root, using their declared protocol. New source/protocol combinations require
a fresh `outputs/experiments/<study-id>/` directory. Historical frozen evidence
is replayed with the source captured alongside it; do not insert later DDPM
code or new metrics into historical conclusions.

## Bloch-direction sphere heatmaps (saved states only)

Render real spherical harmonics and a target/reconstruction/residual comparison
without retraining or generating new observations:

```bash
MPLCONFIGDIR=/tmp/spin-matplotlib .venv/bin/python -m tasks.plot_bloch_heatmaps
```

The default illustration uses the frozen single-spin diagnostic run
`mixed_final_e500_fidelity_l0.0/seed_7` (j=2.5), with generation from `I/d`.
Choose another saved single-spin run using `--run-dir`; `--ell-max` sets the
harmonic atlas range (default 3). `--reconstructed-state` selects a saved state
filename, default `rho_generated_mixed.npy`. Output goes into a fresh timestamped
directory under `outputs/experiments/`, or a new `--output-dir` there.

The PNG/PDF figures include opposite views, a common density color scale and a
separate symmetric residual scale. `fields.npz` saves the plotted values and
`provenance.json` records input/source hashes, frozen-input verification and
rendering conventions. The target is the Husimi-Q density of the encoded
empirical state, not the raw classical source density. Colors show density per
solid angle, without the `sin(theta)` sampling weight. Real harmonics are signed
orthonormal basis functions, not fitted coefficients or probabilities. For
j>1/2 the sphere labels spin-coherent directions rather than the full state
space. These are working visualizations, not newly promoted paper evidence.

Future single-spin CLI runs also save `bloch_comparison.png` alongside their
existing planar maps. Replot existing runs with this task to avoid retraining.

## Directional reconstruction across spin counts and local j

Run the [exploratory spin grid](../notes/protocols/SPIN_GRID_V1.md) from the
repository root:

```bash
MPLCONFIGDIR=/tmp/spin-grid-matplotlib .venv/bin/python -m tasks.spin_grid
```

The grid uses the first one, two or three directions of a common multimodal
target and a common local `j` per cell, from `1/2` to `5/2`. It reports joint
region TV from raw validation to model, raw validation to encoding, and
encoding to model, plus the connected error on the first pair. `metrics.csv`
also includes the third-order diagnostic, the encoded factorized reference,
the split and grid sensitivity checks, trace distances, and measured costs.
The `performance_heatmap` PNG/PDF, samples, states, parameters, trajectories,
and provenance go to a new directory under `outputs/experiments/`. Use
`--outdir` to set the run identity or `--smoke` for an engineering check.

Display the saved three-spin target and reconstruction with:

```bash
MPLCONFIGDIR=/tmp/spin-grid-matplotlib .venv/bin/python -m tasks.plot_spin_grid
```

The distribution renderer defaults to the first study seed and largest local
spin. `--study-dir`, `--seed` and `--two-j` select another saved run. The
first-site figure shows the theoretical raw marginal and continuous Husimi
densities for the validation encoding and generated state. The raw marginal
has its own linear color scale, and the two quantum readouts share a scale.
The joint figure shows all 64 region probabilities in four slices of the
third spin, using the saved unconditional Born probabilities. Input hashes,
normalization checks and exact plotted fields accompany the PNG/PDF figures
in a fresh output directory, configurable with `--outdir`.
