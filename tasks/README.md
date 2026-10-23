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

The next E5 step is the truly independent local-channel control. It needs a
new frozen protocol and fresh selection/test identities. The original
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
