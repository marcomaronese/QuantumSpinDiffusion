# SpinDiffusion

Spin phase-space diffusion for directional data: coherent-state encoding,
isotropic quantum diffusion, learned physical reverse channels, and classical
comparisons. This checkout contains the code and evidence needed to develop
the paper under one research plan.

## Start here

| Reference | Purpose |
|---|---|
| [Paper plan](notes/SPINDIFFUSION_PAPER_PLAN.md) | Current position, next action, E0–E6, claim themes and decision gates |
| [Claim ledger](materials/paper/CLAIMS.md) | Statements supported by the paper's audited evidence |
| [Manuscript](materials/paper/manuscript.pdf) · [sources](materials/paper/main.tex) | Current article |
| [Experiment index](tasks/README.md) | Study and audit entry points |

Use **`codex/paper`** for ongoing work. **`main`** is the consolidated
development baseline. [Branch policy and recovery](notes/BRANCHES.md) explain
the two branches and archived historical references.

## Layout

```text
src/spin_quantum_diffusion/
  quantum/       Reusable spin, channel, loss and training implementations
  classical/     Required classical baselines
tasks/           Experiments and independent audits
tests/           Physicality, regression and workflow checks
materials/paper/ Manuscript, claim ledger and immutable research evidence
notes/           One active plan, protocols and scientific reports
outputs/         Local runs and scratch files, ignored by Git
```

## Setup and checks

Python 3.11 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Dependencies are defined once in `pyproject.toml`. Run studies as modules from
the repository root, under a declared protocol and a fresh output directory.
See [the experiment index](tasks/README.md) before starting a study; do not
repeat completed experiments merely because an entry point exists.

## Paper and current research

The manuscript currently supports a bounded construction and diagnostic study.
The latest two-spin supervision screen and its held-out confirmation are
documented in [the research reports](notes/README.md); they are working results,
not yet additions to the manuscript's frozen evidence. The current next step
is to freeze the E5 local-pretraining/controlled-correlation protocol after the
independent-local comparison's audited no-go, as recorded in the plan.

Rebuild/audit the manuscript and generate distribution packages with:

```bash
make -C materials/paper PYTHON=../../.venv/bin/python all
```

ZIP distributions and figure previews are generated locally rather than
versioned as duplicate copies. Keep all new runs under `outputs/experiments/`.
Never edit `materials/paper/evidence/raw/` or `evidence/source/`; new paper
evidence follows [the results workflow](materials/paper/RESULTS_WORKFLOW.md).
