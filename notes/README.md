# Project notes for agents

This directory holds project history, audits, decisions, and planned work. The
files are documentation; the runnable implementation lives under `src/` and
`tasks/`.

Read in this order:

1. [`../README.md`](../README.md) — current codebase, commands, and conclusions.
2. [`SPINDIFFUSION_PAPER_PLAN.md`](SPINDIFFUSION_PAPER_PLAN.md) — single authoritative research plan and restart checkpoint.
3. [`DIAGNOSTIC_REPORT.md`](DIAGNOSTIC_REPORT.md) — controlled missing-mode and
   hybrid-objective findings.
4. [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) — earlier validation record.
5. [`CODEX_HANDOFF.md`](CODEX_HANDOFF.md) — detailed historical handoff.
6. [`CODEX_PROMPT.txt`](CODEX_PROMPT.txt) — archived continuation prompt.

Update `SPINDIFFUSION_PAPER_PLAN.md` after development with completed items,
evidence paths, decisions, blockers, and the next action. `TODO.md` and
`NEXT_STEPS.md` are compatibility pointers; do not maintain competing lists.

`CODEX_HANDOFF.md`, `VALIDATION_REPORT.md`, and `CODEX_PROMPT.txt` preserve the
state and paths that existed when they were written. They may refer to the
former flat repository layout. Do not treat those old paths as current
instructions; use the root README and this index for present locations.

Scientific claims must be checked against
[`../materials/paper/CLAIMS.md`](../materials/paper/CLAIMS.md), whose scope and
limitations are authoritative for the frozen evidence.

Current research checkpoint: the first fixed-architecture E5 supervision
screen and its independent-observation confirmation are recorded in
[`TWO_SPIN_SUPERVISION_REPORT.md`](TWO_SPIN_SUPERVISION_REPORT.md). Article
alignment with the plan is assessed in
[`materials/paper/PLAN_ALIGNMENT.md`](../materials/paper/PLAN_ALIGNMENT.md).
Both are reports; update the authoritative plan for all active priorities.
