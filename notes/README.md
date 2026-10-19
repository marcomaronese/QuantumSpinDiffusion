# Project notes for agents

This directory holds project history, audits, decisions, and planned work. The
files are documentation; the runnable implementation lives under `src/` and
`tasks/`.

Read in this order:

1. [`../README.md`](../README.md) — current codebase, commands, and conclusions.
2. [`TODO.md`](TODO.md) — single authoritative development plan and restart checkpoint.
3. [`DIAGNOSTIC_REPORT.md`](DIAGNOSTIC_REPORT.md) — controlled missing-mode and
   hybrid-objective findings.
4. [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) — earlier validation record.
5. [`CODEX_HANDOFF.md`](CODEX_HANDOFF.md) — detailed historical handoff.
6. [`CODEX_PROMPT.txt`](CODEX_PROMPT.txt) — archived continuation prompt.

Update `TODO.md` after development with completed items, evidence paths,
decisions, blockers, and the next action. Do not maintain competing task lists.
`NEXT_STEPS.md` is a compatibility pointer with the old-to-new roadmap mapping.

`CODEX_HANDOFF.md`, `VALIDATION_REPORT.md`, and `CODEX_PROMPT.txt` preserve the
state and paths that existed when they were written. They may refer to the
former flat repository layout. Do not treat those old paths as current
instructions; use the root README and this index for present locations.

Scientific claims must be checked against
[`../materials/paper/CLAIMS.md`](../materials/paper/CLAIMS.md), whose scope and
limitations are authoritative for the frozen evidence.
