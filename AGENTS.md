# Instructions for agents

Read `README.md`, then `notes/README.md` and `notes/NEXT_STEPS.md` before making
research changes. Use `materials/paper/CLAIMS.md` when describing conclusions.

Keep the repository boundaries intact:

- reusable quantum code belongs in `src/spin_quantum_diffusion/quantum/`;
- reusable classical code belongs in `src/spin_quantum_diffusion/classical/`;
- experiment orchestration belongs in `tasks/`;
- verification belongs in `tests/`;
- generated runs belong in `outputs/experiments/`;
- papers and other documents belong in `materials/`;
- project history and future plans belong in `notes/`.

Do not edit `materials/paper/evidence/raw/` or
`materials/paper/evidence/source/`. They are immutable, checksummed research
evidence. New paper evidence requires a separate named snapshot and an updated
claim ledger.

Run `.venv/bin/python -m pytest -q` after code changes. Invoke studies as
modules from the repository root, for example
`.venv/bin/python -m tasks.validation_study`. Never write generated study data
into `src/`, `tasks/`, `tests/`, or the repository root.
