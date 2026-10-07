# Experiment tasks

These modules orchestrate studies using the reusable implementations under
`src/spin_quantum_diffusion/`. Run them from the repository root with
`python -m tasks.<name>`.

- `validation_study`: representation and objective sweeps.
- `j25_fidelity_study`: five-seed mode-recovery study at `j=2.5`.
- `reverse_diagnostic`: controlled initialization, supervision, and budget grid.
- `hybrid_objective_study`: predeclared hybrid-weight selection and confirmation.
- `instrument_study`: stochastic ancilla-instrument comparison.
- `two_spin_correlated`: correlated two-spin pilot and classical baselines.
- `validate_two_spin`: independent audit of saved two-spin results.

Defaults write to `outputs/experiments/`. Use a new directory when source or
protocol changes so cached results cannot be mistaken for a fresh run.
