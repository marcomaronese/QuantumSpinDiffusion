# Keeping future paper claims tied to results

1. Define the experiment before running it: scientific question, dataset and
   split, seeds, baseline budgets, primary metrics, selection rule and stopping
   rule. Save that protocol alongside a source snapshot and software versions.
2. Run into a new, clearly named output directory. Never overwrite the frozen
   `evidence/raw/` snapshot or merge new-source runs into historical folders.
3. Save the empirical targets and train/test samples, generated states and
   samples, parameters, full loss histories, trajectories, configuration,
   numerical residuals, timing metadata and source fingerprints. For quantum
   instruments, save per-trajectory states/outcomes if making trajectory claims.
4. Add an immutable dated snapshot under `evidence/additions/`. Give it its own
   manifest with relative paths and SHA-256 checksums. Retain failed runs and
   record exclusions explicitly. Validation selection must precede test access.
5. Recompute the quantities required by each claim from saved states/samples.
   Distinguish independent calculations from checks reusing the training code.
   Report sample count, seed unit, pairing and uncertainty; do not pool changed
   protocols simply to increase the apparent sample size.
6. Extend `scripts/audit_evidence.py` and `scripts/build_assets.py` to include
   the new named snapshot explicitly. Avoid hand-editing generated numerical
   macros and tables. Review rounded numerical examples in the prose too.
7. Update `CLAIMS.md` with exact source paths and the limitations of any new
   conclusion. Promote planned work into the results only when the artifacts
   exist and the audit passes.
8. Recompile and visually inspect the latest PDF, then regenerate both ZIPs.

The initial frozen dataset remains useful even if a future architecture
outperforms it: it documents the original missing-mode mechanism comparison
and the unsuccessful two-spin pilot. Preserve it as an explicit historical
condition rather than removing inconvenient results.
