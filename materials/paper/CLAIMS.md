# Claim-to-evidence ledger

Paths below are relative to this folder. All copied experiment artifacts are
listed with checksums in `evidence/manifest.json`. The fresh audit is
`results/audit.json`; the main numerical input is `results/audited_runs.csv`.

| ID | Claim used in the paper | Support | Scope and limits |
|---|---|---|---|
| C1 | Coherent-state encoding intertwines spherical heat evolution with the isotropic spin channel. | Proposition in `sections/theory.tex`; derivation and kernel transfer function in `sections/appendix.tex`; `evidence/source/diagnostic/spin_multipoles.py`. | A standard representation-theoretic identity used by the model, not a priority claim. Finite-rank encoding is not lossless for arbitrary densities. |
| C2 | The seed-7 target first has two significant encoded modes at j=2.5 in the tested sweep. | `evidence/raw/validation_study/representation_sweep.csv`; historical generator and criterion in `evidence/source/diagnostic/experiments/validation_study.py`; Figure 1. | One dataset, eight tested spins, fixed grid and threshold. No universal minimum-spin theorem. |
| C3 | Forward multipole decay and channel physicality agree to numerical precision. | All `forward_states.npy`, `reverse_states.npy`, `forward_multipoles.npz`, and `reverse_parameters.pt` under `evidence/raw/diagnostic_study/runs/` and `evidence/raw/hybrid_study/{validation,test}/runs/`; `results/audit.json`. | 70 small exact simulations. Analytic CPTP argument is separate. Historical Choi checks are preserved; the fresh audit replays Kraus completeness rather than recomputing every Choi spectrum. |
| C4 | At 100 epochs final supervision recovers both modes in 5/5 seeds versus 1/5 for path supervision, with either initialization. | `evidence/raw/diagnostic_study/{manifest,results,summary}.json`; 40 full run folders; Table 1; `results/audited_runs.csv`. | Same fixed architecture, paired data/init seeds. Path loss is a six-term sum; gradient-scale and interference mechanisms are not disentangled. |
| C5 | At 500 epochs all four conditions recover both modes, with smaller errors under final supervision. | Same as C4; Figure 2. | Does not prove a benefit of diffusion or sequential supervision over direct preparation. Final-only training uses no intermediate forward targets. |
| C6 | Validation selects lambda=0.75; confirmation reduces mean top-rank squared error by 80.3% and trace distance by 51.1%. | `evidence/raw/hybrid_study/{protocol,selection,confirmation}.json`; validation/test matrices and results; Table 2; Figure 3; `results/paper_summary.json`. | Five fresh seed realizations; both objectives already recover 5/5 modes. Trace distance worsens slightly on seed 41; mean reduction is not uniform per-seed dominance. Confirmation is not held-out-data evaluation of a fixed trained model. |
| C7 | Sampled ancilla instruments approximate the deterministic channel in ensemble averages. | Exact Kraus identity in theory; `evidence/raw/instrument_study/{summary.json,runs.csv,ensemble_convergence.csv}`. | Historical 256-trajectory aggregates for the original 100-epoch models. Raw conditional trajectories are unavailable in the original outputs; no new ensemble simulation was performed. |
| C8 | The present two-spin reverse models underperform correlation-aware classical baselines. | `evidence/raw/two_spin_pilot/{density_matrices.npz,comparison.csv,summary.json,config.json,independent_validation.json}`; Table 3; fresh six-state audit. | One independent train/test split, unequal training budgets/objectives. The no-direct model still couples spins through a shared ancilla. |
| C9 | Product-coherent encoding admits correlations without entanglement. | Equation for joint encoding in `sections/theory.tex`: convex mixture of product states; target MI in `evidence/raw/two_spin_pilot/summary.json`. | Quantum mutual information is not an entanglement witness or the classical source mutual information. |

## Claims deliberately excluded

- First quantum / fully quantum / classical-data quantum diffusion model.
- First-ever spin phase-space generative model (targeted search is not exhaustive).
- Exact recovery of the raw classical density from finite-j readout.
- Entanglement is necessary or supplies a benefit for these classical data.
- Coarse-to-fine recovery is demonstrated by the reverse trajectories.
- Hardware-native execution, hardware sampling efficiency, or quantum advantage.
- The new classical DDPM has been benchmarked in the historical archived studies.
- Independent train/test evaluation for the single-spin models.
- Diffusion supervision or six reverse steps are essential to the best result.

## Provenance quality

The diagnostic study and both hybrid splits preserve historical source ZIPs,
matching source fingerprints, per-run metadata and trained parameters. Earlier
objective, instrument, and correlated studies have saved outputs but incomplete
contemporaneous source snapshots. The current source capture is not presented
as an exact historical source for those older studies. None of this evidence
was silently upgraded to include later code changes.
