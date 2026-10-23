# Two-spin supervision screen and held-out confirmation

Completed: 2026-10-22, on `codex/next-milestones`. Working research results;
not promoted into the frozen paper evidence or its claim ledger. Active
priorities remain only in [the official plan](SPINDIFFUSION_PAPER_PLAN.md).

## Scope and design

The first E5 supervision diagnostic is complete under the E1
[screen protocol](protocols/TWO_SPIN_SUPERVISION_V1.md) and subsequent
[confirmation protocol](protocols/TWO_SPIN_CONFIRMATION_V1.md). The existing
two-j=1 shared-ancilla/direct-interaction architecture is unchanged: three
steps, one layer, 39 real parameters. All four conditions use 80 Adam updates
at lr=0.04 and the same targets/initializations within each seed.

Five independent training datasets use seeds 67,71,73,79,83. Each has 1200
training pairs and 1200 independent validation pairs. The selected model is
then evaluated, without retraining or model selection on test data, on 1200
previously unopened test pairs per fitted model. These remain five training
realizations, not ten. The same one-family normalized-Gaussian directional
source uses sigma=0.18 and rotation=0.55 radians throughout.

Baselines were fitted once per training split and held fixed across conditions:
factorized training marginals, tangent heat bootstrap, two-component tensor
mixture, 39-parameter neural moment generator, and ambient Cartesian DDPM.
Neural/DDPM budgets were 300 epochs; the classical sampled output states use
2000 generated pairs. These budgets are explicit and not compute-matched.

## Validation selection

Twenty quantum runs compare composed path sum, its normalized mean, final-only
supervision, and teacher-forced one-step training. All outputs are evaluated
as assembled chains from I_9/9, including the teacher-forced models.

| Supervision | Mean validation trace distance | Mean infidelity | Mean connected-correlation error |
|---|---:|---:|---:|
| Summed path | 0.50175205 | 0.37047951 | 0.44517832 |
| Normalized path | 0.50187451 | 0.37061108 | 0.44530600 |
| Final only | 0.46169453 | 0.33187380 | 0.41342810 |
| Exact-input stepwise | 0.56699902 | 0.43906359 | 0.50167334 |

Final-only passes the predeclared eligibility guardrails, improves mean trace
distance by 0.04005753, and is better in all 5/5 paired datasets. Its
descriptive paired 95% Student-t interval for final minus path trace distance
is [-0.04848195, -0.03163311]. The saved selection authorized confirmation.
Stepwise training fails the global guardrails; teacher-forced step losses do
not imply successful assembled generation. Normalization alone changes little
at this fixed Adam budget. Gradient scales and update norms are saved; these
results do not identify gradient interference as the mechanism.

## Confirmation on previously unopened observations

The final-only choice was frozen before test generation. The exact same fitted
final/path parameters and baseline outputs were used, with no new training.

| Frozen model | Mean test trace distance | Mean infidelity | Mean connected-correlation error |
|---|---:|---:|---:|
| Summed-path quantum | 0.50198360 | 0.36961563 | 0.44631926 |
| Final-only quantum | 0.46199196 | 0.33154518 | 0.41487668 |
| Factorized marginals | 0.32810992 | 0.19150860 | 0.45214396 |
| Tangent heat bootstrap | 0.02701282 | 0.00124499 | 0.01254408 |
| Two-component tensor mixture | 0.04654767 | 0.01121198 | 0.03810984 |
| Neural moment generator | 0.06536000 | 0.00962743 | 0.03764050 |
| Ambient Cartesian DDPM | 0.10217544 | 0.01236757 | 0.08527853 |

The supervision-confirmation gate passes: mean improvement 0.03999164,
better trace distance in 5/5 paired datasets, and all infidelity, connected-
correlation and marginal guardrails pass. The paired trace-distance difference
is -0.03999164, with descriptive 95% interval [-0.04811359, -0.03186969].

This confirms a bounded supervision improvement for the fixed architecture
and budget. It does not establish competitive correlated generation: even
the factorized baseline has substantially lower global state error. The
quantum model improves the connected-correlation metric relative to
factorization while retaining much poorer marginals/global reconstruction.
Do not turn either observation into a general expressivity or entanglement claim.

## Replay, physicality, and artifacts

- Parent study: `outputs/experiments/two_spin_supervision_v1/`.
- Independent confirmation: `outputs/experiments/two_spin_supervision_confirmation_v1/`.
- Screen protocol SHA-256: `387e73320d41964f5a5df7d9edfdf1ab3b123a1ab3a653276a91788e8f508810`.
- Parent artifact-manifest SHA-256: `8ac461d7c3b523d1cfefa958a16b84338b66b60570746fa46bc1c0f0c01ad7a5`.
- Confirmation artifact-manifest SHA-256: `c5ac5d1fde9d8da996badb3d971b19e1b171ae54588dd0965f550b3d9ffdb379`.

The screen preserves all source/protocol copies, split hashes, data and states,
initial/final parameters, losses, gradients/updates, complex local/joint
multipoles, residuals, timings, classical model artifacts, and generated pairs.
No failed study runs were excluded. A separate reduced-budget run in
`two_spin_supervision_v1_smoke/` is an engineering check and is not pooled here.

Provenance clarification: the generic environment-metadata helper wrote its
unused default `seed=7` into the original screen's `reproducibility.json`.
Actual dataset, validation, initialization and baseline seeds are defined by
`protocol.json`, the saved per-seed artifacts, and the exact source snapshot.
Every saved train/validation/test dataset was regenerated from its declared
seed and matched exactly. A separate `provenance_clarification.json` records
that check; the original sealed files were not rewritten. Future invocations
now record `study_seeds` and the protocol hash instead of a misleading single
default seed. This metadata correction changes no model or metric.

Independent NumPy/SciPy replay verified 45 model rows on train and validation
(90 evaluations), 348 sealed artifacts, and the 35 confirmation model/test
rows with 150 sealed artifacts. The maximum training/validation metric
discrepancy is 2.17e-13; the Torch-versus-NumPy confirmation check differs by
at most 2.69e-13. Recomputing the saved NumPy test metrics reproduces them
exactly. Frozen output-state hashes match the parent models.

The largest quantum Kraus-completeness residual is 1.56e-14 and minimum Choi
eigenvalue -2.69e-15 (roundoff). Saved assembled trajectories pass trace,
Hermiticity and positivity checks without output repair. Product multipole
errors satisfy Parseval and forward coefficients satisfy the sum-of-local-
rates heat law. All 62 repository tests pass after implementation.

To audit existing artifacts, invoke the modules from the repository root:

```bash
.venv/bin/python -m tasks.validate_two_spin_supervision
.venv/bin/python -m tasks.validate_two_spin_supervision --confirmation \
  --study-dir outputs/experiments/two_spin_supervision_confirmation_v1
```

These are state-level, one-family comparisons with five statistical units;
intervals are descriptive and not multiplicity-adjusted. Source classical MI,
physical POVM sampling, raw-distribution readout fidelity, broader families,
diffusion-necessity controls, and hardware/resource advantages remain open.
The article still reports the historical evidence; this working study has
not changed its numerical claims, PDF, or distribution archives.
