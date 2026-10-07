# Missing-mode diagnostic results

Completed on 2026-10-06: the 40-run controlled diagnostic and 30-run hybrid
validation/confirmation study. **The unchanged architecture is sufficient to
recover both encoded modes.** At 100 epochs, final-only fidelity supervision
recovers both modes in 5/5 seeds from either start, while path supervision passes
in 1/5. Increasing the path-training budget to 500 epochs also gives 5/5, but
retains larger state and high-rank errors. These results support incomplete
optimization under path supervision and an objective tradeoff as the practical
bottleneck; they do not establish a specific gradient-conflict mechanism.
Finite-time prior mismatch is not the primary limitation in this experiment.

The predeclared hybrid selects lambda=0.75 on separate validation seeds and
confirms 5/5 mode recovery on untouched test seeds. Pure final-only fidelity
also passes 5/5 on those test seeds: the hybrid improves high-rank reconstruction,
not the already-saturated discrete recovery rate. No reverse-capacity increase
is indicated by this single-spin study.

The study keeps the original forward diffusion, dataset construction, and 162-parameter reverse architecture fixed (six steps, three layers, one ancilla, full generators, independent time parameters). Each seed pairs the same target and parameter initialization across all eight conditions. The objective is squared-Uhlmann infidelity. Path supervision sums six losses; final supervision uses only the final target loss. Adam and its learning rate (0.04) are unchanged.

## Controlled comparison

| Start | Supervision | Epochs | Both modes recovered | Mean infidelity | Mean trace distance | Mean Q JS |
|---|---|---:|---:|---:|---:|---:|
| I/d | path | 100 | 1/5 | 0.0420838 | 0.132826 | 0.00576177 |
| I/d | final | 100 | 5/5 | 0.000717138 | 0.0160985 | 2.25282e-05 |
| rho_T | path | 100 | 1/5 | 0.0422144 | 0.133359 | 0.00573288 |
| rho_T | final | 100 | 5/5 | 0.000834737 | 0.0176326 | 2.91643e-05 |
| I/d | path | 500 | 5/5 | 0.0121938 | 0.0690143 | 0.00154531 |
| I/d | final | 500 | 5/5 | 2.97782e-05 | 0.003054 | 7.67668e-07 |
| rho_T | path | 500 | 5/5 | 0.0112215 | 0.0649782 | 0.00142021 |
| rho_T | final | 500 | 5/5 | 2.70411e-05 | 0.00312216 | 7.65627e-07 |

Seeds: 3, 5, 7, 11, 13. Success requires both encoded target modes to be matched one-to-one within 0.35 radians. The unchanged significant-mode criterion uses an 80×160 grid, 0.5 relative peak threshold, and 0.35-radian minimum separation. Missing target modes retain null match angles and recovered heights; a high Q value near a missing mode is not counted as recovery.

## Rank-resolved reconstruction

Relative coefficient error is the norm of the reconstructed-minus-target coefficient vector divided by the target coefficient norm. It retains phase/orientation information; a matching power alone does not imply recovery.

| Start | Supervision | Epochs | Mean ell=4 relative error | Mean ell=5 relative error |
|---|---|---:|---:|---:|
| mixed | path | 100 | 0.601901 | 0.859261 |
| mixed | final | 100 | 0.12392 | 0.395437 |
| forward | path | 100 | 0.593667 | 0.928837 |
| forward | final | 100 | 0.136446 | 0.45209 |
| mixed | path | 500 | 0.270791 | 0.846756 |
| mixed | final | 500 | 0.0198279 | 0.111497 |
| forward | path | 500 | 0.267749 | 0.653667 |
| forward | final | 500 | 0.0197383 | 0.117837 |

## Numerical evaluation correction

An initial diagnostic attempt was interrupted when seed 5, final-only fidelity,
500 epochs exceeded the existing **unitarity** check. Reproduction using the
original implementation gave a unitary residual of 1.0288e-10 at reverse step 1;
its Kraus-completeness residual was 7.2749e-11. The shared assertion reports this
as a trace-preservation failure, but the unitary residual was the failing term.

Evaluating the same product of generator exponentials spectrally at the same
parameters reduced the unitary residual to 2.6649e-14 and Kraus-completeness
residual to 1.6118e-14. The implementation now caches the Hermitian generator
eigensystems and evaluates their scalar phases. This changes numerical
evaluation, not the generators, parameterization, forward channel, or data.
Tests compare unitaries and angle gradients with the original exponential.
No tolerance was relaxed and no state was repaired or projected.

The entire 40-run study was restarted using this implementation; only that
consistent rerun is included above. The interrupted outputs, original source
snapshot, and numerical reproduction are preserved in
`diagnostic_numerical_precheck/`. Mode counts in progress logs are peak counts;
the reported success rates additionally require angular matching.

## Numerical checks and reproducibility

All 40 runs passed the existing physicality and CPTP assertions without state repair. Maximum reverse trace error: 3.95e-14; maximum Kraus completeness error: 2.35e-14; minimum Choi eigenvalue: -1.93e-15; maximum forward multipole-decay error: 7.77e-16.

The audit independently recomputes fidelity, trace distance, Frobenius error, Q JS, and Q correlation from saved density matrices and compares them with the training outputs. NumPy coefficient errors obey Parseval’s identity. Saved forward and reverse trajectories are checked again for physicality. CPTP residuals are computed from trained channel parameters by the main run.

The manifest freezes the jobs, source fingerprint, and mode criteria before training. Every run saves configuration, software metadata, full loss history, parameters, forward/reverse density matrices, ordinary mixed-prior output, and detailed mode/rank diagnostics. Loss histories are measured before each update; the saved state metrics use parameters after the last update.

Paired mean differences with descriptive Student-t 95% intervals are saved in `diagnostic_study/summary.json`. These five-seed intervals are not multiplicity-adjusted significance tests. Run costs in the results are wall times measured during concurrent execution, not isolated hardware benchmarks.

Reproduce with `.venv/bin/python experiments/reverse_diagnostic.py --workers 8`. Raw artifacts are in `diagnostic_study/`; the directory is ignored by Git. Source/protocol changes require a new output directory rather than silently reusing cached results.

## Predeclared hybrid validation and confirmation

The hybrid uses `(1-lambda) * infidelity + lambda * high-rank multipole loss`, with the existing mean-one weights proportional to `(ell+1)^2` over all accessible ranks. It uses the generative prior, final-only supervision, 500 epochs, and the unchanged architecture. Candidates are 0, 0.25, 0.5, and 0.75. Validation seeds are 17, 19, 23, 29, 31; untouched confirmation seeds are 41, 43, 47, 53, 59.

Selection first excludes candidates whose validation mean worsens over lambda=0 by more than 0.005 infidelity, 0.01 trace distance, or 0.001 Q JS. It then maximizes both-mode recovery, breaks ties by rank-4-plus-rank-5 squared coefficient error, then chooses the smaller lambda. The selection is saved before test runs begin; the test compares only the selected candidate and lambda=0.

Selected lambda: **0.75**. Validation mode recovery: **5/5**. Confirmation mode recovery: **5/5**. Test metric guardrails: **True**. Physicality and exact decay: **True**. Single-spin gate: **True**.

Protocol, selection, confirmation, per-run matrices, and paired summaries are saved in `hybrid_study/`. Reproduce with `.venv/bin/python experiments/hybrid_objective_study.py --workers 8` after the diagnostic completes.

### Hybrid global and high-rank outcomes

| Split | Lambda | Both modes | Mean infidelity | Mean trace distance | Mean Q JS | Mean ell=4+5 squared error |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.0 | 5/5 | 3.28128e-05 | 0.00347746 | 9.34825e-07 | 1.53654e-05 |
| validation | 0.25 | 5/5 | 3.57064e-05 | 0.00317813 | 1.07854e-06 | 1.23628e-05 |
| validation | 0.5 | 5/5 | 2.8325e-05 | 0.00219942 | 7.77898e-07 | 4.92431e-06 |
| validation | 0.75 | 5/5 | 3.32836e-05 | 0.00174711 | 8.36724e-07 | 2.59165e-06 |
| test | 0.0 | 5/5 | 3.52262e-05 | 0.00361213 | 1.02213e-06 | 1.42722e-05 |
| test | 0.75 | 5/5 | 2.96353e-05 | 0.00176661 | 1.0969e-06 | 2.81031e-06 |

On the test seeds, the selected hybrid reduces mean rank-4-plus-rank-5 squared
error by about 80% and mean trace distance by about 51% versus pure fidelity.
Mean Q JS rises by approximately 7.48e-8, within the predeclared 0.001 absolute
guardrail. These are descriptive five-seed results, not a quantum-advantage claim.

## Scope and next gate

Steps 1–3 of `NEXT_STEPS.md` are complete. The conditional single-spin capacity
expansion in step 4 is not triggered: the unchanged 162-parameter architecture
passes the validation and confirmation mode gate, physicality, forward decay,
and global-metric guardrails.

The two-spin redesign and subsequent multiseed correlated benchmark (steps 5–6)
remain the next research stage. Their classical baselines have not been changed.
The hardware-readiness gate remains closed because the correlated-model
requirements have not yet been demonstrated. Qiskit and hardware work remain
postponed.

Validation: 34 tests pass, including declared training-start/loss semantics,
hybrid gradients and selection, rank Parseval consistency, missing-mode
handling, and the spectral unitary/gradient regression. The original untracked
`output/` directory was left untouched.
