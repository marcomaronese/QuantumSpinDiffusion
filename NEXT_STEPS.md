# Proposed next development steps

## Execution status — 2026-10-06

Steps 1–3 are complete; see [DIAGNOSTIC_REPORT.md](DIAGNOSTIC_REPORT.md).
The 40-run controlled study identifies path supervision and its optimization
budget as the practical missing-mode bottleneck. At 100 epochs, final-only
fidelity recovers both modes in 5/5 seeds from either start, versus 1/5 for
path supervision. At 500 epochs, all four conditions recover both modes in 5/5,
with final-only supervision retaining substantially lower reconstruction error.

The hybrid weight selected on validation seeds is `lambda=0.75`; it confirms
both modes on 5/5 untouched test seeds with the unchanged architecture and
passes the predeclared global-metric guardrails. Pure final-only fidelity also
recovers both modes on these test seeds. The hybrid improves high-rank accuracy.
All physicality/CPTP and exact forward-decay checks pass. Generator exponentials
now use equivalent spectral evaluation to avoid a reproduced numerical
unitarity-tolerance failure; no tolerance or model assumption was changed.

The conditional capacity expansion in step 4 is not indicated by these results.
Steps 5–6 remain pending, and the hardware-readiness gate remains closed.
The original roadmap below is retained for context.

## Scientific objective

Determine why the `j=2.5` representation can resolve two target modes while
the trained reverse model preserves both modes in only one of five runs. Keep
the forward physics, CPTP requirements, and current data model unchanged until
the bottleneck is identified.

Qiskit and hardware work remain postponed. No quantum-advantage claim should
be made.

## 1. Isolate the source of the missing second mode

Run a controlled `j=2.5` diagnostic study that changes one factor at a time:

1. Start the reverse stack from the exact finite-time forward state `rho_T`.
2. Start it from the generative prior `I/d`.
3. Train with the current path-wise loss.
4. Train against the final target state only.
5. Repeat with a larger optimization budget while leaving the architecture
   unchanged.

This produces a 2x2 comparison of starting state and supervision strategy,
plus an optimization-budget check. It should distinguish:

- finite-time prior mismatch;
- path-loss interference;
- incomplete optimization;
- insufficient channel expressivity.

Do not tune the architecture until this diagnostic is complete.

## 2. Add mode-sensitive and rank-resolved diagnostics

For every run, save:

- reconstruction error for each multipole rank `ell`;
- recovery of the `ell=4` and `ell=5` components;
- target and generated significant-mode counts;
- angular matching error for every target mode;
- recovered relative height of the weaker target mode;
- global state and Q-distribution metrics already in use.

Mode recovery should be treated as a primary outcome. High Q-grid correlation
alone is not sufficient because a smooth single-mode approximation can still
correlate strongly with the target.

## 3. Evaluate an objective aligned with the missing structure

After the diagnostic study, evaluate a predeclared hybrid objective:

```text
L = (1 - lambda) L_fidelity + lambda L_high-rank-multipole
```

Choose `lambda` using validation seeds only. Confirm the selected value on new,
untouched seeds. Avoid directly optimizing the discrete grid mode count at
this stage because it is non-differentiable and depends on the chosen grid and
thresholds.

## 4. Perform a controlled reverse-capacity study

If the unchanged architecture fails the expressivity or optimization checks,
change one feature at a time:

- reverse layers: 3, 5, and 7;
- ancilla count: 1 and 2;
- independent versus time-shared parameters;
- `full` versus `no-twist` generator sets;
- additional symmetry-compatible quadratic generators.

Track parameter count and training cost for every configuration. Do not select
a model from test-seed performance.

### Single-spin success gate

A candidate architecture should:

- recover both encoded target modes in at least 4 of 5 validation seeds;
- confirm that result on new test seeds;
- remain physical and CPTP within the existing tolerances;
- retain exact forward multipole-decay validation;
- avoid material degradation of fidelity, trace distance, and Q divergence.

## 5. Redesign the two-spin reverse ansatz

The current direct `J1-J2` generators improve the pilot only slightly and do
not make it competitive with the correlation-aware classical baselines. Test:

- separate local ancillas plus a controlled correlation ancilla;
- increased Kraus rank;
- richer inter-spin generator families;
- local-channel pretraining followed by correlation-channel training;
- final-only versus path-wise supervision.

Keep the factorized, Riemannian heat-kernel, low-bond tensor-mixture, and
matched-parameter neural baselines fixed during the architecture comparison.

## 6. Expand the correlated experiment after architecture selection

Once an architecture is selected without using test results:

- run at least 5 to 10 independent train/test seeds;
- report paired comparisons and uncertainty;
- track trainable parameters, effective parameters, training time, generation
  time, and sampling time;
- measure mutual-information reconstruction, connected correlations, state
  divergence, and physicality;
- independently recompute all primary metrics from saved density matrices.

The current single-seed two-spin pilot is a diagnostic only, not a benchmark.

## 7. Hardware-readiness gate

Do not begin Qiskit implementation until the abstract model:

1. reliably preserves the two `j=2.5` modes;
2. materially improves over the factorized two-spin baseline;
3. approaches the correlation-aware classical baselines across multiple
   seeds;
4. remains CPTP without output repair or PSD projection;
5. has documented resource and sampling costs.

## Immediate next experiment

Begin the two-spin redesign in step 5 with a controlled final-only versus
path-supervision comparison, using the single-spin result as motivation.
Then compare local/correlation ancilla designs and richer channels while
keeping the classical baselines fixed. Select using validation seeds before
opening new test seeds or expanding the correlated benchmark.
