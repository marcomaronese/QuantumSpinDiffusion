# Two-spin supervision confirmation v1

Declared on 2026-10-22 after validation selection was saved, before generating
any confirmation observations. The parent screen's protocol SHA-256 is
`387e73320d41964f5a5df7d9edfdf1ab3b123a1ab3a653276a91788e8f508810`.
It selected `final` over `path` with all validation guardrails passing.

## Frozen comparison

Use exactly the already fitted `final` and reference `path` models from
`outputs/experiments/two_spin_supervision_v1/`, for training seeds
67,71,73,79,83. Do not retrain, alter weights, choose among checkpoints, or
select another supervision using test metrics. Retain the same five fixed
classical baseline outputs. Check the parent artifact manifest and replay its
metrics before accessing test data; record hashes of selection, manifest,
protocol, model states, and parameters.

Generate 1200 new test pairs per fitted model, using the parent protocol's
closed seed offset +30000 and the unchanged sigma=0.18, rotation=0.55 data
family. Those observations were not used for training or validation selection.
The five independent training realizations are still the statistical units;
confirmation does not increase the number of training realizations to ten.

## Metrics and decision

The primary comparison is mean test trace distance of frozen `final` minus
frozen `path`. Retain the same requirements: absolute mean improvement at
least 0.01, nonworse on at least 4/5 datasets, mean infidelity degradation
<=0.005, connected-correlation-error degradation <=0.01, and mean marginal
trace-distance degradation <=0.01. Check physicality within 1e-10 and
independent NumPy/SciPy recomputation of primary metrics within 2e-9. Report
paired descriptive Student-t 95% intervals without multiplicity-adjusted
significance or population claims. A failure is reported; it does not trigger
selection of another model on these test data.

Baseline comparisons are descriptive and use explicit unequal budgets. The
supervision-improvement gate is separate from competitiveness: the selected
model may remain worse than even factorized marginals. A passed confirmation
permits the next controlled objective diagnostic in E5; it does not validate
diffusion necessity, hardware readiness, physical readout, or all of E6.

## Artifacts and limitations

Run `.venv/bin/python -m tasks.confirm_two_spin_supervision`. Save the new
protocol/source snapshot before creating test observations in a fresh
`outputs/experiments/two_spin_supervision_confirmation_v1/` directory. Save
test pairs and encoded targets, frozen output states/parameters, training and
sampling metadata references, complete test metrics and uncertainty, failures,
and a checksummed artifact manifest. Never write into the parent study.

Tests concern independent observations for these already fitted models and
one data family. They do not establish generalization to new distributions,
sample-wise physical readout, source classical MI recovery, or competitive
correlated generation. Do not promote results into the manuscript without the
separate named-evidence snapshot and paper audit/build workflow.
