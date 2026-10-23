# Two-spin supervision protocol v1

Declared on 2026-10-22 before the validation runs. This protocol implements the
first supervision diagnostic in E5 of the authoritative paper plan, under E1's
split, selection, and provenance rules. It does not complete E5 or substitute
for E4's direct-preparation controls or E6's confirmation and physical readout.

## Question and fixed conditions

Does supervision explain the poor two-spin pilot before increasing capacity?
Keep two j=1 spins, three reverse steps, one layer, one shared ancilla, direct
Jz1Jz2 and Jx1Jx2+Jy1Jy2 controls, and independent step parameters: 39 real
parameters. The forward channel is unchanged (D=1, dt=0.5). Compare:

- `path`: summed infidelity on the composed reverse trajectory;
- `path-normalized`: the same sum divided by three;
- `final`: final composed-state infidelity only;
- `stepwise`: average one-step infidelity using exact forward inputs. All
  step losses share one backward pass and one joint optimizer update per epoch.

Use Adam, lr=0.04, betas=(0.9,0.999), eps=1e-8, 80 fixed updates, no restarts
or early stopping. Use CPU, one Torch thread and deterministic algorithms.
All modes evaluate three channels per update. Pair target and initial angles
(0.03 times independent standard normals) across all four conditions.

Every assembled model is evaluated from I_9/9. Stepwise predictions on exact
inputs are not used as a substitute for generated-chain accuracy. Neither
normalized losses nor gradient norms alone identify gradient interference.

## Splits and randomness

The existing correlated normalized-Gaussian mixture has sigma=0.18 and z-axis
rotation angle=0.55 radians. Five independent training datasets use seeds
67,71,73,79,83 and 1200 pairs each. Validation uses 1200 independent pairs per
fitted model with the same seed plus 10000; initialization uses seed plus
20000. These five datasets, not the twenty conditions, are the paired units.

Confirmation seeds (seed plus 30000, 1200 pairs planned) are declared but no
test observations are generated or evaluated by this task. Validation is used
for selection, so its results cannot be called final held-out confirmation.
Other source families and distribution shifts remain future generalization work.

## Baselines held fixed across supervision conditions

Fit each baseline once on each training split, then reuse it for all conditions:
product of training marginals; tangent heat-kernel bootstrap (bandwidth 0.10);
two-component separable tensor mixture; the existing 39-parameter neural
moment generator (300 updates, lr=0.03); and ambient Cartesian DDPM (300
epochs, 64 diffusion steps, batch 256, width64, lr=0.001). Stochastic baselines
use 2000 generated pairs and the same coherent-state encoding as the quantum
targets. Budgets are explicit and are not compute-matched. Quantum output
states are exact; encoded classical generated states have finite-sample error.

Named baseline seed offsets and all numerical settings are persisted in the
machine-readable protocol before training. The shared-ancilla architecture
is not a truly independent local-channel baseline. The product-of-marginals
density is a classical factorization control, not a learned local-channel
ablation. The heat bootstrap is neither an exact spectral sampler nor a
learned intrinsic diffusion model.

## Metrics, guardrails, and decision

Primary metric: mean trace distance to the independent validation density.
Report training/validation infidelity, Frobenius error, local marginal trace
distances, connected 3x3 spin correlations, quantum mutual-information error,
complex joint multipoles grouped by (ell1,ell2), timings, parameter counts,
gradient/update norms, loss histories and physicality/CPTP residuals.

Eligible candidates satisfy every numerical residual tolerance (1e-10) and
validation mean guardrails relative to `path`: infidelity degradation <=0.005,
connected-correlation-error degradation <=0.01, and mean marginal trace-distance
degradation <=0.01. Among eligible modes choose the lowest validation mean
trace distance; exact ties follow path, path-normalized, final, stepwise.
Proceed toward confirmation only if the chosen alternative improves mean
trace distance by at least 0.01 absolute and is nonworse on at least four of
five paired datasets. Otherwise inspect budget/objective limitations before
capacity changes. Do not choose a model by comparison with baselines on test data.

Report paired Student-t 95% intervals descriptively; do not treat five seeds
as population evidence or multiplicity-adjusted significance. Retain failures
and halt selection if a physicality or independent-recomputation check fails.
Primary state metrics are recomputed independently with NumPy/SciPy. Product
multipole errors must obey Parseval; joint forward coefficients must obey the
sum-of-local-rates decay law. Quantum MI is not classical source MI; the latter
is not estimated in this state-only diagnostic. Common POVM sample metrics,
shot uncertainty, and classical source-MI estimation remain explicitly open.

## Artifact contract and restart

Run `.venv/bin/python -m tasks.two_spin_supervision` from the repository root.
All outputs go to a fresh `outputs/experiments/two_spin_supervision_v1/`.
The task refuses existing directories and paths outside `outputs/experiments`.
Save protocol/source hashes, exact source copies, software/Git metadata,
datasets and split hashes, training targets, forward/reverse trajectories,
initial/final parameters, all histories and baseline samples, validation
selection, failure reports, and a checksummed artifact manifest.

Changed command-line settings produce a different protocol hash and require
a new output identity; reduced-budget runs are engineering checks, not v1
scientific validation. New results remain working artifacts until the separate
paper evidence-promotion workflow has audited them. Do not alter frozen evidence
or the manuscript's historical numerical claims to include this screen.
