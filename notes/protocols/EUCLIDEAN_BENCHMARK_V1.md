# Euclidean synthetic benchmark v1 — QELM comparison point 1

Frozen before the scientific run on 2026-11-17. User-authorized exploratory
benchmark ahead of the E5 redesign; it does not close E4/E5/E6 or enter the
paper evidence. It covers the unconditional 1D/2D families of Figs. 4–5 of
[arXiv:2611.20139v1](https://arxiv.org/html/2611.20139v1). It is an analogous
SpinDiffusion study, not a replication of the QELM architecture, budget or
reported numbers. No class-conditional or image experiment is included.

## Data and immutable identities

The twelve target laws and all parameters are declared in
`src/spin_quantum_diffusion/classical/synthetic.py`, in insertion order:
bimodal, peak plus exponential tail, Student-t(3), Gaussian, log-normal,
skewed three-component mixture; correlated Gaussian, banana,
anti-correlated Gaussian, funnel, two moons and ring. Covariances and
functional forms given in the paper are retained; unspecified widths,
weights and noise parameters are explicit local choices in that table.

For each target index k and realization seed s in `(1031, 2081, 4093)`,
the seed base is `s + 100000*k`. Draw 2048 observations for each of training,
validation, held-out test and an independent real-versus-real reference,
using offsets 0, 10000, 20000 and 30000. Save and checksum the arrays and
train-fitted chart before any model update. Test/reference values are read
for evaluation only after all 36 model pairs have been fitted and saved.
There is no validation selection, checkpoint ranking, hyperparameter tuning,
early stopping or restart. Each fit uses the final checkpoint. Test data
are consumed by this descriptive benchmark and cannot become selection data
for a later claim of independent confirmation.

## Fixed map and explicit truncation

Fit coordinate mean mu and population standard deviation sigma from training
only. With z=(x-mu)/sigma, project z radially onto the disk of radius R=3:
u=(z/R)/max(1, ||z||/R). For 1D, append a zero second coordinate. Encode
the upper-hemisphere direction `(u1,u2,sqrt(1-||u||^2))`. Decode any POVM
direction n by `mu + R*sigma*n[:dimension]`. This is fixed affine classical
postprocessing, with no learned decoder or singular inverse. It round-trips
all in-domain values; out-of-domain values are radially clipped. Report the
clipped fraction separately for every split, including heavy-tail cases.

Represent the training directions by an empirical spin-coherent mixture.
Use j=5/2 (dimension 6) for every target. Read out all quantum states with
the same positive complete coherent POVM: Gauss–Legendre polar order 24 and
48 equispaced azimuths. Recompute on 48 x 96 for grid sensitivity of exact
decoded histograms and weighted coordinate moments. Use standardized edges
`[-inf, linspace(-3,3,B+1), inf]`, B=24 in 1D and B=12 per axis in 2D;
overflow bins retain tail mass. The decoder
has bounded output, so clipping and finite-j smoothing are part of the
measured representation gap, not silently repaired after training.

## Quantum and classical fits

Quantum: the existing single-spin full generator set, one fresh ancilla per
collision, six reverse steps, three independent layers per step; 162 angles.
Forward heat: D=1 and dt=0.35 for six steps. Start generation from I6/6.
Use final-only squared Frobenius loss, Adam at 0.04, 512 full-state updates,
and initialization 0.03*N(0,1) with seed base+40000. Save parameters, every
forward/reverse state, generators, losses, physicality and Kraus residuals,
and training/generation times. No PSD projection or output-state repair.

Classical: the existing ambient Cartesian directional DDPM on the identical
training directions; 64 diffusion steps, width 64, batch 256, 64 epochs,
learning rate 0.001, initialization/noise seed base+50000. This gives
512 optimizer updates, but not equal compute or parameter counts. Save
checkpoint, losses, parameter count and measured costs. Report both direct
decoded DDPM directions and a common-readout comparison: encode 2048 DDPM
directions as a spin state, then use the same coherent POVM/decoder. The
second route contains finite generation error; record its sample count.

Other references: raw-training bootstrap, radially clipped training bootstrap,
encoded training target, encoded held-out target (diagnostic oracle only),
maximally mixed prior, and independent raw reference samples. Factorized,
tensor-mixture and multi-spin neural controls of E5 address a different task;
they are not added as nominally independent single-spin baselines. RFF and
the QELM generator itself remain outside point 1. No efficiency/advantage or
diffusion-necessity claim follows from this budget.

## Metrics, sampling and decision rules

Generate 2048 samples per model/reference, using seed base+60000+condition
index. Raw reference is the independent reference split. Score against the
held-out test, in training-standardized coordinates:

- Primary raw discrepancy: W1 in 1D; mean projected W1 over 32 fixed angles
  `pi*k/32` in 2D. Equal sample counts and sorted empirical quantiles.
- Secondary: energy V-statistic on the first 1024 samples of each set,
  `2 mean|X-Y| - mean|X-X'| - mean|Y-Y'|`, including diagonals. This is the
  biased finite-sample energy/MMD discrepancy, without a significance claim.
- Exact state trace distance and POVM TV against independent validation
  and test encodings; training-to-test gaps. Raw-to-oracle-readout and
  encoded-target-to-model discrepancies have different units and are not
  added or treated as an algebraic error decomposition.
- Decoded means, covariance, fixed-bin raw TV and projection limits/clipped
  fractions are diagnostics. Fixed readout grid refinement compares exact
  decoded binned probabilities, means and second moments; sample metrics
  retain declared finite-sample noise.
- Physicality, forward decay, POVM completeness, Born normalization and
  Kraus completeness tolerance 1e-10. Failure stops the run and preserves
  partial files; nonfinite losses/gradients do likewise.

Save individual values and mean/sample SD over three realizations. Show the
first declared seed, not the visually best run, in distribution panels. Show
all twelve target families and the oracle and DDPM references. Report failures
and representation error even if reconstruction is accurate. This benchmark
has no winner-selection or scientific go/no-go gate; it characterizes the
fixed model. Do not tune this configuration after opening test results.

Before training verify all frozen evidence hashes, record Git state, runtime
versions, protocol/source hashes and a source snapshot in a new directory
under `outputs/experiments/`. An independent NumPy/SciPy replay must verify
saved states, fitted channel trajectories, POVM and primary metrics before
summarizing results. Immutable manuscript evidence is not modified.

An engineering `--smoke` run is separate: seed 991, targets Gaussian and
correlated Gaussian, 64 training / 96 other observations, two quantum
updates, two classical epochs at batch 64 (two updates), energy subset 96.
It checks execution and audit plumbing, with no scientific selection.
