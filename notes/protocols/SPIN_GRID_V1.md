# Directional spin grid v1

Exploratory engineering study for a user-requested heatmap. The E5 scientific
restart remains the controlled correlation stage. This study compares a
single circuit family at fixed updates and does not select a configuration,
open confirmation observations, or enter the paper evidence.

## Target and splits

Each observation is three directions. Each sphere has four von Mises–Fisher
components with concentrations `(55, 85, 24, 40)`. Their mean directions in
polar and azimuthal degrees are `(40, 15)`, `(61, 50)`, `(102, 153)`, and
`(127, 271)`. The azimuthal offsets of the sites are `(0, 65, −110)` degrees.
The first two means are close. Site directions are drawn independently
conditional on the three latent labels.

- With probability 0.55, all three labels are a common label drawn with
  probabilities `(0.36, 0.29, 0.22, 0.13)`.
- With probability 0.45, the labels are `(U, V, (U + V) mod 4)` for independent
  uniform `U, V` on `{0, 1, 2, 3}`. Every pair in this branch is independent
  before mixing branches. The three labels satisfy a joint constraint.

This defines 19 distinct joint label combinations because `(0, 0, 0)` occurs
in both branches. For each seed
`107, 109, 113`, generate 1,536 training and 8,192 independent validation
observations using seed offsets 0 and 10,000. Retain samples, latent labels,
and checksums. All cells use the same samples, restricted to the first `N`
directions. The initialization offset is 20,000. The reserved test offset
is 30,000 and test observations are unopened.

## Grid and budget

Evaluate all 15 combinations of `N ∈ {1, 2, 3}` and
`j ∈ {1/2, 1, 3/2, 2, 5/2}`. All sites in a cell have the same `j`, and the
register dimension is `(2j + 1)^N`. The encoding is an empirical mixture of
product-coherent projectors. The forward channel independently applies
isotropic heat diffusion to each site for three steps with `D = 1` and
`Δt = 0.75`. Save every forward state and the distance of its endpoint to the
maximally mixed prior.

The reverse chain starts from the maximally mixed register, with three
collisions and a fresh shared qubit in `|0⟩` per collision. Each collision has
two layers. Within a layer the circuit visits sites in order, applying the
three local rotations and three site–ancilla couplings of `no-twist`, then
the two shared-ancilla rotations. The generators are Hilbert–Schmidt
normalized on their one-site–ancilla support. Each layer and collision has
independent angles. The number of parameters is `6(6N + 2)`. Initial angles
are normal with standard deviation 0.6 and paired across the grid by
reusing the corresponding standard normal draws. Use full-state Adam
(learning rate 0.055, betas 0.9 and 0.999, epsilon 1e−8) for 160 updates on
one CPU thread. The loss is the final squared Frobenius error plus 0.15 times
the mean intermediate squared Frobenius error, divided by training target
purity. Retain the last checkpoint, initial angles, loss history, both
trajectories, number of parameters and updates, training time and deterministic
state generation time. Circuit and simulation costs vary with the cell.

There is no hyperparameter selection, early stopping or restart. The run
stops on failed physicality or nonfinite loss and retains its partial files.
A separate `--smoke` run uses seed 991, 32/48 observations, one update and
only the endpoint values of `j` to check the implementation.

## Readout and comparison

Assign raw directions to the nearest of the four mean directions on each
site. For encoded and generated states, aggregate the same finite coherent
POVM into those four regions, using Gauss–Legendre order 32 and 64 equally
spaced azimuths per site. Use exact product Born probabilities. Report the
maximum change in total variation (TV) across evaluated states when using
48 polar nodes and 96 azimuths. Check state trace, Hermiticity and positivity,
local POVM completeness, Born normalization and reverse Kraus completeness
at `1e−10` without state repair or probability renormalization. POVM effects
are positive sums of rank-one weighted projectors.

- Primary descriptive quantity is joint TV between the raw validation
  histogram and generated Born probabilities. Show raw-to-encoded and
  encoded-to-generated joint TV in adjacent panels. The gaps share units
  and are related by the triangle inequality.
- Compare the connected distribution `p₁₂ − p₁p₂` on the first two sites
  across `N = 2, 3` using half the L1 error to raw validation. Save first-site
  TV for all `N`. Save the analogous third-order cumulant error and source
  strength for `N = 3`. Joint TV covers progressively more sites as `N`
  increases. Region TV measures population mass on the four declared bins.
- Use full-state trace distance to the validation encoding as a second
  learning metric. Save the train-to-validation gaps in raw TV, encoded TV
  and trace distance. Use the product of the encoded training marginals as
  a fixed factorized reference. The heat-kernel, tensor-mixture, neural and
  DDPM comparisons belong to a subsequent scientific protocol with matched
  budgets, if the paper's correlated-core gate permits it.
- Plot cell means and sample standard deviations across the three paired
  realizations. The standard deviations describe run-to-run variability.
  The run has no shot sampling, formal inference or winner selection.

The run verifies the frozen evidence manifest before training and records
its own protocol, source snapshot and artifact checksums in a fresh directory
under `outputs/experiments/`.
