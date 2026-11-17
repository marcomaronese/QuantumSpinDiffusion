# Euclidean charts and coherent readout: diagnosis v1

User-authorized exploratory representation study, 2026-11-17. No reverse
channel is fitted. The [protocol](protocols/EUCLIDEAN_CHART_V1.md) separates
geometric round-trip, physical coherent-state smoothing and inverse-chart
stability, using fresh calibration, validation and confirmation observations.
These results are outside the manuscript's frozen evidence.

## Why a better geometric inverse is not sufficient

The normalized coherent POVM gives an encoded mixture the spherical kernel

\[
K_j(n,m)=\frac{2j+1}{4\pi}\left(\frac{1+n\cdot m}{2}\right)^{2j}.
\]

This is the kernel already derived in the
[manuscript theory](../materials/paper/sections/theory.tex); the operational
POVM interpretation is discussed by
[Jackson and Caves, Quantum 7, 1085 (2023)](https://arxiv.org/abs/2107.12396).
The consequences below are direct calculations for this study, without a
novelty claim.

For the original radius-3 hemisphere chart, the point at the calibration
mean maps to the north pole. At j=5/2 its coherent readout has
`E[n_x^2]=3/14`, so the fixed coordinate decoder alone produces variance
`9*sigma_x^2*(3/14)=27*sigma_x^2/14` around that point. Its standard deviation
is already about 1.39 calibration standard deviations. This is a local
illustration, not a lower bound for every possible target state. It explains
why exact learning of the encoded state did not preserve narrow Euclidean
features in the earlier benchmark.

## Stereographic tails

For standardized planar radius r and scale s, ordinary stereographic encoding
uses `tan(theta/2)=r/s`. Its inverse has `r=s*cot(delta/2)` near the south pole,
where delta is angular distance to that pole. The decoded planar density is

\[
p(x)=q(n(x))\frac{4s^2}{(s^2+r^2)^2}.
\]

If `q(south)>0`, the radial survival is asymptotically
`4*pi*q(south)*s^2/R^2`. First moments exist, but second moments diverge
logarithmically. A finite grid or finite sample gives finite empirical
variances and can conceal that population pathology. All six target laws
in this study have finite covariance. We therefore report ordinary stereo as
a diagnostic and do not use its sample covariance as a population comparison.
No outliers are removed, clipped or renormalized.

## Radial sinh modification

The second radial chart uses `tan(theta/2)=sinh(r/s)` and inverse
`r=s*asinh(tan(theta/2))=s*log(cot(delta/4))`. Its inverse diverges only
logarithmically at the pole. For a bounded spherical density, radial tails
are bounded by a constant times `exp(-2*R/s)`, so every polynomial moment
is finite. For uniform spherical outcomes, exact radial survivals are

\[
P_{\mathrm{stereo}}(r>R)=\frac{s^2}{s^2+R^2},\qquad
P_{\mathrm{sinh}}(r>R)=\operatorname{sech}^2(R/s).
\]

For the sinh chart, `E[r]=s` and `E[r^2]=2*s^2*log(2)`. Deterministic
quadrature tests check these values and the growth of the stereo second
moment with increasing pole resolution. The two maps have the same angular
derivative `2/s` at the origin. Ordinary stereo is conformal; the sinh
variant trades that property for more stable inverse tails. Neither is
assumed to recover arbitrary distributions at finite j.

## Orientation and spherical control

Because `K_j(R*n,R*m)=K_j(n,m)`, jointly rotating a chart and its inverse
does not change the exact decoded oracle law. Orientation is not a useful
selection axis for this representation-only experiment. Finite grids can
break that invariance numerically, and a later trained reverse ansatz need
not be rotation covariant. Those are different effects. The present study
uses exact continuous coherent POVM quantiles, avoiding the readout grid.

A fresh instance of the original angular bimodal family is retained as a
reference. Its significant maxima are assessed under the declared angular
grid/threshold; this checks a previously useful spherical geometry, rather
than treating geometric invertibility as evidence of Euclidean fidelity.

## Numerical results

The six 2D laws use fresh calibration, validation and closed test data,
three realizations and four independent continuous-POVM batches per case.
The screen has 378 cases (7 maps x 3 spins x 6 laws x 3 realizations);
confirmation has 162 cases with scales frozen at j=5/2. No reverse
channel was trained. Error is 32-direction sliced W1 in calibration-
standardized units, mean +/- sample SD across the three realizations.

| Target | Selected sinh scale | Old map | Ordinary stereo | Sinh | Reduction vs old | Real vs real |
|---|---:|---:|---:|---:|---:|---:|
| correlated_gaussian | 2 | 0.539 +/- 0.016 | 0.312 +/- 0.010 | 0.197 +/- 0.001 | 63.5% | 0.027 |
| banana | 1 | 0.637 +/- 0.012 | 0.256 +/- 0.011 | 0.206 +/- 0.008 | 67.6% | 0.031 |
| anticorrelated_gaussian | 1 | 0.487 +/- 0.015 | 0.259 +/- 0.015 | 0.124 +/- 0.001 | 74.5% | 0.021 |
| funnel | 1 | 0.567 +/- 0.009 | 0.207 +/- 0.019 | 0.105 +/- 0.007 | 81.5% | 0.028 |
| two_moons | 0.5 | 0.432 +/- 0.004 | 0.314 +/- 0.008 | 0.308 +/- 0.005 | 28.7% | 0.020 |
| ring | 0.5 | 0.386 +/- 0.003 | 0.326 +/- 0.007 | 0.322 +/- 0.002 | 16.5% | 0.020 |

All six preselected sinh candidates pass the declared *relative adapter* gate:
at least 10% mean error reduction, no paired realization worse by >5%,
and the one-sided 0.995 radial upper-tail inflation limit. This gate is
not a statement of faithful distribution recovery. In particular it does
not test two-sided tail calibration or preservation of holes and modes.

Ordinary stereo improves some central features but produces excessive
tails: its maximum empirical 0.995 radius ratios range from 2.12 to 4.71 across targets.
Its theoretical infinite second moments also remain, regardless of a
finite empirical covariance. No tail trimming was used.

## Fixed-map resolution controls

The scales below remain those selected at j=5/2; they were not retuned
at larger j. These are representation-only controls, not trained model
comparisons or evidence for a universal minimum spin.

| Target | Sinh j=2.5 | Sinh j=5 | Sinh j=10 |
|---|---:|---:|---:|
| correlated_gaussian | 0.197 | 0.129 | 0.086 |
| banana | 0.206 | 0.178 | 0.149 |
| anticorrelated_gaussian | 0.124 | 0.106 | 0.090 |
| funnel | 0.105 | 0.093 | 0.082 |
| two_moons | 0.308 | 0.249 | 0.204 |
| ring | 0.322 | 0.241 | 0.172 |

## Support structure: post-hoc annulus check

Visual inspection prompted an additional exploratory diagnostic after
confirmation. The target annulus has nominal radius 2; measure the
fraction of decoded observations inside radius 1 around its true origin.
This was not a selection or gate criterion. It changes no candidate,
draw or predeclared conclusion. Raw test mass in this hole is 0.00%.

| Readout spin | Old map hole mass | Stereo hole mass | Sinh hole mass |
|---|---:|---:|---:|
| 2.5 | 6.0% | 11.6% | 26.8% |
| 5 | 8.1% | 7.6% | 10.3% |
| 10 | 9.0% | 3.0% | 2.1% |

The lower scalar error does not imply restoration of the annulus hole.
Two-moons structure is also visibly smoothed in the saved first-seed
panels. A future choice intended to preserve support must declare
structural guardrails and use new selection/confirmation data. These
results do not justify calling the low-j Euclidean benchmark solved.

The post-hoc geometry interpretation is consistent with that failure. For
the selected ring scale s=0.5, a standardized radius near sqrt(2) maps close
to the south pole. Its tangential angular stretch is only about 0.17 radians
per standardized unit, compared with 1/3 for the old hemisphere chart.
Thus the candidate can reduce aggregate transport error while making the
annulus's azimuthal detail harder to resolve. This is an interpretation of
the frozen result, not a new scale selection or an additional confirmation.

## Verification and provenance

- Screen: `outputs/experiments/euclidean_chart_v1_screen_20261117_v2`, 378 cases, 477 hashes; independent maximum decoding replay difference 1.469e-10.
  Manifest: `cf3f18e1eca689c1850c1168e8d6285ea73e5111325eb215f4a06de270d9afdb`.
- Confirmation: `outputs/experiments/euclidean_chart_v1_confirm_20261117`, 162 cases, 260 hashes; independent maximum decoding replay difference 7.856e-11.
  Manifest: `6b3d2ec4b50ff472365b7c6c9c757fa88dac9af277feb0f5e2e2d9f69aee62c7`.
- Fresh angular reference: 2 separated significant coherent-readout maxima at j=5/2 under the fixed grid and threshold.
- Kernel rotation-covariance residual: 1.110e-15.
- The initial screen was replayed under the corrected independent verifier,
  which now respects the bounded chart's normalization order at the clipped
  equator. Algebraically equivalent orderings created sqrt(machine-epsilon)
  direction differences there. The original screen is retained; primary
  scores and selections in the replay are identical. It is not an additional
  independent data realization, and no map or selection criterion changed.
- Independent checks reconstruct coherent mixtures, continuous quantiles,
  inverse maps, all primary scores, pole densities and the angular reference.
  The repository suite passes 112 tests.
- Screen/confirmation source fingerprint: `673fbd51cbb3b14b33f19e9835dcf07c52f822ce9961ed455c4e722ab0f2da0d`.
- SciPy runtime: 1.18.1; frozen numerical source and NumPy/Torch/Python versions are saved with both phases.
- No reverse model, quantum-advantage gate or manuscript claim is promoted.
  Confirmation observations are consumed and cannot be reused for selection.

## Artifacts and next use

- [Error comparison](../outputs/experiments/euclidean_chart_v1_confirm_20261117/chart_comparison.png).
- [Decoded distributions](../outputs/experiments/euclidean_chart_v1_confirm_20261117/decoded_distributions.png): first declared seed; fixed display windows, all tails included in metrics.
- [Frozen selection](../outputs/experiments/euclidean_chart_v1_screen_20261117_v2/selection.json).
- [Decisions and exploratory hole masses](../outputs/experiments/euclidean_chart_v1_analysis_20261117/decisions.json).

The sinh family provides better-conditioned adapter candidates, not a
trained generative result. It removes geometric clipping and makes the
decoded moment problem well behaved, but finite-spin smoothing remains.
Any reverse-training comparison on these adapters requires a separate
frozen protocol and fresh evaluation observations. The authoritative
paper restart remains the controlled E5 correlation-stage protocol.
