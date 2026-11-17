# Euclidean chart/readout diagnosis v1

Frozen before execution, 2026-11-17. User-authorized representation study,
without reverse-channel training. Earlier benchmark test data are consumed
and will not be read. E5 remains the paper restart; no evidence is promoted.

## Question and analytic checks

Study the entire route x -> sphere -> coherent encoding -> coherent POVM ->
inverse chart. Geometric invertibility alone is not sufficient. The physical
kernel is K_j(n,m)=(2j+1)/(4*pi)*((1+n.m)/2)^(2j), as derived in the manuscript
and in the spin-coherent POVM literature (Jackson and Caves,
https://arxiv.org/abs/2107.12396). The following tail and rotation conclusions
are direct mathematical consequences derived for this study, not claims of
novelty or claims copied from that reference.

In standardized planar polar coordinates, compare seven fixed maps:

1. Original bounded hemisphere map, radius 3, unchanged baseline.
2. Stereographic: tan(theta/2)=r/s, s in (0.5,1,2).
3. Sinh-stereographic: tan(theta/2)=sinh(r/s), s in (0.5,1,2).

Both radial families are invertible away from the south pole, have identical
radial and tangential angular derivatives 2/s at the origin, and use no
truncation. The sinh variant trades conformality for logarithmic inverse
growth near the pole. Mean/std preprocessing is fitted on calibration only;
no covariance whitening, density transport, learned decoder or test-derived
parameters are allowed.

With nonzero Husimi density at the inverse pole, ordinary stereographic
decoding has radial survival proportional to R^-2: first moments exist but
second moments diverge. The sinh variant has exponentially bounded radial
tails and finite moments of every order. For uniform spherical outcomes,
the exact survivals are s^2/(s^2+R^2) and sech^2(R/s), respectively;
the sinh mean radius is s and second radial moment is 2*s^2*log(2).
Check these identities numerically without truncating output. Do not compare
population covariance for ordinary stereographic output as though it existed.

Joint rotation of encoder and inverse does not change the continuum kernel
or decoded law: dot products are invariant. Verify kernel rotation invariance
numerically; do not search over orientations as independent candidates.

## Data and random roles

Use the six 2D target laws already declared in classical/synthetic.py:
correlated Gaussian, banana, anti-correlated Gaussian, funnel, two moons,
ring. For original target index k and fresh seeds s=(1871,4099,8089), set
base=s+100000*k. Calibration, validation and independent validation reference
have 4096 points each, with offsets 0,10000,20000. The screen does not
generate confirmation data. The calibration chart is shared by all maps.

For j=(2.5,5,10), sample the *continuous* coherent POVM exactly, using
t=(1+n.m)/2=U^(1/(2j+1)) and uniform tangent azimuth. No quadrature grid
or decoder clipping is used. Save common calibration indices and two uniform
arrays for four batches of 4096 draws, seed base+30000+100*(2j).
All seven maps share those draws within a spin/realization. The encoded state
is the empirical calibration mixture; validation is used only for scoring.

The main comparison is j=2.5. j=5,10 are representation-resolution controls,
without changing or fitting a reverse model. There are 378 screen cases.
An engineering smoke run uses only ring, seed 991, j=2.5, three maps
(baseline, stereographic s=1, sinh s=1), 64 points and two batches.

## Metrics and selection before confirmation

Report mean 32-projection sliced W1 over four batches against validation,
energy V on the first batch/first 1024 points, empirical radial quantiles
0.5/0.95/0.995 and maximum, and empirical outlier mass beyond the validation
0.995 radius. Save round-trip error, clipped fraction, spherical noise angle,
and training radius quantiles. Report covariance only for finite-variance
maps. All distances use calibration-standardized coordinates. Record local
angular stretches for radial maps; baseline has explicit bounded support.

For each target, select the stereo scale by minimum mean validation sliced
W1 across the three realizations at j=2.5; break ties by ascending s. For
sinh, first retain scales whose pooled 0.995 output radius is <=1.5 times
validation in every realization, then minimize the same score; if none pass,
retain the unconstrained minimum for diagnostics but mark it ineligible.
Save and checksum selection.json before generating confirmation.
The candidate recommended for retraining is sinh only when its validation
W1 is at least 10% lower than baseline and its pooled 0.995 output radius
is no more than 1.5 times validation in every realization. Otherwise recommend
no map change. Ordinary stereo is a diagnostic because all six target laws
have finite covariance, whereas generic stereo readout does not.

## Closed confirmation

In a separate fresh output directory, verify screen hashes/source identity,
copy frozen selections and calibration data, then generate 4096 fresh test
and real-reference points per target/realization (offsets 50000,60000).
Use fresh shared POVM quantiles at base+70000+100*(2j). Evaluate baseline and
the two selected radial scales at every spin: 162 cases. No rescreening at
larger j, parameter changes, restarts or checkpoint selection.

A preselected sinh candidate passes its bounded readout-improvement gate
only if, at j=2.5, test mean sliced W1 improves by at least 10%, no paired
realization worsens by more than 5%, and the 0.995 radius ratio remains <=1.5
in all realizations. Report no-go/inconclusive outcomes as such. This is a
gate for an adapter candidate, not good raw-density recovery, a reverse-model
gate, efficiency, or evidence for arbitrary-density recovery.

## Verification and artifacts

Before execution verify the 1269 frozen evidence hashes and save protocol,
source hashes, runtime versions, datasets and noise identities. Use fresh
outputs/experiments directories, preserve every screen/confirmation case,
all decoded batches and encoded states. Fail on nonfinite values, undefined
pole inversions, or state physicality residual >1e-10. No state repair or
outlier deletion. The original bounded baseline alone reports its clipping.

Independent NumPy/SciPy replay must reconstruct the maps, continuous POVM
quantiles, decoding, radial tails and all primary scores from saved inputs.
Check exact angular moments and rotation invariance as analytic controls.
Retain a fresh angular bimodal reference using the existing directional data
generator, j=2.5, 2048 observations, seed 71873, and report its coherent
readout's separated significant maxima on a fixed 65x128 angular grid
(5x5 local maxima, height >=0.25 of global maximum, minimum separation 20deg).
This illustrates the original spherical task; it is not a Euclidean winner.
All observations, including confirmation, are consumed at completion.
