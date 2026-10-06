# Tasks A–C validation report

## Scope and configuration

The existing single-spin model was kept at its default scientific settings:
`j=2`, `D=1`, `dt=0.35`, six diffusion/reverse steps, three reverse layers,
2,000 directional samples, and 100 training epochs. Reproducibility and Task C
comparisons used seeds 3, 5, 7, 11, and 13 with identical architecture,
optimizer, learning rate, and epoch budget. No Qiskit or hardware model was
added.

## Defect found before behavior changes

The original forward helper symmetrized and renormalized every output, and the
reverse collision function did the same after its partial trace. Although the
underlying exact maps were already physical, dividing by the output trace made
the callable maps nonlinear. Consequently, they were not literally CPTP maps
on arbitrary operators and could hide trace or Hermiticity defects.

Those repairs were removed. Physicality is now asserted rather than imposed,
and the reverse unitary exposes its Kraus operators and Choi matrix for direct
CPTP checks. The 100-epoch loss, final state error, and Q divergence are
unchanged to the digits printed by the original run.

## Numerical validation

| Check | Result |
|---|---:|
| Tests | 13 passed |
| Forward maximum trace error | `8.88e-16` |
| Forward maximum Hermiticity error | `2.74e-18` |
| Forward minimum state eigenvalue | `5.71e-4` |
| Distance from `I/d`, initial → terminal | `0.5210 → 0.006494` |
| Reverse maximum trace error | `2.71e-11` |
| Reverse maximum Hermiticity error | `1.01e-16` |
| Reverse minimum state eigenvalue | `1.95e-2` |
| Reverse maximum Kraus completeness error | `2.47e-11` |
| Reverse minimum Choi eigenvalue | `-1.00e-15` (roundoff) |

All physicality and CPTP residuals pass the specified `1e-10` tolerance.

Across all 25 Task C runs, the worst trace residual was `3.68e-11`, the worst
Kraus-completeness residual was `3.90e-11`, the minimum Choi eigenvalue was
`-1.63e-15` (roundoff), and the maximum multipole-decay error remained
`2.22e-16`.

## Multipole result

An orthonormal irreducible basis `T_(ell,m)` was constructed for every
`0 <= ell <= 2j`. It satisfies the adjoint-Casimir eigenoperator equation,
the spherical-tensor adjoint relation, and reconstructs arbitrary density
matrices from `c_(ell,m) = Tr[rho T_(ell,m)^dagger]`.

Across every accessible `(ell,m)` and all seven saved times, the maximum
absolute discrepancy from

`c_(ell,m)(t) = exp[-D ell(ell+1)t] c_(ell,m)(0)`

was `2.22e-16`. The observed multipole powers consequently follow
`P_ell(t) = exp[-2D ell(ell+1)t] P_ell(0)`. At the latest times the rank-4
curve reaches the floating-point noise floor, which is visible in the
log-scale comparison plot but does not represent a physical discrepancy.

## Five-seed baseline

For the unchanged Frobenius baseline, mean ± sample standard deviation was:

| Metric | Five-seed result |
|---|---:|
| Final Frobenius error | `0.00841 ± 0.00228` |
| Trace distance | `0.09488 ± 0.01498` |
| Infidelity | `0.03351 ± 0.00726` |
| Q-grid Jensen–Shannon divergence | `0.005208 ± 0.001326` |
| Q-grid correlation | `0.99864 ± 0.00055` |

Every seed reduced its training loss and remained physical/CPTP within the
declared tolerance.

## Finite-j representation resolution

Using a fixed 80×160 Q grid, local maxima at least 50% as high as the global
maximum, and a minimum angular separation of `0.35` radians, the encoded target
has one significant maximum for `j = 0.5, 1, 1.5, 2` and two for
`j = 2.5, 3, 4, 5`. Thus the current dataset first resolves as bimodal at
`j=2.5` (`d=6`) under this declared diagnostic. This grid criterion is a
numerical resolution diagnostic, not a theorem about all possible mode
definitions.

## Task C objective comparison

All primary state and Q metrics were independently recomputed from the saved
density matrices before aggregation. Results below are mean ± sample standard
deviation across the same five seeds.

| Objective | Frobenius error | Trace distance | Infidelity | Q-grid JS |
|---|---:|---:|---:|---:|
| Frobenius | `0.00841 ± 0.00228` | `0.09488 ± 0.01498` | `0.03351 ± 0.00726` | `0.005208 ± 0.001326` |
| Multipole, rank-balanced | `0.01059 ± 0.00330` | `0.10553 ± 0.01932` | `0.03214 ± 0.00685` | `0.004029 ± 0.000892` |
| Multipole, high-rank | `0.00765 ± 0.00239` | `0.08573 ± 0.01488` | `0.04112 ± 0.01291` | `0.007716 ± 0.003058` |
| Fidelity | `0.00951 ± 0.00329` | `0.09253 ± 0.02128` | `0.02141 ± 0.00756` | `0.002125 ± 0.000675` |
| Husimi JS | `0.02499 ± 0.00882` | `0.14249 ± 0.02701` | `0.03770 ± 0.00954` | `0.002170 ± 0.000325` |

Fidelity loss is the strongest balanced candidate. Relative to Frobenius on
paired seeds, it reduced infidelity by `0.01210` (95% paired t interval
`[-0.01339, -0.01081]`) and Q-grid JS by `0.003082`
(`[-0.003949, -0.002216]`), winning all five paired runs on both metrics. Its
Frobenius-error difference was not resolved with five seeds.

The objectives expose real tradeoffs rather than one universal winner:

- High-rank multipole loss reduced its intended high-rank diagnostic by
  `0.003397` versus Frobenius (`[-0.005658, -0.001137]`, five of five runs),
  but did not improve Q-grid JS or infidelity consistently.
- Rank-balanced multipole loss improved Q-grid JS in all five paired runs but
  worsened Frobenius error and trace distance.
- Direct Husimi-JS training matched fidelity on Q-grid JS but discarded more
  density-matrix information, producing materially worse Frobenius and trace
  distances. Its coarse 16×32 training grid also differs from the 80×160
  evaluation grid.

Frobenius therefore remains the compatibility default. Fidelity is the
recommended next experimental objective when Q-distribution and quantum-state
fidelity are the priorities; high-rank multipole weighting should be used only
when fine multipole recovery is explicitly the target.

## Reverse/Q limitation

The preserved `j=2` representation does not cleanly resolve the two classical
input clusters as two separate Husimi-Q maxima: even the encoded target is a
broad merged structure. The generated state reproduces that band-limited
target well, but this run does **not** establish recovery of two distinct
decoded modes and makes no quantum-advantage claim.
