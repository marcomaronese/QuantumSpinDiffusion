# Tasks A–E validation and correlated-pilot report

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

## Task D — configurable and reproducible architecture

The original behavior remains the default, while the reverse model now
supports:

- `full`, `no-twist`, and `minimal` generator families;
- one or more ancilla qubits, with all `2^a` Kraus branches exposed;
- independent parameters at every reverse time or one shared parameter block;
- configurable evaluation-grid resolution;
- complete `config.json`, source/software `reproducibility.json`, architecture
  metrics, state arrays, figures, and model parameters for every run.

The two-ancilla regression uses four Kraus operators and passes Kraus
completeness and Choi-positivity checks. A shared-parameter training regression
also decreases its loss while keeping every generated state physical. The
suite now contains 19 passing tests.

## Higher-spin fidelity study

The fidelity objective was evaluated at `j=2.5`, where all five encoded target
states have two significant Q modes under the fixed 80×160 grid, 50% relative
height, and 0.35-radian separation criterion. Across seeds 3, 5, 7, 11, and
13, the 100-epoch results were:

| Metric | Mean ± sample standard deviation |
|---|---:|
| Frobenius error | `0.01664 ± 0.00566` |
| Trace distance | `0.13283 ± 0.02571` |
| Infidelity | `0.04208 ± 0.01317` |
| Q-grid Jensen–Shannon divergence | `0.005762 ± 0.001738` |
| Q-grid correlation | `0.99650 ± 0.00099` |

Only one of five trained models retained both significant target modes. Four
runs produced one significant generated maximum even though their global
Q-grid correlations remained above `0.995`. This is a useful negative result:
good global distribution/state metrics do not guarantee recovery of a weaker
secondary mode. The physicality/CPTP residuals remained below `3.45e-11`, and
the maximum forward multipole-decay error was `7.77e-16`.

## Task E — stochastic quantum instrument

Each reverse collision now exposes measurement branches
`K_k rho K_k^dagger`, their probabilities, normalized conditional states, and
seed-reproducible sampled trajectories. The normalization occurs only after an
outcome is selected; the underlying channel remains linear. Tests verify that
the exact probability-weighted branch average equals the deterministic CPTP
map.

Using 256 trajectories for each of the five `j=2.5` learned models:

| Metric | Five-run mean ± sample standard deviation |
|---|---:|
| Ensemble squared-Frobenius error to deterministic state | `0.000701 ± 0.000279` |
| Ensemble trace distance to deterministic state | `0.02466 ± 0.00540` |
| Deterministic target fidelity | `0.95792 ± 0.01317` |
| Mean individual-trajectory target fidelity | `0.83306 ± 0.01414` |
| Mean trajectory diversity, squared Frobenius | `0.14479 ± 0.01161` |
| Mean branch entropy | `0.55655 ± 0.03460` nats |

All conditional states were physical; the minimum trajectory eigenvalue was
`3.47e-11`, maximum trace error was `4.44e-16`, and replay of every saved
deterministic state was exact. Individual conditional trajectories need not
resemble the mixed target as closely as their ensemble; that gap is the
expected convex-mixture behavior.

## Separate two-spin correlated pilot

A non-destructive `M=2`, `j=1` pilot was added with local forward diffusion,
a shared-ancilla reverse ansatz, and an ablation that adds direct
`Jz1 Jz2` and `Jx1 Jx2 + Jy1 Jy2` generators. It uses separate train/test
direction-pair samples and evaluates reconstruction of a test density with
quantum mutual information `0.34454` nats.

| Model | Parameters | Trace distance | MI absolute error | Connected-correlation error |
|---|---:|---:|---:|---:|
| Factorized marginals | 16 effective | `0.3272` | `0.3445` | `0.4525` |
| Riemannian heat-kernel KDE | 0 trainable | `0.0342` | `0.0139` | `0.0157` |
| Separable tensor mixture, bond `chi=2` | 33 effective | `0.0405` | `0.0468` | `0.0391` |
| Quantum reverse, no direct interaction | 33 trainable | `0.5328` | `0.3253` | `0.4849` |
| Quantum reverse, direct interaction | 39 trainable | `0.5286` | `0.3195` | `0.4623` |
| Matched-parameter neural generator | 39 trainable | `0.0526` | `0.0431` | `0.0324` |

Direct inter-spin generators help slightly, but both small quantum reverse
models are substantially worse than the correlation-aware classical
baselines in this single-seed pilot. The independently recomputed NumPy/SciPy
metrics agree with the reported values to a maximum absolute discrepancy of
`2.08e-13`. The reverse channels remain CPTP to numerical precision.

This pilot is not a final benchmark: it has one train/test seed, the
Riemannian model is a heat-kernel KDE rather than a learned reverse SDE, joint
quantum Husimi sampling cost is not yet measured, and the shared ancilla can
mediate correlations even in the no-direct-interaction ablation. No quantum
advantage is claimed.
