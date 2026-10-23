# Independent-local control: E5 working result

Date: 2026-10-23. Protocol and implementation were committed as `b2250b0`
before the full study. This is a validation diagnosis on one data family,
not independent confirmation or a promoted manuscript result.

## Study and provenance

- Protocol: [TWO_SPIN_LOCAL_CONTROL_V1.md](protocols/TWO_SPIN_LOCAL_CONTROL_V1.md).
- Run: `outputs/experiments/two_spin_local_control_v1/`.
- Machine protocol SHA-256:
  `f2a7b992060e6919d5710544b25898eab669f438cf909b500b4cee7d6134f6b8`.
- Sealed artifact-manifest SHA-256:
  `e1a02998bf99e0d8995ed35fa2aceccf79e58c75c2d8592ecddc4fd9194a4f31`.
- Five new training realizations: 101,103,107,109,113; 1200 training and
  1200 independent validation pairs per realization. No previous study's
  observations reused. The source snapshot records a clean `codex/paper`
  checkout at `b2250b06d2736637c6b77297687e2ac27c091fe8`.
- Fifteen quantum fits: independent local channels (48 parameters, two distinct
  ancillas), shared ancilla without direct interactions (33), shared ancilla
  with direct interactions (39). All use three steps, one layer, final-only
  infidelity, 80 Adam updates and generation from I9/9. Shared-model starting
  angles are paired by generator name. Local halves have independent RNG roles.
- Five classical controls fitted once per training split, with frozen outputs
  shared across comparisons. Neural/DDPM: 300 updates/epochs; stochastic
  classical outputs: 2000 generated pairs. All state comparisons use the same
  coherent encoding; this is not a common finite-POVM sampling experiment.

## Validation findings

Means across five paired training realizations; lower errors are better.

| Model | Trace distance | Infidelity | Connected-correlation error | Mean marginal trace distance |
|---|---:|---:|---:|---:|
| Learned independent local | 0.36636852 | 0.19597085 | 0.44750036 | 0.10533557 |
| Shared ancilla, no direct interactions | 0.45477755 | 0.31732239 | 0.40809043 | 0.17025295 |
| Shared ancilla, direct interactions | 0.46577034 | 0.33773337 | 0.40688486 | 0.18938427 |
| Empirical factorized marginals | 0.32507980 | 0.18922029 | 0.44750036 | 0.02161952 |
| Tangent heat bootstrap | 0.02981909 | 0.00154766 | 0.01167448 | 0.02319094 |
| Separable tensor mixture | 0.04941728 | 0.01113372 | 0.03749532 | 0.02161952 |
| Neural moment generator | 0.07578946 | 0.01450442 | 0.04832772 | 0.03043761 |
| Ambient Cartesian DDPM | 0.11123201 | 0.01482805 | 0.09632261 | 0.07695727 |

The two shared-ancilla models reduce connected-correlation error but worsen
trace distance and infidelity on all five realizations. Their mean marginal
errors also worsen. Relative to learned independent local channels:

| Paired effect, shared minus independent local | No direct interactions | Direct interactions |
|---|---:|---:|
| Mean trace-distance change | +0.08840903 | +0.09940182 |
| Descriptive paired 95% interval | [0.06543046, 0.11138759] | [0.09397877, 0.10482486] |
| Mean infidelity change | +0.12135154 | +0.14176252 |
| Mean connected-correlation-error change | -0.03940993 | -0.04061550 |
| Mean marginal-error change | +0.06491739 | +0.08404870 |
| Nonworse trace-distance realizations | 0/5 | 0/5 |

These intervals are descriptive, not population or multiplicity-adjusted
significance statements. Full values and uncertainty for all metrics are in
`results.json` and `selection.json` in the working run.

## Decision and interpretation

**No-go.** Neither shared candidate passes the predeclared infidelity or
marginal guardrails (+.005 and +.01). Neither improves the primary trace
distance, although both reduce correlation error. `eligible` is empty and
no candidate is selected. The selection summary's zero improvement fields
are placeholders for an absent eligible candidate; use the paired effects
above to describe the actual differences.

No confirmation pairs or test targets were generated. The declared test
identities remain closed. Do not choose another architecture on those data or
relabel validation as confirmation. The earlier supervision confirmation
remains a separate bounded result and is not invalidated by this new study.

The independent-local trajectories remain products, with zero connected
correlations and quantum MI to numerical precision, as required by their
construction. A shared ancilla without direct spin-spin controls does produce
some joint structure here; it is not an independent-channel baseline. That
structure is insufficient to improve global reconstruction in this protocol.

Different parameters (48/33/39), ancilla dimensions/ranks, control normalization
and compute prevent isolating a causal effect of direct coupling, comparing
optimal expressivity, or claiming a resource-matched result. The empirical
factorized state is a distinct nontrained control, not a proven optimum or
bound on learned product states. All quantum conditions underperform the
listed classical controls in mean trace distance on this validation study.
Classical sampled-state uncertainty is not matched to exact quantum states.
Source classical MI, other families, physical readout and diffusion necessity
remain untested.

## Verification and figure

All **67 tests pass**. Independent replay verifies **395 sealed artifacts**,
40 model rows and 80 train/validation metric evaluations. Metrics are replayed
with NumPy/SciPy; all 45 fitted reverse channel steps are reconstructed with
`scipy.linalg.expm` from saved generators and weights, without Torch's channel
implementation. Maximum residuals:

- state-metric discrepancy: 1.4933e-13;
- channel/state replay discrepancy: 1.2626e-14;
- Kraus completeness: 2.3474e-15;
- minimum Choi eigenvalue: -2.8749e-15;
- independent-local product/correlation/MI residual: 1.1103e-15.

The original 1269 frozen paper-evidence hashes also pass unchanged. No frozen
paper evidence, claim-ledger numerical conclusion, or manuscript asset changed.

A partial **F6 working diagnostic** is available as
`outputs/experiments/two_spin_local_control_v1/figures/local_control.pdf`
and `.png`: per-realization trace distance, connected-correlation error and
marginal error, including the fixed classical controls. It was rendered after
audit and visually inspected; `figures/provenance.json` records input, output
and plotting-source hashes. The topology diagram and any manuscript promotion
remain separate work. The plot is not an additional numbered manuscript figure.

Mean measured training / exact-state generation times per fitted realization
on this CPU run: independent local 0.5336s / 0.001814s; shared no-direct
0.2738s / 0.000812s; shared direct 0.3115s / 0.000910s. These are local wall
times with cached generator spectra, not hardware or sampling-efficiency claims.
Complete baseline timing and parameter information remains in each metrics row.

Reproduce under a fresh output identity:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/spindiffusion-mpl \
  .venv/bin/python -m tasks.two_spin_local_control --outdir outputs/experiments/two_spin_local_control_v1_replay
.venv/bin/python -m tasks.validate_two_spin_local_control --study-dir outputs/experiments/two_spin_local_control_v1_replay
.venv/bin/python -m tasks.plot_two_spin_local_control --study-dir outputs/experiments/two_spin_local_control_v1_replay
```

The separate seed-997 engineering check used two updates and 40 observations;
it is excluded from all scientific aggregates above. Working outputs are
local and ignored by Git; the committed protocol, source and report permit
reproduction, but do not replace a future named evidence snapshot.

## Next action

Freeze a new E5 protocol for **local-channel pretraining followed by a
controlled correlation stage**, with a frozen-local reference, an explicit
total-budget control and marginal/global guardrails. The current result
motivates testing this separation; it does not establish its success or identify
an optimization mechanism. Count every pretraining/fine-tuning update and
declare fresh selection/closed-confirmation identities before training. Do not
change the current study or start a capacity sweep. Active priorities and
status remain solely in [the official plan](SPINDIFFUSION_PAPER_PLAN.md).
