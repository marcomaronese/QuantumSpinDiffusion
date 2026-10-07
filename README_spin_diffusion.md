# Spin phase-space quantum diffusion: minimal generative prototype

This proof-of-concept learns a **bimodal distribution of directions on the
sphere** with the spin-j / multipolar quantum-diffusion construction.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_spin_diffusion.txt
.venv/bin/python spin_diffusion_toy.py --epochs 700
```

Fast sanity check:

```bash
.venv/bin/python spin_diffusion_toy.py --epochs 100
```

Run the regression suite with:

```bash
.venv/bin/python -m pytest -q
```

## Classical diffusion benchmark

Every new single-spin run and two-spin pilot also trains a classical DDPM.
`spin_classical_diffusion.py` implements a time-conditioned MLP, Gaussian
forward noising, epsilon-prediction training, and stochastic reverse sampling
from a Gaussian prior, following [Ho et al.](https://arxiv.org/abs/2006.11239).
It uses a cosine noise schedule with 64 steps. Diffusion happens in Cartesian
coordinates (3 per direction); final samples are normalized onto each sphere.
This is an ambient-space DDPM, not intrinsic Riemannian diffusion. Joint
two-spin samples are modeled together in six dimensions, retaining the ability
to learn correlations.

The baseline uses exactly the quantum run's training directions. Generated
directions pass through the same spin-coherent encoding before comparison of
Frobenius error, fidelity, trace distance, and Husimi-Q JS divergence. Single-spin
metrics use the existing empirical training target; the two-spin pilot uses its
independent test target. Encoding sampled classical outputs introduces Monte
Carlo error. These are not raw-density comparisons or evidence of quantum
advantage, and neither parameter counts nor compute budgets are matched.

Fixed defaults, independent of the quantum objective and epoch count: 300 full
data epochs, batches of 256, two hidden layers of width 64, Adam at `0.001`.
Training and generation use isolated seeded random generators. Configure the
budget on either training entry point with:

```bash
.venv/bin/python spin_diffusion_toy.py --epochs 100 \
  --classical-epochs 300 --classical-steps 64 \
  --classical-batch-size 256 --classical-hidden-width 64 \
  --classical-learning-rate 0.001
```

Each run saves a `classical/` directory with the model checkpoint, input and
generated angles, loss history, source/data hashes, optimizer update count,
parameter count, seeds, and timings. Single-spin runs additionally save the
encoded generated state, Q/loss plots, and metrics; `validation_metrics.json`
contains a `classical` section. `classical_comparison.json` pairs quantum and
classical errors, costs, and configurations. All single-spin study runners
(including diagnostic, hybrid, and instrument studies) write this comparison;
the fidelity and diagnostic reports also include classical mode recovery.
The two-spin comparison and independent audit include a `Classical DDPM` row.

Historical cached single-spin runs have no classical model and are rejected by
new study runs: use a fresh output directory, or `--force` in the validation or
fidelity study. Diagnostic/hybrid manifests still require a fresh directory when
source changes. Historical output files are not upgraded automatically.

To reproduce samples from a checkpoint:

```python
from pathlib import Path
from spin_classical_diffusion import load_classical_model

model = load_classical_model(Path("spin_diffusion_output/classical/model.pt"))
samples = model.sample(2000, seed=8)  # training seed 7 + 1
```

Tests cover analytic forward moments and reverse posterior conditioning,
terminal prior convergence, valid spherical samples, joint data, random-state
isolation, checkpoint reproduction, held-out learning improvement, CLI outputs,
and rejection of cached runs that omit the benchmark.

## Quantum training controls

Select a Task C training objective with, for example:

```bash
.venv/bin/python spin_diffusion_toy.py --objective fidelity
.venv/bin/python spin_diffusion_toy.py \
  --objective multipole --multipole-weighting high-rank
.venv/bin/python spin_diffusion_toy.py --objective husimi-js
```

Available objectives are `frobenius`, `multipole`, `fidelity`, `hybrid`,
`trace-distance`, and `husimi-js`. Frobenius remains the regression default.
Uniform multipole weights are mathematically identical to Frobenius loss;
`rank-balanced` and `high-rank` provide distinct rank-dependent objectives.

Configure the Task D reverse architecture with, for example:

```bash
.venv/bin/python spin_diffusion_toy.py \
  --generator-set no-twist \
  --parameter-sharing shared \
  --ancilla-count 2
```

The defaults remain the original `full` generator family, independent
parameters at every reverse time, and one ancilla. Every run now saves
`config.json`, `reproducibility.json`, architecture metrics, and the trained
parameters in addition to the scientific outputs.

Reproduce the five-seed objective and representation study with:

```bash
.venv/bin/python experiments/validation_study.py
```

Run the mode-resolving `j=2.5` fidelity study and the Task E quantum-instrument
comparison with:

```bash
.venv/bin/python experiments/j25_fidelity_study.py
.venv/bin/python experiments/instrument_study.py
```

The instrument implementation samples normalized conditional states only
after ancilla measurement. The probability-weighted conditional branches are
tested against the deterministic CPTP channel.

Run the separate two-spin correlated pilot and its independent metric audit:

```bash
.venv/bin/python experiments/two_spin_correlated.py
.venv/bin/python experiments/validate_two_spin.py
```

That pilot includes the required factorized, spherical heat-kernel,
low-bond separable tensor-mixture, and matched-parameter neural baselines. It
is a single-seed architecture probe, not an advantage benchmark.

Default model:
- spin `j=2`;
- equivalent symmetric encoding on `N=2j=4` physical qubits;
- spin Hilbert-space dimension `d=N+1=5`;
- six forward/reverse diffusion steps;
- one fresh ancilla qubit per reverse step;
- symmetry-preserving trainable collision channels.

Outputs include:
- `config.json`
- `reproducibility.json`
- `target_Q.png`
- `generated_Q.png`
- `training_loss.png`
- `multipole_power.png`
- `multipole_decay_comparison.png`
- `forward_multipoles.npz`
- `validation_metrics.json`
- generated `(theta, phi)` samples
- target/generated density matrices
- trained reverse parameters.

The run asserts trace-one, Hermiticity, and positivity for every saved
forward/reverse state. The tests additionally check Kraus completeness and
Choi positivity for every sampled reverse collision channel.

## What this proves — and what it does not

It tests the central construction:
1. classical directional distribution -> spin-j mixed state;
2. isotropic quantum forward diffusion;
3. approach to the maximally mixed state;
4. learned non-unitary reverse dynamics;
5. sampling of a generated spherical distribution through the Husimi-Q
   representation.

It is **not yet** a full sample-wise image diffusion model. For images/fields,
the next step is multi-spin latent encoding plus a stochastic quantum
instrument / trajectory-level reverse model.

At the default finite resolution `j=2`, the Husimi-Q readout is strongly
band-limited. The two classical source clusters are visible as a broadened
structure rather than two sharply resolved local maxima; comparisons are
therefore made against the encoded target Q distribution, not the unfiltered
classical density.

The diagnostic representation sweep resolves two significant encoded-Q
maxima starting at `j=2.5` for the current dataset and declared grid-mode
criterion. This is a resolution diagnostic, not a change to the default model.

The initial five-seed fidelity study showed that representational resolution
alone did not guarantee learned mode preservation: path-wise training recovered
both target modes in only one of five 100-epoch runs. The later controlled
diagnostic identified supervision and optimization budget as the practical
bottleneck. Final-only training recovered both modes in all five runs at 100
epochs, while path-wise training reached five of five at 500 epochs without an
architecture change. The first two-spin pilot still places the small quantum
reverse ansatz behind all three correlation-aware classical baselines. These
limits and the completed follow-up are detailed in `DIAGNOSTIC_REPORT.md`.

## Controlled missing-mode diagnostic

Run the fixed-architecture, five-seed 2x2 study at 100 and 500 epochs:

```bash
.venv/bin/python experiments/reverse_diagnostic.py --workers 8
```

This compares `--starting-state mixed|forward` (`I/d` or exact `rho_T`) and
`--supervision path|final`, using fidelity and the original 162-parameter
architecture. The training defaults are unchanged. `rho_generated.npy` is the
output from the declared start; `rho_generated_mixed.npy` always evaluates the
trained stack from the generative prior. Saved trajectories, parameters, loss
histories, configurations, source hashes, and run logs support audit and resume.
Resuming with different source or protocol is rejected: use a new output folder.

Every diagnostic run saves per-rank coefficient errors and power ratios, explicit
matches (including missing matches) for every target mode, weaker-mode relative
height, physicality/CPTP residuals, and independently recomputed state/Q metrics.
The mode criterion remains the existing 80x160 grid, 0.5 relative threshold,
0.35-radian peak separation, and 0.35-radian matching tolerance. Missing weaker
modes have a null recovered height, rather than a misleading sampled Q value.

After that study completes, run the predeclared hybrid validation and confirmation:

```bash
.venv/bin/python experiments/hybrid_objective_study.py --workers 8
```

The hybrid is `(1-lambda) * infidelity + lambda * high-rank multipole loss`,
using the existing normalized `(ell+1)^2` weights. Lambda candidates, fresh
validation/test seeds, metric guardrails, and tie-breaking rules are frozen in
`hybrid_study/protocol.json` before any hybrid training. Selection is written
before test runs start. Test results never select lambda or architecture.

Reverse controls now evaluate each Hermitian generator exponential through its
cached eigendecomposition. This is the same collision ansatz and has the same
angle derivatives; it avoids accumulated numerical unitarity error seen in a
500-epoch run with the generic matrix-exponential implementation. Tests compare
both the unitaries and gradients with the original formula. CPTP tolerances
remain `1e-10`, and no output normalization or PSD projection is introduced.
