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

Select a Task C training objective with, for example:

```bash
.venv/bin/python spin_diffusion_toy.py --objective fidelity
.venv/bin/python spin_diffusion_toy.py \
  --objective multipole --multipole-weighting high-rank
.venv/bin/python spin_diffusion_toy.py --objective husimi-js
```

Available objectives are `frobenius`, `multipole`, `fidelity`,
`trace-distance`, and `husimi-js`. Frobenius remains the regression default.
Uniform multipole weights are mathematically identical to Frobenius loss;
`rank-balanced` and `high-rank` provide distinct rank-dependent objectives.

Reproduce the five-seed objective and representation study with:

```bash
.venv/bin/python experiments/validation_study.py
```

Default model:
- spin `j=2`;
- equivalent symmetric encoding on `N=2j=4` physical qubits;
- spin Hilbert-space dimension `d=N+1=5`;
- six forward/reverse diffusion steps;
- one fresh ancilla qubit per reverse step;
- symmetry-preserving trainable collision channels.

Outputs include:
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
