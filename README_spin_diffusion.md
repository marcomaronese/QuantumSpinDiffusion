# Spin phase-space quantum diffusion: minimal generative prototype

This proof-of-concept learns a **bimodal distribution of directions on the
sphere** with the spin-j / multipolar quantum-diffusion construction.

## Run

```bash
python spin_diffusion_toy.py --epochs 700
```

Fast sanity check:

```bash
python spin_diffusion_toy.py --epochs 100
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
- generated `(theta, phi)` samples
- target/generated density matrices
- trained reverse parameters.

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
