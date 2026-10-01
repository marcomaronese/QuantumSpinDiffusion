# CODEX HANDOFF — Spin Phase-Space Quantum Diffusion

## 1. Project goal

Build and validate a quantum generative model based on the correspondence

\[
\text{spherical fields} \leftrightarrow \text{spin-}j \text{ density matrices},
\]

with forward diffusion implemented as isotropic open-system dynamics and the reverse process implemented as a trainable quantum channel.

The long-term goal is not merely to reproduce a classical diffusion model with a PQC inserted somewhere. The research question is:

> Can a spin/multipolar quantum representation provide an efficient and physically structured generative model for correlated spherical or latent data, with a reverse process formulated as quantum recovery?

The project should progress from a minimal single-spin directional-data model to multi-spin correlated generative models and then, if justified, to spherical fields or latent representations of images.

---

## 2. Core mathematical formulation

### Spin-j representation

A spin-j system has Hilbert-space dimension

\[
d = 2j+1.
\]

It can be represented physically as the fully symmetric subspace of

\[
N=2j
\]

qubits, with collective angular-momentum operators

\[
J_\alpha = \frac12 \sum_{k=1}^{N}\sigma_\alpha^{(k)},
\qquad \alpha\in\{x,y,z\}.
\]

The symmetric-subspace dimension is

\[
N+1 = 2j+1.
\]

### Spin coherent states

A point on the sphere \(\Omega=(\theta,\phi)\) is represented by

\[
|\Omega\rangle_j
=
e^{-i\phi J_z}e^{-i\theta J_y}|j,j\rangle.
\]

In the symmetric-qubit representation this is simply

\[
|\Omega\rangle_j
=
|\Omega\rangle_{1/2}^{\otimes N}.
\]

A classical probability distribution \(p(\Omega)\) may be encoded as

\[
\rho_p
=
\int_{S^2}
p(\Omega)\,
|\Omega\rangle\langle\Omega|\,
d\Omega.
\]

For finite \(j\), this is a band-limited representation: multipoles above \(\ell=2j\) are not independently representable.

### Multipolar expansion

Use irreducible spherical tensor operators

\[
T_{\ell m},
\qquad
0\le \ell \le 2j,
\]

so that

\[
\rho
=
\sum_{\ell=0}^{2j}
\sum_{m=-\ell}^{\ell}
\rho_{\ell m} T_{\ell m}.
\]

This is the quantum analogue of

\[
f(\theta,\phi)
=
\sum_{\ell,m}
a_{\ell m}Y_{\ell m}(\theta,\phi).
\]

Interpretation:
- \(\ell=0\): normalization / monopole;
- \(\ell=1\): dipole;
- \(\ell=2\): quadrupole;
- higher \(\ell\): progressively finer angular structure.

In the symmetric-qubit encoding, higher multipoles correspond to higher-order collective/many-body correlations.

---

## 3. Forward diffusion

Use isotropic angular diffusion:

\[
\frac{d\rho}{dt}
=
-D
\sum_{\alpha=x,y,z}
[J_\alpha,[J_\alpha,\rho]].
\]

The spherical tensor operators are eigenoperators:

\[
\sum_\alpha
[J_\alpha,[J_\alpha,T_{\ell m}]]
=
\ell(\ell+1)T_{\ell m}.
\]

Therefore

\[
\rho_{\ell m}(t)
=
e^{-D\ell(\ell+1)t}\rho_{\ell m}(0).
\]

Consequences:
- high-\(\ell\) structure disappears first;
- low-\(\ell\) structure survives longer;
- the system approaches the maximally mixed state on the spin-j sector,

\[
\rho_T \rightarrow \frac{I_d}{d}.
\]

This is the analogue of a diffusion-model forward noising process.

---

## 4. Reverse model

The reverse dynamics must NOT be implemented as \(U^\dagger\).

The forward dynamics is deliberately irreversible on the observed system. The reverse must be learned as a CPTP map:

\[
\mathcal R_{\theta,t}:
\rho_t\rightarrow\rho_{t-1}.
\]

Implement each reverse step through a Stinespring/collision model:

1. append an ancilla qubit in \(|0\rangle\);
2. apply a parameterized unitary \(U_{\theta,t}\);
3. trace/reset the ancilla.

\[
\mathcal R_{\theta,t}(\rho)
=
\operatorname{Tr}_a
\left[
U_{\theta,t}
(\rho\otimes|0\rangle\langle0|)
U_{\theta,t}^\dagger
\right].
\]

Useful symmetry-aware generators include

\[
J_x,\;J_y,\;J_z,\;J_z^2,
\]

and ancilla couplings

\[
J_x\otimes X_a,\quad
J_y\otimes Y_a,\quad
J_z\otimes Z_a.
\]

In the symmetric \(N=2j\) qubit representation:
- \(J_x,J_y,J_z\) are collective single-qubit rotations;
- \(J_z^2\) becomes pairwise \(ZZ\)-type interactions;
- \(J_\alpha\otimes\sigma_\alpha^a\) becomes system-ancilla pair coupling.

---

## 5. Current minimal prototype

Files:
- `spin_diffusion_toy.py`
- `requirements_spin_diffusion.txt`
- `README_spin_diffusion.md`

Current task:
- synthesize a bimodal directional distribution on \(S^2\);
- encode it as a spin-j mixed state;
- apply exact isotropic forward diffusion;
- train a stack of parameterized reverse collision channels;
- generate \(\rho_{\rm gen}\) from \(I/d\);
- read out a Husimi-Q distribution

\[
Q_\rho(\Omega)
=
\frac{2j+1}{4\pi}
\langle\Omega|\rho|\Omega\rangle;
\]

- sample classical directions from the generated Q distribution.

Default:
- \(j=2\);
- equivalent symmetric encoding: \(N=4\) system qubits;
- spin Hilbert-space dimension: \(d=5\);
- one ancilla qubit per reverse step.

Run:

```bash
pip install -r requirements_spin_diffusion.txt
python spin_diffusion_toy.py --epochs 700
```

Fast test:

```bash
python spin_diffusion_toy.py --epochs 100
```

---

## 6. Important conceptual caveat

The current single-spin model learns one classical directional probability distribution encoded as one density matrix:

\[
p(\Omega)\leftrightarrow\rho.
\]

This is a valid generative model for directional data, but it is not yet analogous to a full image diffusion model where each sample is a structured high-dimensional object.

The next important step is therefore:

\[
\boxed{\text{single spin} \rightarrow \text{multiple correlated spins}}
\]

with

\[
\rho
\in
\mathcal H_j^{(1)}
\otimes\cdots\otimes
\mathcal H_j^{(M)}.
\]

Then each spin can represent a patch/local degree of freedom and entanglement can represent global correlations.

---

## 7. Codex tasks — immediate priority

Do these in order.

### Task A — validate the existing toy code

Run the current script and verify:
- trace preservation;
- Hermiticity;
- positivity of every density matrix;
- convergence of the forward process toward \(I/d\);
- training loss decreases;
- generated Husimi-Q distribution reproduces the target modes.

Add automated assertions/tests for:
- `Tr(rho)=1`;
- minimum eigenvalue \(\ge -10^{-10}\);
- `rho == rho^\dagger`;
- CPTP behavior of each reverse collision channel.

Create a concise test suite.

### Task B — add explicit multipole diagnostics

Implement irreducible spherical tensor operators \(T_{\ell m}\).

For every forward step compute

\[
c_{\ell m}(t)
=
\operatorname{Tr}
[
\rho_t T_{\ell m}^\dagger
].
\]

Verify numerically that

\[
c_{\ell m}(t)
\approx
e^{-D\ell(\ell+1)t}
c_{\ell m}(0).
\]

Produce plots of
- total power per multipole rank,

\[
P_\ell(t)
=
\sum_m |c_{\ell m}(t)|^2;
\]

- theoretical vs observed decay.

This is a central validation of the model.

### Task C — improve the generative training objective

Current Frobenius-density-matrix loss is a simple proof of concept.

Add alternatives:
- multipolar loss

\[
L_{\rm multipole}
=
\sum_{\ell,m}
w_\ell
|c_{\ell m}^{\rm pred}-c_{\ell m}^{\rm target}|^2;
\]

- quantum fidelity loss;
- trace-distance diagnostics;
- Husimi-Q divergence / Jensen-Shannon divergence.

Allow objective selection from CLI.

### Task D — make the reverse architecture configurable

Add CLI/config options for:
- `j`;
- diffusion steps `T`;
- reverse layers;
- generator set;
- shared-vs-independent parameters across time;
- ancilla count;
- learning rate;
- seed.

Save:
- config;
- metrics;
- model parameters;
- figures;
- reproducibility metadata.

### Task E — trajectory / stochastic reverse model

The deterministic mixed-state reverse is only the first stage.

Add a stochastic quantum-instrument version where reverse steps include ancilla measurement:

\[
\mathcal I_{\theta,t}^{(k)}(\rho)
=
K_k\rho K_k^\dagger,
\]

with normalized conditional state

\[
\rho'=
\frac{K_k\rho K_k^\dagger}
{\operatorname{Tr}[K_k\rho K_k^\dagger]}.
\]

The measurement outcome \(k\) supplies stochasticity.

Compare:
- deterministic CPTP reverse;
- stochastic quantum-instrument reverse.

This is important for making the reverse process closer to sample-wise generative diffusion.

---

## 8. Second-stage model: multi-spin correlated generation

Once the single-spin implementation is validated, create a separate experiment, not a destructive rewrite.

Start with \(M=2\) spins, preferably \(j=1\) or \(j=3/2\), to keep the Hilbert space manageable.

Construct a controlled correlated synthetic dataset such as

\[
p(\Omega_1,\Omega_2)
\]

with tunable mutual information.

Examples:
- two correlated directional modes;
- \(\Omega_2\) concentrated near a rotation of \(\Omega_1\);
- mixture of correlated and anticorrelated components.

Encode samples as

\[
|\Omega_1\rangle\otimes|\Omega_2\rangle
\]

and average over the empirical dataset to obtain \(\rho_{\rm data}\).

The reverse ansatz should include both:
- local spin generators \(J_\alpha^{(1)},J_\alpha^{(2)}\);
- inter-spin couplings, e.g.

\[
J_z^{(1)}J_z^{(2)},
\qquad
J_x^{(1)}J_x^{(2)}+
J_y^{(1)}J_y^{(2)}.
\]

Measure whether entangling reverse channels improve reconstruction of joint correlations.

---

## 9. Required classical baselines

Do not claim quantum advantage from the quantum model alone.

For every correlated experiment add:
1. independent/factorized spherical model;
2. classical Riemannian diffusion on \(S^2\) or \((S^2)^M\);
3. tensor-network baseline where feasible;
4. generic small neural generative model with matched parameter count.

The research question is not only quality, but how resources scale with correlation complexity.

Track:
- parameter count;
- training cost;
- sampling cost;
- mutual-information reconstruction;
- divergence from target;
- tensor-network bond dimension where applicable.

---

## 10. Hardware-oriented implementation

Do not move to hardware before validating the abstract spin-space model.

After validation, build a Qiskit implementation of the symmetric encoding.

For \(N=2j\) system qubits:
- collective rotations with parallel single-qubit gates;
- \(J_z^2\) with pairwise `RZZ`;
- system-ancilla \(J_x X_a\), \(J_y Y_a\), \(J_z Z_a\) through pair interactions;
- reset/reuse ancilla after each reverse step.

Important:
- preserve the permutation-symmetric subspace;
- monitor leakage using \(J^2\);

\[
\langle J^2\rangle
\stackrel{\rm ideal}{=}
j(j+1).
\]

Add noisy simulations before any real backend run.

---

## 11. Suggested repository structure

```text
spin-quantum-diffusion/
├── README.md
├── CODEX_HANDOFF.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── spin_qdiff/
│       ├── __init__.py
│       ├── spin.py
│       ├── coherent.py
│       ├── multipoles.py
│       ├── forward.py
│       ├── reverse.py
│       ├── losses.py
│       ├── sampling.py
│       ├── diagnostics.py
│       └── datasets.py
├── experiments/
│   ├── single_spin_bimodal.py
│   ├── multipole_decay.py
│   └── two_spin_correlated.py
├── tests/
│   ├── test_spin.py
│   ├── test_channels.py
│   ├── test_forward.py
│   └── test_multipoles.py
└── outputs/
```

Refactor the existing monolithic script into this structure only after creating regression tests that preserve its behavior.

---

## 12. Definition of success for the first milestone

The first milestone is complete when all of the following are true:

1. The single-spin model runs end-to-end from dataset generation to new samples.
2. Every forward and reverse state is numerically physical.
3. The forward multipoles follow the predicted

\[
e^{-D\ell(\ell+1)t}
\]

law.
4. The reverse channel reconstructs the target spherical distribution materially better than the maximally mixed prior.
5. Results are reproducible across at least 5 random seeds.
6. Metrics and figures are saved automatically.
7. The code is modular and tested.
8. No quantum-advantage claim is made at this stage.

---

## 13. Coding principles

- Use PyTorch complex tensors for differentiable matrix simulation initially.
- Keep system sizes small and exact rather than prematurely optimizing.
- Prefer mathematically transparent code.
- Add type hints and docstrings.
- Separate physics definitions from experiment scripts.
- Never silently project an invalid density matrix back to the PSD cone; surface the problem.
- Use fixed seeds and save complete run configuration.
- Preserve a classical reference implementation for every quantum channel.
- Keep all equations in comments/docstrings consistent with the implementation.

---

## 14. First Codex instruction

Start by reading:
- `CODEX_HANDOFF.md`
- `README_spin_diffusion.md`
- `spin_diffusion_toy.py`

Then:

> Run the current toy model without changing its scientific assumptions. Inspect it for mathematical or implementation errors. Add tests for density-matrix physicality and trace preservation. Then implement irreducible spherical tensor operators and verify numerically that the forward channel produces the predicted \(e^{-D\ell(\ell+1)t}\) decay for every accessible multipole rank. Do not yet refactor the entire codebase or add Qiskit. Report any discrepancy between theory and the current implementation before changing the model.
