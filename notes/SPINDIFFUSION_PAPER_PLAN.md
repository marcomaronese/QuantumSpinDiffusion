# SpinDiffusion paper plan — authoritative research roadmap

Status: active planning document; no new experiment is implied by this plan.
Last reviewed: 2026-10-22.

This is the single authoritative Codex plan for developing the SpinDiffusion
paper. Read it before research work and update its status, evidence links,
decisions, blockers, and next action after each development session. The
claim-to-evidence ledger remains authoritative for what the existing evidence
actually supports; this plan defines work, not findings.

## Restart point

**Next action: E0 — reconcile this plan with the active repository snapshot and
claim ledger, then freeze the E1 protocol before launching any new run.** Do not
start hardware, scaling, or latent-data work while the earlier gates are open.

For each new study, first record its protocol, source fingerprint, immutable
dataset/split identity, seeds, compute budget, metrics, selection rule, and
stopping rule under the repository's current experiment conventions. Use a new
run identity under `outputs/experiments/`; never overwrite historical output
or frozen paper evidence. Promote only audited results into a new named,
checksummed evidence snapshot.

## Paper objective and scope

Develop a careful paper about finite-dimensional spin phase-space diffusion
for directional data: coherent-state encoding of spherical distributions,
isotropic quantum heat evolution, and reverse reconstruction with learned
CPTP collision channels. Establish what the construction does, where finite
resolution limits it, and how it compares with suitable classical methods.

The paper may report negative results. Its scope must follow the evidence:
finite-j representation and reconstruction diagnostics are in scope;
quantum advantage, hardware efficiency, general recovery of arbitrary
densities, and a universal minimum spin are not established. Multi-spin
results remain exploratory until independently confirmed under a frozen,
fair protocol.

## Claim set C1–C5

These are the five claims to organize the paper around. They are bounded
statements with explicit evidence requirements, not commitments that every
claim will survive the experiments. Use the detailed paper claim ledger for
source paths and audited numerical values. When ledger IDs already exist, keep
their identity; this plan's thematic grouping does not renumber or replace
them.

| Plan claim | Bounded statement | Evidence needed / limit |
|---|---|---|
| **C1 — Representation and transfer** | Spin-coherent encoding transfers spherical heat evolution to an isotropic spin channel; the encoding is finite-rank. | Derivation of the coherent-state kernel and multipole transfer function, plus numerical checks. It is a representation-theoretic identity, not a novelty or lossless-encoding claim. |
| **C2 — Forward dynamics and physicality** | Accessible multipoles decay as predicted, and implemented forward/reverse channels preserve the required state/channel constraints within declared numerical tolerances. | Exact small-system replay, trace/Hermiticity/positivity checks, Kraus completeness or Choi checks as appropriate, and independent audit. State numerical evidence separately from the analytic CPTP argument. |
| **C3 — Finite-resolution reconstruction** | For declared synthetic directional distributions and tested spins, the learned reverse model can reconstruct selected encoded structure from the maximally mixed prior; performance depends on objective, optimization, and finite-j resolution. | Train/validation/test separation, rank-resolved and mode diagnostics, global state/readout metrics, multiple independent seeds, and explicit encoding/readout error. No arbitrary-density recovery or universal minimum-spin claim. |
| **C4 — Training and diffusion contribution** | Any benefit of the selected objective or multi-step diffusion protocol is limited to controlled comparisons against direct preparation and fewer-step alternatives under common targets and budgets. | Paired protocol, validation-only selection, held-out confirmation, optimization/resource accounting, and classical diffusion controls. Until those comparisons pass, report diagnostic findings only; do not say diffusion is necessary. |
| **C5 — Correlated extension and sampling** | A collision-channel extension can represent and sample joint directional structure; its measured quality and costs relative to classical baselines are empirical and may be inferior. | Truly factorized baseline, shared-ancilla condition clearly labeled, common readout, joint correlation metrics, physical POVM/instrument checks, independent splits, and uncertainty. No entanglement necessity, hardware efficiency, or advantage claim. |

Map these themes to the manuscript ledger as follows: transfer/representation
to its analytic encoding and representation entries; physicality to forward
and channel audit entries; reconstruction to mode/objective entries; diffusion
necessity to direct/fewer-step controls; correlated sampling to instrument and
two-spin entries. Preserve the original ledger IDs and their caveats when
reconciling the full repository.

## Experiment sequence E0–E6

Proceed in order. Each exit gate is a go/no-go decision. A no-go narrows the
paper claim or sends work back to the named earlier phase; it does not trigger
an automatic architecture sweep.

### E0 — Evidence, implementation, and claim audit

1. Read the repository instructions, current implementation, experiment
   entrypoints, paper claim ledger, evidence manifest, and latest reports.
2. Confirm source/evidence hashes and reproduce the audit without modifying
   immutable evidence. Classify every statement as analytic, numerically
   verified, exploratory, or unsupported.
3. Preserve completed findings and failures, including the existing two-spin
   underperformance and instrument-evidence limitations.
4. Record current code paths and restart checkpoint here. The prior roadmap,
   TODOs, and handoffs are history or compatibility pointers, not competing
   task queues.

**Exit:** all proposed paper statements map to a source or are labeled as
open hypotheses; immutable snapshots pass integrity checks.

### E1 — Freeze the common protocol and evaluation contract

Before training, declare and save:

- Data families: the existing two-mode spherical target for controlled
  diagnostics, plus at least one distinct distribution family for
  generalization. Generate independent train, validation, and test samples;
  document sample counts, mixture weights, angular concentration, and the
  data-generating seed.
- Representation: spin values, angular grid/quadrature, forward diffusion
  rate/time/steps, coherent-state encoding, and a common declared readout.
- Roles of randomness: separate data/split, initialization, model, and sampling
  seeds; pair conditions on data and initialization where the comparison calls
  for it. Keep confirmation seeds and test samples closed until selection.
- Model budget: architecture, optimizer, update/epoch definition, learning
  rate schedule, parameter count, restart policy, maximum updates, wall-clock
  or accelerator budget, and generation/sampling budget. Count all pretraining
  and fine-tuning updates.
- Selection: validation metric and tie-break, hyperparameter search range,
  guardrails, multiplicity policy, uncertainty interval, and stopping rule.
- Error decomposition: report raw-data-to-encoding/readout discrepancy,
  learned-state error against encoded targets, and finite-sample error
  separately. Use one representation/readout across compared models where
  possible; report unavoidable differences explicitly.
- Baselines fixed before comparisons: factorized spherical model, spherical
  heat-kernel model, low-bond tensor-mixture model, neural baseline, and the
  repository's ambient Cartesian DDPM. The DDPM is not an intrinsic
  Riemannian model and has a different historical budget; do not retroactively
  describe archived runs as matched comparisons.

Save the protocol and source fingerprint before a run. A protocol change means
a new study identity. Do not use test outcomes for tuning, exclusions, or
stopping.

**Go:** protocol and evaluation code can be independently inspected and all
baselines have declared inputs, budgets, and outputs.
**No-go:** resolve data leakage, readout mismatch, or undefined metrics before
training.

### E2 — Validate representation, forward process, and resolution

1. Verify spin operators, coherent states, normalization, spherical measure,
   and the finite-rank cutoff `ell <= 2j`.
2. Check the isotropic Lindblad evolution against
   `exp[-D ell(ell+1)t]` for every accessible rank and preserve unmodified
   state outputs for replay.
3. Audit Hermiticity, trace, minimum eigenvalue, trace-preserving/unital
   properties, and channel CPTP residuals. Never repair invalid outputs by PSD
   projection.
4. Quantify representation/readout smoothing and grid/quadrature error as
   spin and resolution vary. Repeat mode counts across grids and thresholds;
   use angular matching and weak-mode height, not grid correlation alone.
5. Keep encoding capacity separate from learned reverse performance.

**Go:** analytic and numerical results agree within predeclared tolerances and
resolution choices do not silently determine a headline claim.
**No-go:** fix implementation or narrow the representation claim; do not begin
model tuning to compensate for a forward-physics discrepancy.

### E3 — Single-spin reverse training and objective selection

Use a fixed architecture for the initial controlled diagnosis. Compare final-
only supervision, the existing summed path loss, and normalized path loss;
pair data/initialization seeds, record gradient scales and updates, and add
stepwise training on exact forward inputs versus composed-chain training.
Evaluate the assembled chain from `I/d`; teacher-forced step accuracy is not
enough. Optional stepwise pretraining plus final-only fine-tuning must count
its added budget. Exact finite-time `rho_T` may be a diagnostic only; always
report generation from `I/d` and never provide hidden forward randomness.

Select objective weights using validation only. The existing `lambda=0.75`
result is a candidate from historical evidence, not a default for new data or
two-spin tasks. Confirm the frozen selection with independent model/data
seeds and held-out samples.

Primary metrics: trace distance and fidelity of density matrices; Frobenius
error as a diagnostic; per-rank complex multipole error/power; mode count,
angular matching and weaker-mode height; common-readout spherical JS/KL or
other declared divergence; and calibration/readout residuals. Report
per-seed values, aggregate uncertainty, training curves, parameters, timings,
and all failed runs. For sampled outputs include sample counts and finite-sample
uncertainty.

**Go:** the selected objective improves its predeclared primary target on
validation and passes global-state/readout guardrails, then confirms on closed
seeds. **No-go:** retain the negative result and investigate gradient scale,
optimization, or representation before increasing capacity.

### E4 — Test whether diffusion and the proposed structure add value

At the selected training objective, compare common-target alternatives:

- direct state preparation or one learned channel;
- fewer reverse steps and the full six-step stack;
- path supervision and final-only supervision, including normalized path loss;
- a learned intrinsic spherical diffusion baseline (the heat-kernel bootstrap
  alone is not learned reverse diffusion);
- the declared factorized, heat-kernel, tensor-mixture, neural, and ambient DDPM
  baselines.

Use paired splits and seeds, document parameter/update/compute budgets, and
report quality, costs, and uncertainty. A parameter-count match alone does not
match optimization or sampling cost. Fix fair resource normalization before
opening confirmation tests.

**Go:** a diffusion-specific statement is allowed only if multi-step diffusion
beats direct/fewer-step alternatives on held-out data under declared budgets
and guardrails. **No-go:** if direct preparation matches or wins, remove the
necessity claim and describe the method as a channel-based reconstruction
construction.

### E5 — Two-spin correlation diagnosis and controlled redesign

First lock E1 for joint data and use the existing architecture to compare
final-only fidelity, summed path loss, normalized path loss, stepwise versus
composed-chain training, and optional budget-counted pretraining. Save local
and joint multipoles, connected correlations, source classical mutual
information, quantum mutual information (explicitly not the same quantity),
state/sample metrics, physicality/CPTP residuals, trajectories, parameters,
and costs.

Only after that diagnosis, change one factor at a time:

1. Establish a truly independent local-channel baseline: separate ancillas,
   product prior, and no shared randomness or coupling. Label the existing
   shared-ancilla model correctly; it can correlate outputs without direct
   spin-spin gates.
2. Compare local channels plus a controlled correlation channel/ancilla, then
   local-channel pretraining followed by correlation-channel training.
3. If E3/E5 diagnosis warrants it, vary Kraus rank/ancilla count, reverse
   layers (3/5/7), parameter sharing, local quadratic terms, and richer
   inter-spin generators one at a time. Count trainable/effective parameters,
   updates, compute, generation, and sampling costs.
4. Add local multipole and cross-spin correlator losses individually. Keep
   global fidelity/trace-distance and marginal guardrails. A restricted set of
   matched observables is not full-state reconstruction.
5. Introduce sample MMD only after a common physical readout exists; verify a
   positive-definite spherical/product-sphere kernel and account for shot bias.

Keep factorized, heat-kernel, low-bond tensor-mixture, neural, and DDPM
baselines fixed through selection. Do not automatically run a large sweep.

**Go:** paired validation supports a selected architecture/objective and joint
correlation improvement is not explained only by better marginals or changed
readout; global guardrails pass. **No-go:** report underperformance or
inconclusive evidence, keep the old result, and do not make general claims
about expressivity.

### E6 — Physical sampling, independent confirmation, and paper readiness

1. Construct a finite coherent-state POVM with nonnegative weights, complete
   logical support, positivity and completeness checks, and Born sampling.
   Quantify how any overlap-matrix whitening changes the Husimi readout;
   state-dependent grid normalization is not a fixed POVM.
2. Use the same POVM/readout for target and generated states. Validate
   quadrature resolution; use product POVMs for joint samples. For instruments,
   retain conditional states, probabilities, outcome histories, and samples;
   compare trajectory ensembles to the deterministic channel before claims.
   Ignoring ancilla outcomes leaves the unconditional channel unchanged;
   instruments alone do not invert discarded harmonics. Outcome-conditioned
   feedback is a separate architecture.
3. Freeze selected models and confirm on 5–10 independent train/test
   realizations where feasible, with paired baselines, held-out samples for
   each fitted model, uncertainty, sample counts, and training/generation/
   sampling costs. Independently recompute primary metrics from saved states
   and samples.
4. Reassess single-spin generalization beyond one synthetic family; vary mode
   separation/imbalance, noise, spin, grid, and peak thresholds. Separate
   representation limits from optimization failures.
5. Promote only audited results into a new named evidence snapshot with
   checksums. Update the claim ledger, evidence audit, figures/tables,
   manuscript interpretation, PDF, and distribution archives together.
6. Complete targeted primary-source novelty review, authors/affiliations,
   corresponding-author, funding, contributions, competing-interest and
   required AI declarations, venue/style, and public code/data archive/DOI
   details. Never invent author or availability information.

**Go:** paper language matches independently audited evidence and all
artifacts/build products are reproducible. **No-go:** narrow the paper to the
supported construction and diagnostics; list missing evidence explicitly.

## Scientific safeguards and out-of-scope gates

- Finite spin truncates accessible harmonics; neither a recovery channel nor
  an ancilla instrument restores arbitrary discarded structure.
- A convex mixture of product coherent states can contain classical
  correlations without entanglement. Quantum mutual information is not source
  classical mutual information and is not an entanglement witness.
- Do not call ambient Cartesian DDPM intrinsic spherical diffusion. Do not
  call the heat-kernel bootstrap a learned reverse-diffusion baseline.
- Do not claim advantage, novelty priority, general minimum spin, universal
  inversion, entanglement necessity, hardware-native execution, or efficiency
  without the corresponding evidence and targeted review.
- Hardware gate stays closed until reliable single-spin held-out recovery,
  improvement over a truly factorized baseline, competitive correlated
  baselines over independent splits, CPTP behavior without output repair, and
  measured resource/sampling costs are established.
- Before Qiskit, validate abstract random collective-rotation convergence to
  the exact Lindblad channel and separate time-step from Monte Carlo error.
  Check symmetric-sector preparation `I_j/(2j+1)`, gate decompositions and
  generator normalization, `J^2` leakage, logical/transpiled depth, two-qubit
  gates, SWAPs, shots, reset, and timing. Progress exact pretraining → noisy
  simulation → limited hardware fine-tuning only after the gate.
- Scale spin count, correlation complexity, or image/field latents only after
  correlated-core validation; include classical decoding, readout, and cost.
  The existing tensor mixture is not a general tensor-network benchmark.
- Petz maps may serve as small-system diagnostics/teachers with an explicit
  training-derived reference state and support. Never derive that reference
  from held-out evaluation targets or call recovery of it universal inversion.

## Milestones and completion record

| Milestone | Exit artifact |
|---|---|
| M0: evidence contract | Current claim map, integrity audit, scoped paper outline |
| M1: frozen protocol | Versioned protocol, split/seed map, source fingerprint |
| M2: validated forward representation | Multipole/physicality/resolution audit |
| M3: selected reverse training | Validation report and untouched confirmation |
| M4: diffusion/baseline decision | Paired comparison, cost table, go/no-go rationale |
| M5: correlated decision | Joint protocol, factorized control, selected or negative result |
| M6: publication package | Physical readout audit, independent confirmation, claim ledger, manuscript and reproducible package |

After each session append date, completed phase/milestone, commit/run/evidence
IDs, decisions, blockers, and exactly one next action below. Do not mark an
experiment complete because code exists or a task can run.

### Session log and current status

- 2026-10-22: plan established from the current code, handoff, TODO, paper
  ledger, and referenced conversation request. No experiment run and no new
  scientific evidence created. Next action is E0 reconciliation, then freeze
  E1 before any run.

## Compatibility and historical mapping

`notes/TODO.md`, `notes/NEXT_STEPS.md`, `notes/CODEX_HANDOFF.md`, and archived
prompts/reports must point here for active work. Keep historical observations,
equations, caveats, and evidence references; mark them as historical when they
conflict with this sequence. Preserve compatible details by incorporating
them into E0–E6 or the safeguards above. Do not keep a second active queue.

Previous-roadmap mapping: repository/test foundation and audited forward
physics → E0/E2; single-spin missing-mode diagnosis and hybrid objective → E3;
two-spin supervision/objectives/capacity → E5; direct-preparation and
few-step controls → E4; physical readout, independent evaluation, evidence
promotion, and submission work → E6; hardware and large scaling → gated
out-of-scope work. Completed results remain historical evidence, not newly
scheduled prerequisites.
