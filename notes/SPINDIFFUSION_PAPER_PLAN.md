# SpinDiffusion paper plan — authoritative research roadmap

Status: active research roadmap; execution status is recorded below.
Last reviewed: 2026-10-23.

This is the single authoritative Codex plan for developing the SpinDiffusion
paper. Read it before research work and update its status, evidence links,
decisions, blockers, and next action after each development session. The
claim-to-evidence ledger remains authoritative for what the existing evidence
actually supports; this plan defines work, not findings.

## Restart point

Work on `codex/paper`, derived from the consolidated development baseline in
`main`. [Branch policy and historical recovery](BRANCHES.md) define the two
branch roles. Repository cleanup does not rerun or complete a scientific gate.

**Next action: E5 — freeze the truly independent local-channel control protocol
using the selected final-only supervision, before further objective/capacity
changes.** The fixed-architecture supervision screen and independent-observation
confirmation passed their bounded improvement gate on 2026-10-22; they do not
pass the competitive-correlation or hardware gates. The E0 frozen-evidence audit
also passed on 2026-10-22;
historical single-spin successes are not prerequisites to rerun. Do not
start hardware, scaling, or latent-data work while the earlier gates are open.

Completed protocols: [`supervision screen`](protocols/TWO_SPIN_SUPERVISION_V1.md)
and [`held-out confirmation`](protocols/TWO_SPIN_CONFIRMATION_V1.md).
See [`the working results report`](TWO_SPIN_SUPERVISION_REPORT.md). Test
observations in that confirmation are now consumed: never reuse them for
subsequent selection. New objective/architecture studies need fresh protocols,
selection splits and closed confirmation identities.

### Progress across the complete plan

| Phase | Current state | Remaining requirement |
|---|---|---|
| E0 | Frozen evidence replay passed: 1269 manifest entries, 70 single-spin runs and six correlated models; manuscript/claim mapping reviewed | Audit new results independently before evidence promotion |
| E1 | Screen and subsequent confirmation protocols frozen and executed with separate train/validation/test observations, fixed budgets/guardrails and baselines | Future studies require new frozen protocols; broader generalization remains open |
| E2 | Historical forward decay/physicality replay passed; representation sweep preserved | Broader resolution, grid/threshold and readout robustness remain open |
| E3 | Existing single-spin supervision/hybrid results preserved and replayed | Independent evaluation per fitted single-spin model, normalized/stepwise controls and broader families remain open |
| E4 | Open; not established by final-only training | Direct/fewer-step and learned intrinsic diffusion comparisons |
| E5 | Pilot, 20-run supervision screen and selected-final-versus-path confirmation complete; final-only improves but still underperforms classical baselines | Truly independent local-channel control next; controlled joint-objective work and conditional capacity studies remain open |
| E6 | Historical instruments/package exist; independent observations confirmed the bounded E5 supervision improvement on five fitted realizations | Physical finite POVM, complete trajectories, confirmation after future redesign, broader robustness, evidence promotion and publication remain open |

This checkpoint resumes E5 because the two-spin pilot and the single-spin
diagnosis already exist. E0/E1 are required before each new study. E4 remains
a gate for diffusion-specific claims, not a prerequisite for diagnosing the
two-spin optimization problem. This dependency rule overrides a literal
interpretation that every historical single-spin experiment must be repeated.

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

Follow the dependencies and current checkpoint above. Each exit gate is a
go/no-go decision. A no-go narrows the
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

## Paper sections and figure production tracker

Reviewed: 2026-10-23. This tracker is part of the authoritative plan, not a
second task queue. Update it after each study, evidence promotion, or manuscript
revision. An existing section is not proof that its research gate is complete.
The manuscript currently contains **three figures and three scientific tables**,
all built from historical frozen evidence. The latest E5 supervision results
remain [working results](TWO_SPIN_SUPERVISION_REPORT.md), outside that snapshot.

### Article sections

Section files below are under `materials/paper/sections/`. Keep the present
narrative order; E0–E6 are experiment phases, not mandatory section headings.

| Section / source | Present state | Required integration or remaining check |
|---|---|---|
| Abstract / `abstract.tex` | Written for the bounded construction and historical diagnostics | Revise last, after final claim and evidence decisions; do not announce planned improvements |
| 1. Introduction / `introduction.tex` | Directional-data motivation, construction and bounded contribution present | Complete targeted primary-source positioning and align contributions with surviving plan C1–C5 |
| 2. Spin phase-space construction / `theory.tex` | Encoding/readout, heat transfer, CPTP reverse channels and correlated product encoding present | Add an explanatory pipeline diagram if useful (F1); document the chosen physical finite POVM and independent/shared-ancilla distinction when established |
| 3. Numerical methods and experimental design / `methods.tex` | Historical datasets, architecture, objectives, selection, metrics and pilot baselines present | Document fresh split identities, true local control, common readout, training/generation/sampling budgets and independent confirmation for every promoted study |
| 4. Results / `results.tex` | Resolution/forward physics; single-spin supervision; hybrid weighting; instruments and unsuccessful two-spin pilot present | Add controlled diffusion comparison (F5), separate correlated diagnosis (F6), physical sampling (F7), and robustness/costs (F8) only as evidence becomes available; retain negative outcomes |
| 5. Discussion and limitations / `discussion.tex` | Finite resolution, missing controls and correlated underperformance acknowledged | Reconcile each go/no-go outcome; separate representation, optimization, readout and sampling limits; no unsupported advantage or diffusion-necessity claim |
| 6. Conclusion / `discussion.tex` | Bounded historical conclusions present | Revise after the results and ledger; describe any remaining negative or inconclusive findings |
| Data/code availability, metadata and references / `discussion.tex`, `../metadata.tex`, bibliography | Draft material exists | Verify release/archive links, author information, contributions, funding, required declarations and venue details; do not invent missing information |
| Appendices A–C / `appendix.tex` | Transfer/tensor conventions; physicality/numerical evaluation; statistics/provenance present | Extend with reproducible checks and supplementary panels below, protocol/budget tables and evidence provenance |

### Figure map and dependencies

**F1–F8 are planning IDs, not current LaTeX figure numbers.** They describe
scientific questions and panels; merge or move them to the supplement as results
and venue space warrant. They do not impose eight main-text figures or new
experiments beyond E0–E6. Existing figure numbers and labels stay unchanged
until manuscript integration. “Present” below means present with historical
audited support, not completion of every broader phase.

| Planning ID / destination | Panels or scientific question | Current state and dependency |
|---|---|---|
| F1 — Construction / Theory | Spherical data → coherent encoding → exact forward diffusion → learned reverse channel from `I/d` → physical readout; distinguish training targets from generation inputs | Theory exists; explanatory drawing not produced. Label finite-rank smoothing and avoid implying an inverse of discarded modes. E2/E6; plan C1–C2 |
| F2 — Representation and forward physics / Results 4.1 | Encoded mode count versus spin; measured rank powers versus exact decay; link to smoothing/resolution checks | **Present as Figure 1, `figures/physics.pdf`**. Wider grid/threshold/family/readout robustness still pending; seed-7 threshold is not universal. E2; plan C1–C2 |
| F3 — Single-spin training diagnosis / Results 4.2 | Paired path/final errors at fixed budgets, mode recovery and complex coefficient error by rank | **Present as Figure 2, `figures/diagnostic.pdf`**. Normalized-path, exact-input stepwise and independent observations per fitted single-spin model remain open. Two-spin controls do not close these single-spin gaps. E3; plan C3–C4 |
| F4 — Objective selection and confirmation / Results 4.3 | Validation weight selection and paired confirmation of high-rank error/global trace distance | **Present as Figure 3, `figures/hybrid.pdf`**. Historical five-realization result supports a bounded mean improvement; broader family/held-out evaluation remains open. E3; plan C3–C4 |
| F5 — Contribution of diffusion / new Results subsection | Direct/one-channel, fewer-step and full-stack quality versus declared training/generation costs; learned intrinsic spherical diffusion and fixed classical comparators | Required comparison data missing. Freeze fair budgets and common targets/readout before training; final-only success alone is insufficient. E4; plan C4 |
| F6 — Two-spin correlation diagnosis / dedicated Results subsection | Independent-local/shared-ancilla/explicit-correlation topology; paired supervision effects; joint-state, marginal and connected-correlation errors versus fixed baselines | Old pilot is Table 3 only. New supervision screen/confirmation data exist in working runs, but no new paper figure or evidence promotion. Truly independent local control and controlled redesign are next; capacity studies remain conditional. E5; plan C5 |
| F7 — Physical readout and instruments / dedicated Results subsection | Fixed positive complete POVM and common target/model readout; Born samples; deterministic-versus-trajectory ensemble convergence with sampling uncertainty | Historical instrument aggregates exist; original raw conditional trajectories are absent. New finite-POVM checks, raw histories and sample validation are missing. E6; plan C2/C5 |
| F8 — Robustness, generalization and resources / Results or supplement | Quality/cost across independent realizations, families, spin, noise and mode separation/imbalance; separate encoding, learning, readout and finite-sample errors | Latest E5 working study covers five realizations of one family only. Broader evaluation, single-spin held-out observations and complete training/generation/sampling costs remain open. E3/E6; plan C3–C5 |

### Supplementary figure and table coverage

These are coverage groups, not a second experiment sequence. Include the panels
needed to substantiate the retained claims; conditional architecture work does
not become mandatory because a supplementary group exists.

| Group | Coverage and current evidence | Phase / dependency |
|---|---|---|
| S1 — Representation sensitivity | Spin, grid/quadrature and peak-threshold sweeps; angular matching, weak-mode height and raw-source versus encoded/readout smoothing. Historical spin sweep exists; expanded robustness missing | E2/E6; extends F2/F8 |
| S2 — Numerical physicality | Trace, Hermiticity, eigenvalues, Kraus completeness/Choi checks and forward-decay residuals. Historical audits exist; new architectures and physical POVM need their own checks | E0/E2/E6; supports F2/F6/F7 |
| S3 — Training diagnosis | Learning curves, gradient scales, normalized versus summed objectives, exact-input versus assembled-chain errors; per-rank complex errors. Latest two-spin working diagnostics exist; additional single-spin controls missing; do not infer gradient interference from loss curves alone | E3/E5; supports F3/F4/F6 |
| S4 — Joint structure and conditional redesign | Marginals, joint multipoles and connected correlations; distinguish source classical MI from quantum MI. Add ancilla rank, layers 3/5/7, parameter sharing, generators or pretraining panels only if a preceding diagnosis warrants those studies; include their full budget | E5; supports F6 |
| S5 — Sampling and readout sensitivity | Positive/complete product POVM, quadrature and whitening effects, finite-shot convergence, retained outcomes/conditional states; sample MMD only with verified positive-definite kernel and shot-bias treatment. Most new data missing | E6, then sample-based E5 losses; supports F7 |
| S6 — Uncertainty, error sources and cost | Per-realization paired effects, declared uncertainty, failures, held-out sample counts, encoding/learning/readout/sampling error decomposition and resource accounting. Partial historical/working data exist; complete final-model evaluation remains open | E1/E3–E6; supports F3–F8 |

Existing scientific tables are **T1 diagnostic supervision**, **T2 hybrid
selection/confirmation**, and **T3 the historical two-spin pilot**, generated
from `tables/diagnostic.tex`, `tables/hybrid.tex` and `tables/two_spin.tex`.
`tables/numbers.tex` contains generated macros, not a fourth scientific table.
Add or extend protocol/split/budget, final-comparison/resource, and
claim-to-evidence tables where needed. Do not append newer DDPM or E5 results
to the historical one-split table as if they shared its protocol.

For every numerical panel retain: protocol/run identity, source fingerprint,
target/readout definition, saved state/sample inputs, metric implementation,
individual-realization values, sample counts, uncertainty and rendering source.
Do not treat harmonics' power alone as phase/orientation accuracy, visual peaks
as a robustness test, or matched parameter counts as matched compute. Show
unequal budgets explicitly. Training, validation and confirmation must remain
separate. Working data become paper results only through a new named snapshot,
independent audit, ledger update and the assets/manuscript/PDF/archive workflow;
never edit the existing immutable evidence to fill a panel.

### Immediate next step and its expected figure output

Freeze a fresh **E5 independent local-channel control protocol**, keeping the
selected final-only supervision. Define a product prior, separate ancillas and
no inter-spin coupling or shared randomness. Retain the shared-ancilla model
as a distinct condition; the empirical factorized-marginal baseline is also
distinct from learned independent local channels. Fix datasets, train/validation
splits and unopened confirmation identities, pairing rules, budgets, classical
baselines, primary trace-distance metric, connected-correlation diagnostics,
marginal/global-state guardrails and decision thresholds before any new run.
Previously opened confirmation observations cannot select the new control.

After protocol freeze, implement and validate the control, then run its paired
validation comparison. Select and confirm only according to that protocol.
The first intended output is **F6**: topology panel, per-realization state-error
comparison, marginal errors and connected-correlation residuals. A fair negative
result is publishable evidence; it is not permission to start an uncontrolled
capacity sweep. This documentation update does not implement that control,
launch a study, produce new figures or close E5/E6.

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

- 2026-10-23: inventoried the manuscript's three existing figures, three
  scientific tables and section/appendix structure against the ledger and E0–E6.
  Added the section/figure production tracker here, including supplementary
  coverage, evidence boundaries and dependencies. No scientific result or
  manuscript asset changed, and no study was run. Next action remains: freeze
  the fresh E5 independent local-channel control protocol with final-only
  supervision.
- 2026-10-23: repository organization defines `main` as the consolidated
  development baseline, including the previously uncommitted E5 checkpoint,
  and `codex/paper` as the active paper workstream. Old branch tips are retained
  by archive tags. The streamlined paper checkout keeps code, tests, protocols,
  scientific reports, manuscript and immutable evidence; historical handoffs
  and redundant generated distributions remain recoverable from the baseline.
  Consolidated baseline: `61fd9e5` (`snapshot/development-2026-10-23`). The paper
  cleanup omits 15 recoverable historical/duplicate files, while keeping all
  scientific implementations and immutable evidence. Verification: 62 tests,
  all 1269 evidence hashes, both E5 audits, and a complete isolated manuscript
  audit/figure/PDF/package build pass. Build log:
  `outputs/tmp/reorganization-paper-build-20261023.log`.
  Scientific status is unchanged: next action is the independent local-channel
  control protocol in E5, using fresh selection/test identities.
- 2026-10-22: completed the first E5 supervision screen (20 quantum runs, five
  fixed baselines on each of five datasets), then froze and executed a
  separate confirmation using the same fitted models and previously unopened
  observations. Mean test trace distance: final-only 0.46199196 versus summed
  path 0.50198360; improvement in 5/5 datasets, all declared guardrails pass.
  Final-only remains worse than factorized marginals (0.32810992) and the
  other tested classical baselines. Source/protocol hashes, full working
  artifacts and independent replay are linked in `TWO_SPIN_SUPERVISION_REPORT.md`.
  The frozen paper evidence and its claims are unchanged; new observations
  have not been promoted. Validation and confirmation audits pass; 62 tests
  pass. Next action: freeze the genuinely independent local-channel control
  protocol using final-only supervision, with fresh selection/test identities.
- 2026-10-22: E0 replay passed (fresh audit in
  `outputs/experiments/e0_audit_20261022/`); manuscript section/claim mapping
  recorded in `materials/paper/PLAN_ALIGNMENT.md`. E1 supervision protocol
  frozen; E5 implementation adds normalized-path and exact-input stepwise
  training, paired state/gradient diagnostics, fixed baselines and independent
  NumPy/SciPy metric replay. Reduced-budget engineering check passed; the
  scientific validation screen was launched in
  `outputs/experiments/two_spin_supervision_v1/`. No new manuscript result or
  paper evidence was promoted at that checkpoint. This launch checkpoint is
  superseded by the completed validation/confirmation entry above.
- 2026-10-22: reconciled the local documentation with remote
  `codex/next-milestones` at `7656010`, preserving the superseded TODO in
  `history/DEVELOPMENT_PLAN_2026-10-19.md`. The complete code, paper evidence,
  and this plan now share the `codex/next-milestones` workstream. Repository
  alignment is documentation maintenance, not completion of E0's scientific
  audit. Next action remains E0 reconciliation and evidence integrity audit,
  followed by freezing E1 before any new run.
- 2026-10-22: plan established from the current code, handoff, TODO, paper
  ledger, and referenced conversation request. No experiment run and no new
  scientific evidence created. Next action is E0 reconciliation, then freeze
  E1 before any run.

## Compatibility and historical mapping

`notes/TODO.md` and `notes/NEXT_STEPS.md` point here for active work. Earlier
handoffs, prompts and the superseded TODO are retained in the consolidated
`main` baseline and `snapshot/development-2026-10-23`, as described in
[BRANCHES.md](BRANCHES.md), rather than in the paper checkout. Their useful
observations, equations, caveats and evidence references remain in the plan,
scientific reports and immutable evidence. Do not keep a second active queue.

Previous-roadmap mapping: repository/test foundation and audited forward
physics → E0/E2; single-spin missing-mode diagnosis and hybrid objective → E3;
two-spin supervision/objectives/capacity → E5; direct-preparation and
few-step controls → E4; physical readout, independent evaluation, evidence
promotion, and submission work → E6; hardware and large scaling → gated
out-of-scope work. Completed results remain historical evidence, not newly
scheduled prerequisites.
