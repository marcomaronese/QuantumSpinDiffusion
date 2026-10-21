# Archived development plan — 2026-10-19

Historical record only. Active priorities and restart instructions are in
[SPINDIFFUSION_PAPER_PLAN.md](../SPINDIFFUSION_PAPER_PLAN.md). The original
plan below is preserved for scientific context; its task queue is superseded.

---

# Development TODO — primary reference

Last updated: 2026-10-19. This is the single authoritative development plan.
Read it before research changes; update it after each development session.
`README.md` defines paths; `materials/paper/CLAIMS.md` defines supported claims.
Historical roadmaps and the theory document do not override this queue.
Unchecked items are proposals, not implemented results.

## Restart here

**Next action: P0, then P1 — declare and implement the controlled two-spin
supervision comparison before changing its architecture.** This checkpoint
changes documentation only; no new experiment or model change was performed.

- Continue on `codex/next-milestones`; do not assume `main` contains the work.
- Check local changes and live remote heads. Before this documentation update,
  on 2026-10-19, GitHub and local HEAD both pointed to
  `aa434e72eb07ef0d13b803d62923e208ab523dd5`; `main` remained at initial commit
  `c37888145c9f6cb69db7137af259c2b7f23cc536`, six commits behind.
- Read `DIAGNOSTIC_REPORT.md`, `../materials/paper/CLAIMS.md`,
  `../src/spin_quantum_diffusion/quantum/correlated.py`, and
  `../tasks/two_spin_correlated.py` for the immediate experiment.
- Use a new `outputs/experiments/two_spin_supervision_<protocol-id>/` directory;
  preserve historical outputs and frozen evidence.
- Qiskit, hardware, image latents and large scaling studies remain deferred.
  Do not automatically launch the entire backlog.

Checkpoint synchronization: this session's host exposes `.git` read-only and
SSH cannot resolve GitHub. Remote publication uses the GitHub connector instead
of local Git writes. Local branch/index metadata may therefore still show the
pre-checkpoint commit and documentation changes; compare the live remote and
file contents before recommitting. Reconcile the checkout with normal Git when
its metadata is writable, preserving any subsequent local work.

## Completed foundation and interpretation limits

- [x] Stabilize package layout, study entry points, tests and output boundaries.
- [x] Validate forward multipole decay and small-system physicality/CPTP.
- [x] Complete the 40-run single-spin supervision/start/budget diagnostic:
  at 100 epochs final-only recovers both modes in 5/5 seeds versus 1/5 for path
  supervision; at 500 epochs all four conditions pass.
- [x] Complete mode/rank diagnostics and hybrid validation/confirmation:
  `lambda=0.75` improves mean high-rank error and trace distance on five new
  seed realizations; both objectives recover 5/5 modes. These are reconstruction
  results, not held-out evaluation of a frozen fitted model.
- [x] Implement collision channels and sampled ancilla instruments. Archived
  instrument evidence covers aggregates; original raw trajectories are incomplete.
- [x] Complete the two-spin pilot and audit: the quantum models underperform
  the tested correlation-aware classical baselines.
- [x] Produce a complete manuscript draft and checksummed evidence archive.

The old conditional single-spin capacity expansion was not triggered. Do not
repeat it as a prerequisite. A shared ancilla can correlate spins without direct
spin-spin gates: the current no-direct model is not a factorized baseline.

## Ordered development queue

Main sequence: P0 → P1 → P2 → P3 → P4 → P5. P6 is a separate single-spin
workstream after P0; P7 accompanies all implementation. P8 is gated. P9 covers
publication. Complete P4 before claiming a benefit from diffusion structure.
The sample-MMD extension in P2 is deferred until P5 readout exists and does not
block P3; the state/observable objectives can be evaluated first.

### P0 — Freeze the next protocol

- [ ] Declare dataset sizes/families, spin, noise, independent train/validation/
  test samples and seed roles, paired initializations, optimizer/update and
  compute budgets, primary metrics, selection, guardrails and stopping rules.
- [ ] Separate encoding/readout resolution error, learned-state error and
  finite-sample error. Use the same encoding/readout across models; report
  discrepancies from raw classical data separately.
- [ ] Hold factorized, spherical heat-kernel, low-bond tensor-mixture and neural
  baselines fixed during controlled comparisons. Include the current DDPM in
  new studies, but do not call it intrinsic Riemannian diffusion or retroactively
  insert it into frozen historical results.
- [ ] Save protocol and source fingerprint before training. Select on validation
  only and keep confirmation tests closed. Changed protocols need new identities.

### P1 — Diagnose two-spin supervision with fixed architecture

- [ ] Compare final-only fidelity, existing summed path loss and normalized path
  loss, pairing targets/initializations and recording gradient scales and updates.
  Normalization alone does not disentangle gradient interference.
- [ ] Compare stepwise training on exact forward inputs with composed-chain
  training. Evaluate the assembled chain from the generative prior to expose
  accumulated error; teacher-forced step accuracy is insufficient.
- [ ] Evaluate optional stepwise pretraining plus final-only fine-tuning, counting
  its extra training budget explicitly.
- [ ] Use exact finite-time `rho_T` only as a diagnostic where needed; always
  report generation from `I/d`. Never supply hidden forward randomness.
- [ ] Save global/state metrics, local and joint multipole errors, connected
  correlations, quantum MI, losses, parameters, trajectories, physicality/CPTP
  residuals and timings. Quantum MI is not source classical MI.

Exit: paired validation results select supervision before capacity changes.
Retain failed runs and avoid inferring a general expressivity limit.

### P2 — Align the objective with joint structure

- [ ] At selected supervision, add local multipole and cross-spin correlator
  losses one factor at a time, retaining global fidelity/trace-distance guardrails.
  Matching a restricted observable set is not full-state reconstruction.
- [ ] Select weights on validation; do not inherit single-spin `lambda=0.75`.
- [ ] Add sample MMD only after a common readout is defined (P5), using a verified
  positive-definite spherical/product-sphere kernel and accounting for shot bias.

Exit: correlation improvement passes global guardrails and cannot be explained
only by better marginals or a changed evaluation representation.

### P3 — Controlled ancilla and capacity redesign

- [ ] Establish truly independent local channels: separate ancillas, product
  prior, no shared randomness/coupling. Retain the old shared-ancilla condition.
- [ ] Compare local ancillas plus a controlled correlation channel/ancilla.
- [ ] Compare local-channel pretraining followed by correlation-channel training.
- [ ] If necessary, vary Kraus rank/ancilla count, layers (3/5/7), parameter
  sharing, local quadratic controls and richer cross-spin generators separately.
  `Jz1Jz2` and `Jx1Jx2 + Jy1Jy2` are already implemented.
- [ ] Record trainable/effective parameters, compute and generation costs;
  match budgets where possible and report mismatches.

Select architecture on validation before confirmation. Capacity increases are
conditional on P1/P2, not an automatic large hyperparameter sweep.

### P4 — Test the contribution of diffusion

- [ ] Compare direct state preparation or a single learned channel, fewer reverse
  steps and the six-step stack using common targets/readout and documented
  parameter/optimization budgets. Include a single-spin control.
- [ ] Include normalized path supervision and selected final/stepwise training.
- [ ] Add a learned intrinsic spherical diffusion baseline; the existing
  heat-kernel bootstrap is not a learned reverse diffusion model.
- [ ] Report quality, costs and uncertainty. Retain negative results if direct
  preparation matches or wins; narrow the diffusion interpretation accordingly.

### P5 — Physical readout and independent correlated confirmation

- [ ] Build a finite coherent-state POVM with positive weights and complete
  logical support; verify positivity, completeness and Born sampling. Quantify
  how `S^(-1/2) A_k S^(-1/2)` changes Husimi readout. State-dependent grid
  renormalization is not itself a fixed POVM.
- [ ] Use the same POVM on generated/target states, validate quadrature resolution
  and extend to joint samples through local product POVMs.
- [ ] Reuse instrument routines; save conditional states, probabilities, outcome
  histories and final samples. Check ensemble/readout convergence against the
  deterministic channel before making new trajectory claims.
- [ ] Freeze the selected protocol/model and confirm on 5–10 independent
  train/test realizations, with held-out data per fitted model, paired baselines,
  uncertainty, sample counts and training/generation/sampling costs.
- [ ] Independently recompute primary state/sample metrics from saved artifacts;
  include marginals, connected correlations, MI and physicality/CPTP.

Ignoring ancilla outcomes leaves the unconditional channel unchanged. Instruments
alone do not restore discarded harmonics or improve its final marginal. Feedback
conditioned on outcomes is a separate optional future architecture.

### P6 — Single-spin generalization and representation robustness

- [ ] Evaluate frozen fitted models on independent evaluation samples; separate
  validation from final tests and extend beyond one synthetic distribution family.
- [ ] Vary mode separation, imbalance, noise, spin, grid and peak thresholds.
  Separate representation failures from optimization failures.
- [ ] Retain angular mode matching, weaker-mode height, per-rank complex
  coefficient errors and global metrics. The `j=2.5` threshold is dataset-specific.
- [ ] Quantify encoding/readout smoothing before higher-spin or latent proposals.
  Neither instruments nor recovery channels invert arbitrary finite-spin loss.

### P7 — Reproducibility and new paper evidence

- [ ] Save source/software metadata, datasets/splits, seeds, histories, parameters,
  states/samples, residuals, costs and failed-run/exclusion records per protocol.
- [ ] Run `.venv/bin/python -m pytest -q` after code changes; invoke studies as
  modules at repository root. Do not repair invalid outputs by PSD projection.
- [ ] Promote only audited results into named checksummed snapshots under
  `materials/paper/evidence/additions/`; never edit frozen `raw/` or `source/`.
- [ ] Follow `../materials/paper/RESULTS_WORKFLOW.md`: update claim ledger,
  audit/assets, manuscript interpretation, PDF and both ZIPs together.

### P8 — Deferred execution and scaling

Hardware gate remains closed: require reliable single-spin recovery, material
improvement over a truly factorized baseline, approach to correlation-aware
classical baselines across independent splits, CPTP without output repair, and
documented resource/sampling costs.

- [ ] Before Qiskit, validate abstract random collective-rotation convergence
  to the exact Lindblad channel, with time-step and Monte Carlo error separated.
- [ ] Once gated, prepare the prior using uniform coherent-state ensembles;
  verify `I_j/(2j+1)` in the symmetric sector, not full qubit maximally mixed states.
- [ ] Verify gate decompositions against exact operators, including generator
  normalization/angles. Track resets, `J^2` leakage, logical/transpiled depth,
  two-qubit gates, SWAPs, shots and timing.
- [ ] Use `Jz` readout only as a discrete smoke test; implement the selected
  spherical POVM with explicit dilation/readout resource costs.
- [ ] Progress exact pretraining → shot/noise simulation → limited hardware
  fine-tuning, retaining calibration snapshots and raw/mitigated metrics.
- [ ] After correlated success, scale spin count/correlation complexity at fixed
  accuracy against trajectories, genuine MPS/MPO methods and strong classical
  generators. The current tensor mixture is not a general TN benchmark.
- [ ] Introduce image/field latent encoders/decoders only after correlated-core
  validation, including readout and classical decoding costs.

Optional Petz reference: small-system diagnostic/teacher only, with explicit
training-derived reference state and support. Recovering that reference from its
forward image is not universal inversion, generalization or guaranteed generation
from `I/d`. Never construct it from held-out evaluation targets.

### P9 — Publication readiness

- [ ] Extend the targeted primary-source novelty review beyond
  `../materials/paper/LITERATURE.md`; avoid unsupported priority claims.
- [ ] Supply authors, affiliations, corresponding author, funding, contributions,
  competing interests, required AI declarations and venue/style.
- [ ] Designate public code/data archive and permanent DOI where appropriate;
  verify availability claims. Pushing code is not journal submission.
- [ ] Choose scope: the construction/diagnostic paper can retain current limits;
  stronger generative, diffusion, correlated, hardware or advantage claims need
  the corresponding evidence above.

## Maintenance and source mapping

After each session record completed IDs, artifact/commit references, decisions,
blockers and the next action here. Do not mark research complete just because
an API/task exists. Historical reports remain history, not competing queues.

| Previous source | Consolidated destination |
|---|---|
| `NEXT_STEPS.md` steps 1–3 | Completed foundation; P6 for new generalization |
| Old conditional capacity step 4 | Not triggered for original single-spin result; conditional P3 |
| Old two-spin steps 5–6 | P0–P3 and P5 |
| Old hardware step 7 | P8, still gated |
| Paper submission research gaps | P4, P6, P7, P9 |
| Theory §§5–9 and appendix B | P1–P3, P5, P8 |
| Theory §§10–13 | P6 representation limits and P8 deferred scaling/latents |

Theory reviewed on 2026-10-19: *Spin Phase-Space Quantum Diffusion Models*,
technical project document dated 2026-10-06, 39 PDF pages, supplied as
`spin_phase_space_quantum_diffusion.pdf`. It supplies proposals, not new evidence.
The decisions needed to restart are recorded here without requiring attachment
access. No research results or immutable evidence were changed at this checkpoint.
