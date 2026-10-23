# Manuscript alignment with the authoritative paper plan

Reviewed: 2026-10-22. This is a manuscript assessment, not an alternative task
list. Active priorities and the restart action remain in
[`notes/SPINDIFFUSION_PAPER_PLAN.md`](../../notes/SPINDIFFUSION_PAPER_PLAN.md).

## Finding

The present article follows the scientific narrative of the official plan for
the evidence already acquired. It is a construction and diagnostic paper with
an unsuccessful correlated pilot. It is not yet the completed paper envisaged
after all E0–E6 gates pass. The plan originally specified claim themes and an
experiment sequence, rather than mandatory manuscript section titles; the
mapping below makes that correspondence explicit.

The existing section order is Introduction → spin phase-space construction →
numerical methods and design → results → discussion/limitations → conclusion,
followed by availability, references, and appendices. This order is suitable:
E0–E6 are research phases, not seven compulsory article sections.

## Narrative and evidence map

Plan theme IDs C1–C5 and ledger IDs C1–C9 are separate namespaces. Do not change
the ledger IDs or present a plan theme as an independently verified result.

| Official-plan theme | Present article location | Ledger support | Remaining scope limitation |
|---|---|---|---|
| Plan C1: representation and heat transfer | Theory: encoding/readout and exact spherical heat evolution; Results: resolution/forward dynamics; readout-transfer appendix | Ledger C1–C2 | Resolution threshold is tied to the one dataset, grid, and peak criterion; no universal minimum spin. |
| Plan C2: forward dynamics and physicality | Theory: physical reverse maps; Results: forward diagnostics; numerical-physicality appendix | Ledger C3 | Audited small exact simulations; no hardware validation. |
| Plan C3: reconstruction and training | Methods: supervision/objectives and confirmation protocol; Results: missing-mode diagnostic and multipole weighting | Ledger C2, C4–C6 | Single-spin confirmation seeds are independent realizations, not held-out observations for each fitted model; normalized/stepwise controls and broader families remain open. |
| Plan C4: contribution of diffusion | Methods distinguishes final-only from path supervision; Results and Discussion explicitly limit interpretation | Ledger C4–C5 support diagnostics only; necessity claim excluded | Direct preparation, fewer-step controls, and learned intrinsic spherical diffusion comparisons are absent. No diffusion-specific benefit is established. |
| Plan C5: correlated extension and sampling | Theory: instruments/product phase spaces; Methods: two-spin pilot/classical comparators; Results: instrument consistency and correlated limitation | Ledger C7–C9 | Original two-spin pilot has one split and unequal budgets, and underperforms classical comparators; no truly independent local-channel ablation, physical finite-POVM study, or independent correlated confirmation. |

## What the current article already states accurately

- Methods distinguishes composed path training from teacher forcing and notes
  that its loss is a sum, not normalized.
- The article labels the two-spin no-direct model as shared-ancilla; it is not
  a factorized-channel control.
- The current ambient DDPM is described as a later implementation with no
  results in the frozen paper evidence; it is not silently inserted into the
  historical comparison.
- Discussion names direct preparation, normalized supervision, and learned
  spherical diffusion as missing comparisons. It does not claim a necessity
  of six steps or coarse-to-fine learned recovery.
- The correlated target can be separable; quantum mutual information does not
  establish entanglement or source classical mutual information.

## Audit and integration boundary

The 2026-10-22 replay verified all 1269 manifest entries, 70 single-spin runs,
and six correlated-pilot model metrics. Maximum independently recomputed
state-metric discrepancies were approximately 2.69e-14 for single spin and
2.08e-13 for the two-spin pilot. Fresh output is under
`outputs/experiments/e0_audit_20261022/`; the immutable snapshot was not edited.
Tensor/peak/channel replay reuses historical source and must not be described
as an independent implementation of every diagnostic. Historical instrument
trajectory aggregates were retained rather than rerun.

The new E5 supervision screen and its subsequent independent-observation
confirmation are recorded in the
[working results report](../../notes/TWO_SPIN_SUPERVISION_REPORT.md).
They have not been promoted into the claim ledger or manuscript results. When
promotion is warranted, follow `RESULTS_WORKFLOW.md`: named checksummed
addition, independent audit, claim update, generated assets, manuscript,
rendered PDF, and both archives together. No textual claim of completed E4,
E5, or E6 should precede that evidence.
