# Repository branches

The repository uses two branches from 2026-10-23:

| Branch | Purpose |
|---|---|
| `main` | Consolidated development baseline: all previous code, reports, manuscript, evidence, and completed supervision study. Preserve this reference when starting the paper workstream. |
| `codex/paper` | Active paper workstream, derived from the consolidated `main`; contains the focused paper repository and follows `SPINDIFFUSION_PAPER_PLAN.md`. |

The only development queue is
[`SPINDIFFUSION_PAPER_PLAN.md`](SPINDIFFUSION_PAPER_PLAN.md). The manuscript's
supported claims remain defined by
[`materials/paper/CLAIMS.md`](../materials/paper/CLAIMS.md).

## Retired branches and recovery

Earlier branch tips are retained as Git tags before their branch names are
retired. Tags preserve their original commits and complete ancestry:

| Former branch | Archive tag |
|---|---|
| `main` before consolidation | `archive/2026-10-23/main` |
| `codex/next-milestones` | `archive/2026-10-23/codex-next-milestones` |
| `codex/spindiffusion-paper-plan` | `archive/2026-10-23/codex-spindiffusion-paper-plan` |
| `research/spindiffusion-paper-roadmap` | `archive/2026-10-23/research-spindiffusion-paper-roadmap` |

`snapshot/development-2026-10-23` identifies the consolidated development
baseline, including the previously uncommitted E5 implementation and reports.
The earlier paper-plan branch contains only superseded documentation; its
history is joined into the baseline after its scientific plan content has
been incorporated into the current canonical plan. The join retains the
current package layout and avoids reintroducing flat prototype entry points.

Historical handoffs, supporting theory notes, duplicate dependency lists, and
previous distribution bundles remain in the consolidated `main` snapshot.
The active paper checkout keeps their useful findings in the plan, reports,
claim ledger and immutable evidence, and regenerates distribution ZIPs as
needed. Files absent from the paper branch are recoverable from the baseline
tag; no history is rewritten.

Local generated runs remain under ignored `outputs/experiments/` directories.
They are not removed by branch cleanup and are not automatically published as
paper evidence. The recent supervision results still require the named-
snapshot promotion workflow before becoming manuscript results.
