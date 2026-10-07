# Spin phase-space quantum diffusion paper

This is a separate, self-contained manuscript project. It preserves the existing
repository manuscript and experiment outputs. The paper is written in English
and reports the completed numerical evidence, including negative results.

## Read or upload

- `manuscript.pdf`: compiled paper, including references and methodological appendices.
- `overleaf.zip`: upload with **New Project → Upload Project** in Overleaf.
  Select `main.tex` as the main document and pdfLaTeX as the compiler.
- `research_bundle.zip`: the complete project, including frozen raw evidence,
  source snapshots, regeneration scripts, and the compiled manuscript.
- `metadata.tex`: replace the author/affiliation field before submission.
- `CLAIMS.md`: claim-to-evidence index and boundaries on interpretation.
- `LITERATURE.md`: verified reference sources and scope of the literature search.
- `SUBMISSION_NOTES.md`: remaining research and author-supplied information.

The Overleaf ZIP contains all typesetting dependencies, vector figures, BibTeX,
and compact result tables. It does not require Python, PyTorch, shell escape, an
external bibliography service, or any path outside the uploaded project. Raw
binary experiment evidence is provided in the complete research bundle rather
than burdening the Overleaf editor with more than a thousand experiment files.
No upload, public publication, or journal submission has been performed.

## Structure

```text
main.tex, metadata.tex, references.bib, latexmkrc
sections/        Full manuscript and appendices
figures/         Reproducible vector PDF figures and PNG previews
tables/          Generated LaTeX tables and numerical macros
results/         Audited per-run CSV, JSON summaries, verification records
evidence/
  manifest.json  Original relative locations and SHA-256 checksums
  raw/           Frozen experiment matrices, parameters, metrics and protocols
  source/
    diagnostic/  Exact historical source from the completed 70-run studies
    current/     Separately labelled current source; includes later DDPM work
scripts/         Freeze, audit, figure/table generation, and ZIP packaging
build/           Local compilation logs (not included in deliverable archives)
```

## Rebuild locally

From the repository root, using its existing Python environment:

```bash
make -C paper all
```

From an extracted research bundle, install the scientific dependencies from
`evidence/source/current/requirements_spin_diffusion.txt` in an environment of
your choice, then run `make PYTHON=/path/to/python all`. NumPy, SciPy, PyTorch
and Matplotlib are required for auditing/figures, and a standard TeX Live
installation with pdfLaTeX and BibTeX is required to compile the paper.
Exact versions recorded by the main experiments were Python 3.12.14,
PyTorch 2.14.1+cpu, NumPy 2.5.3, SciPy 1.18.1, and Matplotlib 3.11.2. These are
recorded environment facts, not a guarantee that every version is available
from your package index. No retraining is necessary to rebuild the paper.

Individual steps, from this folder:

```bash
python scripts/audit_evidence.py
python scripts/build_assets.py
make pdf
python scripts/package.py
```

The audit independently recomputes state metrics for 70 single-spin runs and
six two-spin outputs. It also checks frozen-file hashes, physicality, exact
forward decay, saved parameter replay, Kraus completeness, mode recovery, and
rank errors. Tensor generation, peak finding and channel replay reuse the
historical implementation; the audit describes that limitation explicitly.
Historical instrument statistics are retained, but individual instrument
trajectories were not available to re-audit. See `results/audit.json`.

## Preserve and add evidence

The evidence snapshot is intentionally non-overwriting. Do not edit files under
`evidence/raw/` or `evidence/source/`; they are checked against `manifest.json`.
`scripts/freeze_evidence.py` records the initial copy operation and refuses to
replace an existing snapshot. See `RESULTS_WORKFLOW.md` for adding new studies.

Numbers in the three main tables and numerical macros are generated from the
audited per-run CSV. The other numerical examples in the prose are rounded
descriptions of that fixed snapshot; changing the evidence requires reviewing
the narrative as well as regenerating assets.

## Research status

This is a complete research manuscript draft, not a claim of submission
readiness. It deliberately avoids a “first fully quantum diffusion” claim,
quantum advantage, demonstrated hardware sampling, and generalization beyond
the studied synthetic data. Author names and institutional declarations cannot
be inferred from repository paths. The two-spin failure and outstanding
classical/direct-preparation comparisons are part of the paper, not omitted
results.
