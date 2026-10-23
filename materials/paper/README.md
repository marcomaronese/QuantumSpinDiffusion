# SpinDiffusion manuscript

The article is a construction and controlled diagnostic study, including the
negative two-spin pilot. It is a research draft; its scope is defined by the
audited evidence, not by planned experiments.

| File | Purpose |
|---|---|
| [manuscript.pdf](manuscript.pdf) · [main.tex](main.tex) | Read the article or edit its source |
| [CLAIMS.md](CLAIMS.md) | Exact claim-to-evidence ledger and limitations |
| [PLAN_ALIGNMENT.md](PLAN_ALIGNMENT.md) | Article-to-plan mapping and remaining evidence gaps |
| [LITERATURE.md](LITERATURE.md) | Primary sources and novelty-review boundaries |
| [RESULTS_WORKFLOW.md](RESULTS_WORKFLOW.md) | Audit and promote new evidence |
| [SUBMISSION_NOTES.md](SUBMISSION_NOTES.md) | Publication context; the task queue is the official plan |

## Build and export

From the repository root:

```bash
make -C materials/paper PYTHON=../../.venv/bin/python all
```

This audits frozen results, generates figures/tables, compiles with pdfLaTeX
and BibTeX, and produces `overleaf.zip` and `research_bundle.zip`. The ZIPs,
their package hashes and PNG figure previews are local generated products;
they are ignored by Git instead of storing duplicate distributions in source.
Use `make -C materials/paper package` to package the current built manuscript.

`overleaf.zip` contains the standalone typesetting dependencies: upload it as
a new Overleaf project and select `main.tex`. The research bundle also contains
the evidence and regeneration scripts. To rebuild an extracted bundle, install
its scientific dependencies from
`evidence/source/current/requirements_spin_diffusion.txt`, then run
`make PYTHON=/path/to/python all`. A TeX installation is needed for compilation.

## Files and evidence

```text
sections/    Article and appendices
figures/     Vector PDF figures; optional PNG previews are generated
tables/      Generated LaTeX tables and numerical macros
results/     Audit, per-run metrics and paper summary
evidence/    Manifest plus immutable raw/ and source/ snapshots
scripts/     Audit, figure/table generation and packaging
```

Never edit `evidence/raw/` or `evidence/source/`. The audit checks all 1269
manifest entries and replays the 70 historical single-spin runs and six
correlated-pilot outputs. Some diagnostics reuse historical tensor/peak/channel
code; individual historical instrument trajectories were unavailable. See
`results/audit.json` and `CLAIMS.md` for the exact scope.

The original one-time capture script and historical distribution verification
are preserved in the consolidated development baseline described in
[`notes/BRANCHES.md`](../../notes/BRANCHES.md). New evidence uses a separate
named snapshot under the results workflow. Working outputs are not silently
added to the article. Authors, declarations and venue remain author decisions;
no quantum advantage or hardware result is established.
