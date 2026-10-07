"""Produce a minimal standalone Overleaf ZIP and a full research bundle."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DOCS={'README.md','CLAIMS.md','LITERATURE.md','RESULTS_WORKFLOW.md','SUBMISSION_NOTES.md'}


def main():
    core=[p for p in ROOT.iterdir() if p.name in {'main.tex','metadata.tex','references.bib','latexmkrc'}|DOCS]
    core += sorted((ROOT/'sections').glob('*.tex')) + sorted((ROOT/'tables').glob('*.tex'))
    core += sorted((ROOT/'figures').glob('*.pdf'))
    core += sorted((ROOT/'results').glob('*.json')) + sorted((ROOT/'results').glob('*.csv'))
    core=[p for p in core if p.name!='packages.json']
    with zipfile.ZipFile(ROOT/'overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(core): z.write(p,p.relative_to(ROOT))
    # Explicitly omit transient output and ZIPs, preventing recursive archives.
    files=[p for p in ROOT.rglob('*') if p.is_file()
           and p.name not in {'overleaf.zip','research_bundle.zip','packages.json'}
           and not any(part in {'build','tmp','__pycache__'} for part in p.relative_to(ROOT).parts)]
    with zipfile.ZipFile(ROOT/'research_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files): z.write(p,p.relative_to(ROOT))
    info={name:{'bytes':(ROOT/name).stat().st_size,'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}
          for name in ('overleaf.zip','research_bundle.zip')}
    (ROOT/'results/packages.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__': main()
