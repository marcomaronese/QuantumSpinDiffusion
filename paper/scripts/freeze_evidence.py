"""Explicit, non-overwriting capture of repository evidence for this paper.

Run once from any directory. To add a later experiment, use a new named
snapshot and update the claim ledger; do not replace the present evidence.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
OUT = PAPER / 'evidence'
STUDIES = ['diagnostic_study', 'hybrid_study', 'validation_study',
           'j25_fidelity_study', 'instrument_study', 'two_spin_pilot']


def main():
    if (OUT / 'manifest.json').exists():
        raise SystemExit('Evidence is already frozen. Create a new snapshot to add results.')
    entries = []
    for study in STUDIES:
        for source in sorted((ROOT / study).rglob('*')):
            if not source.is_file() or any(x.startswith('.') for x in source.relative_to(ROOT / study).parts):
                continue
            if source.suffix not in {'.json', '.csv', '.npy', '.npz', '.pt', '.zip', '.py', '.log'}:
                continue
            target = OUT / 'raw' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            entries.append({'path': str(target.relative_to(PAPER)),
                            'source': str(source.relative_to(ROOT)),
                            'bytes': target.stat().st_size,
                            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    # Historical source is authoritative for the 70 diagnostic/hybrid runs.
    archive = ROOT / 'diagnostic_study/source_snapshot.zip'
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe snapshot member')
            dest = OUT / 'source/diagnostic' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
    # Current source is separately identified and is not represented as the
    # source that produced every historical experiment.
    current = list(ROOT.glob('spin_*.py')) + list((ROOT / 'experiments').glob('*.py'))
    current += list((ROOT / 'tests').glob('*.py')) + [ROOT / 'requirements_spin_diffusion.txt']
    for source in current:
        dest = OUT / 'source/current' / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    for p in sorted((OUT / 'source').rglob('*')):
        if p.is_file():
            entries.append({'path': str(p.relative_to(PAPER)), 'bytes': p.stat().st_size,
                            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {'snapshot_date': '2026-10-06', 'scope': 'Existing results; no training rerun during manuscript preparation',
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                'historical_source': 'source/diagnostic',
                'limitations': ['Earlier validation, instrument and two-spin runs lack a complete contemporaneous source snapshot.',
                               'Instrument archive contains aggregate CSV/JSON, not individual conditional trajectories.',
                               'Current source adds a classical DDPM absent from these historical results.',
                               'Original absolute paths in copied metadata are retained as provenance; scripts use relative paths.'],
                'files': entries}
    OUT.mkdir(exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Frozen {len(entries)} files, {sum(x["bytes"] for x in entries)/1e6:.2f} MB')


if __name__ == '__main__':
    main()
