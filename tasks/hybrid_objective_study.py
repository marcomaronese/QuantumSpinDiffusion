#!/usr/bin/env python3
"""Validation-only hybrid selection followed by untouched-seed confirmation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

from tasks.reverse_diagnostic import ROOT, execute, source_digest

VALIDATION_SEEDS = [17, 19, 23, 29, 31]
TEST_SEEDS = [41, 43, 47, 53, 59]
LAMBDAS = [0.0, 0.25, 0.5, 0.75]


def choose_lambda(rows):
    """Guard global metrics, then maximize mode recovery and minimize rank error."""
    groups = {weight: [r for r in rows if r['hybrid_lambda'] == weight] for weight in LAMBDAS}
    baseline = groups[0.0]
    if not baseline or any({r['seed'] for r in g} != set(VALIDATION_SEEDS) for g in groups.values()):
        raise ValueError('Selection requires every predeclared validation run')
    limits = {'infidelity': .005, 'trace_distance': .01, 'q_js_divergence': .001}
    eligible = []
    for weight, group in groups.items():
        if all(statistics.fmean(r[m] for r in group) <= statistics.fmean(r[m] for r in baseline) + delta
               for m, delta in limits.items()):
            eligible.append(weight)
    def score(weight):
        group = groups[weight]
        return (-sum(r['target_mode_count'] == 2 and r['all_target_modes_recovered'] for r in group),
                statistics.fmean(sum(r['ranks'][str(ell)]['squared_coefficient_error'] for ell in (4, 5)) for r in group),
                weight)
    selected = min(eligible, key=score)
    return {'selected_lambda': selected, 'eligible_lambdas': eligible,
            'validation_modes_recovered': -score(selected)[0], 'guardrail_absolute_deltas': limits,
            'selection_uses_test_results': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--diagnostic',
        type=Path,
        default=ROOT / 'outputs' / 'experiments' / 'diagnostic_study',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=ROOT / 'outputs' / 'experiments' / 'hybrid_study',
    )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    # Enforce ordering: complete the 2x2 study at two budgets before starting.
    manifest = json.loads((args.diagnostic / 'manifest.json').read_text())
    diagnostic = json.loads((args.diagnostic / 'results.json').read_text())
    if len(diagnostic) != len(manifest['jobs']) or len(diagnostic) < 40:
        raise ValueError('Complete the five-seed diagnostic first')
    if len({r['epochs'] for r in diagnostic}) != 2 or len({(r['starting_state'], r['supervision']) for r in diagnostic}) != 4:
        raise ValueError('Diagnostic must contain the 2x2 comparison and budget check')
    if set(VALIDATION_SEEDS + TEST_SEEDS) & {r['seed'] for r in diagnostic}:
        raise ValueError('Hybrid validation and test seeds must be fresh')
    protocol = {'validation_seeds': VALIDATION_SEEDS, 'test_seeds': TEST_SEEDS, 'lambdas': LAMBDAS,
                'epochs': args.epochs, 'start': 'mixed', 'supervision': 'final',
                'reason': 'Generative prior with objective aligned directly to the final target',
                'high_rank_loss': 'existing mean-one weights proportional to (ell+1)^2 across all ranks',
                'selection': 'guard mean infidelity (+.005), trace distance (+.01), Q JS (+.001) vs lambda=0; '
                             'maximize both-mode recovery; tie-break by ell=4+5 squared coefficient error, then smaller lambda',
                'success_gate': 'at least 4/5 validation and 4/5 test successes, physicality and decay within 1e-10, guardrails on each split',
                'source_sha256': source_digest(),
                'selection_code_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()}
    args.outdir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.outdir / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing hybrid protocol differs; use a fresh output directory')
    protocol_path.write_text(json.dumps(protocol, indent=2))
    def jobs(seeds, weights):
        return [dict(seed=seed, epochs=args.epochs, starting_state='mixed', supervision='final',
                     objective='hybrid', hybrid_lambda=weight) for weight in weights for seed in seeds]
    validation = execute(args.outdir / 'validation', jobs(VALIDATION_SEEDS, LAMBDAS), protocol, args.workers)
    selection = choose_lambda(validation)
    selection_path = args.outdir / 'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Frozen selection differs')
    selection_path.write_text(json.dumps(selection, indent=2))
    test = execute(args.outdir / 'test', jobs(TEST_SEEDS, sorted({0.0, selection['selected_lambda']})), protocol, args.workers)
    chosen = [r for r in test if r['hybrid_lambda'] == selection['selected_lambda']]
    baseline = [r for r in test if r['hybrid_lambda'] == 0.0]
    physical = all(r['max_trace_error'] < 1e-10 and r['minimum_eigenvalue'] >= -1e-10
                   and r['max_trace_preservation_error'] < 1e-10 and r['minimum_choi_eigenvalue'] >= -1e-10
                   and r['max_multipole_decay_error'] < 1e-10 for r in validation + test)
    guardrails = all(statistics.fmean(r[m] for r in chosen) <= statistics.fmean(r[m] for r in baseline) + delta
                     for m, delta in selection['guardrail_absolute_deltas'].items())
    successes = sum(r['target_mode_count'] == 2 and r['all_target_modes_recovered'] for r in chosen)
    result = {**selection, 'test_modes_recovered': successes, 'test_guardrails_pass': guardrails,
              'physicality_and_forward_decay_pass': physical,
              'single_spin_gate_pass': selection['validation_modes_recovered'] >= 4 and successes >= 4 and physical and guardrails}
    (args.outdir / 'confirmation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
