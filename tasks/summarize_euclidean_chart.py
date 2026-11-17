"""Document audited chart decisions and a post-hoc annulus-support diagnostic."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import scipy

from .euclidean_chart_study import ROOT, digest, verify_manifest, write


def summarize(screen, confirmation, output):
    if output.exists() or not output.is_relative_to(ROOT / "outputs/experiments"):
        raise ValueError("use a fresh outputs/experiments directory")
    for directory in (screen, confirmation):
        verify_manifest(directory)
        audit=json.loads((directory/"audit.json").read_text())
        if not audit["passed"] or audit["manifest_sha256"]!=digest(directory/"manifest.json"):
            raise AssertionError("audited parent required")
    rows=json.loads((confirmation/"results.json").read_text())
    selection=json.loads((confirmation/"selection.json").read_text())
    output.mkdir(parents=True)
    decisions={}
    lines=["## Numerical results", "",
        "The six 2D laws use fresh calibration, validation and closed test data,",
        "three realizations and four independent continuous-POVM batches per case.",
        "The screen has 378 cases (7 maps x 3 spins x 6 laws x 3 realizations);",
        "confirmation has 162 cases with scales frozen at j=5/2. No reverse",
        "channel was trained. Error is 32-direction sliced W1 in calibration-",
        "standardized units, mean +/- sample SD across the three realizations.", "",
        "| Target | Selected sinh scale | Old map | Ordinary stereo | Sinh | Reduction vs old | Real vs real |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for target,choice in selection.items():
        group=[r for r in rows if r["target"]==target and r["two_j"]==5]
        by_kind={kind:[r for r in group if r["map"]["kind"]==kind]
                 for kind in ("bounded","stereographic","sinh_stereographic")}
        scores={kind:np.array([r["sliced_w1"] for r in group]) for kind,group in by_kind.items()}
        improvement=1-scores["sinh_stereographic"].mean()/scores["bounded"].mean()
        paired=all(next(c for c in by_kind["sinh_stereographic"] if c["seed"]==b["seed"])["sliced_w1"]<=1.05*b["sliced_w1"]
                   for b in by_kind["bounded"])
        tails=all(r["tail_ratio_995"]<=1.5 for r in by_kind["sinh_stereographic"])
        passed=bool(choice["candidate_eligible"] and improvement>=.1 and paired and tails)
        decisions[target]={"selected_sinh":choice["selected"][-1],"mean_w1_reduction":float(improvement),
            "passed_predeclared_adapter_gate":passed,"paired_guardrail":paired,"upper_tail_guardrail":tails,
            "maximum_sinh_q995_ratio":max(r["tail_ratio_995"] for r in by_kind["sinh_stereographic"]),
            "maximum_stereo_q995_ratio":max(r["tail_ratio_995"] for r in by_kind["stereographic"])}
        formatted=[f"{scores[k].mean():.3f} +/- {scores[k].std(ddof=1):.3f}" for k in scores]
        floor=np.mean([r["real_reference"]["sliced_w1"] for r in by_kind["bounded"]])
        lines.append(f"| {target} | {choice['selected'][-1]['length']:g} | "+" | ".join(formatted)+f" | {100*improvement:.1f}% | {floor:.3f} |")
    lines += ["", "All six preselected sinh candidates pass the declared *relative adapter* gate:",
        "at least 10% mean error reduction, no paired realization worse by >5%,",
        "and the one-sided 0.995 radial upper-tail inflation limit. This gate is",
        "not a statement of faithful distribution recovery. In particular it does",
        "not test two-sided tail calibration or preservation of holes and modes.", "",
        "Ordinary stereo improves some central features but produces excessive",
        "tails: its maximum empirical 0.995 radius ratios range from "+
        f"{min(d['maximum_stereo_q995_ratio'] for d in decisions.values()):.2f} to {max(d['maximum_stereo_q995_ratio'] for d in decisions.values()):.2f} across targets.",
        "Its theoretical infinite second moments also remain, regardless of a",
        "finite empirical covariance. No tail trimming was used.", "",
        "## Fixed-map resolution controls", "",
        "The scales below remain those selected at j=5/2; they were not retuned",
        "at larger j. These are representation-only controls, not trained model",
        "comparisons or evidence for a universal minimum spin.", "",
        "| Target | Sinh j=2.5 | Sinh j=5 | Sinh j=10 |", "|---|---:|---:|---:|"]
    for target in selection:
        values=[np.mean([r["sliced_w1"] for r in rows if r["target"]==target
                        and r["two_j"]==two_j and r["map"]["kind"]=="sinh_stereographic"]) for two_j in (5,10,20)]
        lines.append("| "+target+" | "+" | ".join(f"{v:.3f}" for v in values)+" |")
    ring={}
    for two_j in (5,10,20):
        ring[str(two_j)]={}
        for kind in ("bounded","stereographic","sinh_stereographic"):
            group=[r for r in rows if r["target"]=="ring" and r["two_j"]==two_j and r["map"]["kind"]==kind]
            masses=[float(np.mean(np.linalg.norm(np.load(confirmation/r["artifact"])["decoded"],axis=-1)<1.)) for r in group]
            ring[str(two_j)][kind]={"mean":float(np.mean(masses)),"sd":float(np.std(masses,ddof=1))}
    reference_group=[r for r in rows if r["target"]=="ring" and r["two_j"]==5 and r["map"]["kind"]=="bounded"]
    raw_hole=float(np.mean([np.mean(np.linalg.norm(np.load(confirmation/r["data_directory"]/"data.npz")["reference"],axis=-1)<1.)
                            for r in reference_group]))
    lines += ["", "## Support structure: post-hoc annulus check", "",
        "Visual inspection prompted an additional exploratory diagnostic after",
        "confirmation. The target annulus has nominal radius 2; measure the",
        "fraction of decoded observations inside radius 1 around its true origin.",
        "This was not a selection or gate criterion. It changes no candidate,",
        "draw or predeclared conclusion. Raw test mass in this hole is "+f"{100*raw_hole:.2f}%.", "",
        "| Readout spin | Old map hole mass | Stereo hole mass | Sinh hole mass |",
        "|---|---:|---:|---:|"]
    for two_j in (5,10,20):
        lines.append(f"| {two_j/2:g} | "+" | ".join(f"{100*ring[str(two_j)][kind]['mean']:.1f}%" for kind in ("bounded","stereographic","sinh_stereographic"))+" |")
    lines += ["", "The lower scalar error does not imply restoration of the annulus hole.",
        "Two-moons structure is also visibly smoothed in the saved first-seed",
        "panels. A future choice intended to preserve support must declare",
        "structural guardrails and use new selection/confirmation data. These",
        "results do not justify calling the low-j Euclidean benchmark solved.", "",
        "## Verification and provenance", ""]
    audits=[json.loads((directory/"audit.json").read_text()) for directory in (screen,confirmation)]
    for label,directory,audit in zip(("Screen","Confirmation"),(screen,confirmation),audits):
        lines += [f"- {label}: `{directory.relative_to(ROOT)}`, {audit['cases']} cases, {audit['hashed_artifacts']} hashes; independent maximum decoding replay difference {audit['maximum_decoding_replay_error']:.3e}.",
                  f"  Manifest: `{audit['manifest_sha256']}`."]
    angular=json.loads((screen/"angular_control.json").read_text())
    lines += [f"- Fresh angular reference: {angular['significant_modes']} separated significant coherent-readout maxima at j=5/2 under the fixed grid and threshold.",
        f"- Kernel rotation-covariance residual: {audits[-1]['kernel_rotation_residual']:.3e}.",
        "- The initial screen was replayed under the corrected independent verifier,",
        "  which now respects the bounded chart's normalization order at the clipped",
        "  equator. Algebraically equivalent orderings created sqrt(machine-epsilon)",
        "  direction differences there. The original screen is retained; primary",
        "  scores and selections in the replay are identical. It is not an additional",
        "  independent data realization, and no map or selection criterion changed.",
        "- Independent checks reconstruct coherent mixtures, continuous quantiles,",
        "  inverse maps, all primary scores, pole densities and the angular reference.",
        "  The repository suite passes 112 tests.",
        f"- Screen/confirmation source fingerprint: `{json.loads((confirmation/'metadata.json').read_text())['source_fingerprint']}`.",
        f"- SciPy runtime: {scipy.__version__}; frozen numerical source and NumPy/Torch/Python versions are saved with both phases.",
        "- No reverse model, quantum-advantage gate or manuscript claim is promoted.",
        "  Confirmation observations are consumed and cannot be reused for selection.", "",
        "## Artifacts and next use", "",
        f"- [Error comparison](../{confirmation.relative_to(ROOT)}/chart_comparison.png).",
        f"- [Decoded distributions](../{confirmation.relative_to(ROOT)}/decoded_distributions.png): first declared seed; fixed display windows, all tails included in metrics.",
        f"- [Frozen selection](../{screen.relative_to(ROOT)}/selection.json).",
        f"- [Decisions and exploratory hole masses](../{output.relative_to(ROOT)}/decisions.json).", "",
        "The sinh family provides better-conditioned adapter candidates, not a",
        "trained generative result. It removes geometric clipping and makes the",
        "decoded moment problem well behaved, but finite-spin smoothing remains.",
        "Any reverse-training comparison on these adapters requires a separate",
        "frozen protocol and fresh evaluation observations. The authoritative",
        "paper restart remains the controlled E5 correlation-stage protocol.", ""]
    report=ROOT/"notes/EUCLIDEAN_CHART_REPORT.md"
    prefix=report.read_text().split("## Numerical results")[0]
    report.write_text(prefix+"\n".join(lines))
    write(output/"decisions.json",{"screen_manifest_sha256":digest(screen/"manifest.json"),
        "confirmation_manifest_sha256":digest(confirmation/"manifest.json"),"decisions":decisions,
        "posthoc_ring_hole_mass":ring,"raw_ring_hole_mass":raw_hole,
        "scope":"adapter relative error gates; not faithful support recovery or reverse training"})
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    shutil.copy2(report,output/report.name)
    write(output/"manifest.json",{path.name:digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    print(json.dumps({"relative_adapter_gates_passed":sum(d["passed_predeclared_adapter_gate"] for d in decisions.values()),
        "minimum_gain":min(d["mean_w1_reduction"] for d in decisions.values()),
        "maximum_gain":max(d["mean_w1_reduction"] for d in decisions.values()),"posthoc_ring":ring},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir",type=Path,required=True)
    parser.add_argument("--confirmation-dir",type=Path,required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    args=parser.parse_args()
    summarize(args.screen_dir.resolve(),args.confirmation_dir.resolve(),args.outdir.resolve())
