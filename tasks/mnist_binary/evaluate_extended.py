"""Compare the paired extended MNIST runs on their saved validation split."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .__main__ import DATA, OUTPUTS, ROOT, load_checkpoint, new_output, write_json
from .data import load_training_data
from .model import DIGITS, generate_images, local_measurement


SETTINGS = {
    "classic_short": "Classico breve",
    "classic_long": "Classico lungo",
    "rich": "Modello ricco",
}
SEEDS = (7, 19)
CLASSIFIER_SEED = 31
PREVIEW_SEED = 42


def fit_classifier(images, labels):
    torch.manual_seed(CLASSIFIER_SEED)
    classifier = nn.Sequential(
        nn.Conv2d(1, 12, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(12, 24, 4, 2, 1), nn.ReLU(),
        nn.Flatten(), nn.Linear(24 * 7 * 7, 2),
    ).float()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.002)
    for _ in range(4):
        for indices in torch.randperm(len(images)).split(128):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(images[indices])
            F.cross_entropy(logits, labels[indices]).backward()
            optimizer.step()
    return classifier.eval()


def pixel_rms(images, weights):
    total = weights.sum()
    if total == 0:
        return None
    weights = weights / total
    average = weights @ images
    variance = weights @ np.square(images) - np.square(average)
    return float(np.sqrt(max(0, variance.mean())))


def audit_density(states):
    np.testing.assert_allclose(states, states.conj().swapaxes(-2, -1), atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.trace(states, axis1=-2, axis2=-1), 1, atol=1e-10, rtol=0)
    if np.linalg.eigvalsh(states).min() < -1e-10:
        raise ValueError("Invalid density matrix in run")


def audit_born(frame, states, stored):
    replayed = np.einsum("ki,aij,kj->ak", frame.conj(), states, frame, optimize=True).real
    np.testing.assert_allclose(replayed, stored, atol=1e-10, rtol=0)
    np.testing.assert_allclose(stored.sum(axis=1), 1, atol=1e-10, rtol=0)
    if stored.min() < 0:
        raise ValueError("Invalid Born probabilities in run")


@torch.no_grad()
def evaluate_run(directory, classifier, train_rows, validation_rows):
    with np.load(directory / "split.npz") as split:
        np.testing.assert_array_equal(split["train"], train_rows)
        np.testing.assert_array_equal(split["validation"], validation_rows)
    checkpoint = directory / "checkpoint.pt"
    codec, model = load_checkpoint(checkpoint)
    measurement = local_measurement(model.config).product()
    frame = measurement.frame.numpy()
    states_path = directory / "states.npz"
    metrics_path = directory / "metrics.json"
    metadata = json.loads((directory / "metadata.json").read_text())
    if "preset" in metadata["model"]:
        for path, expected in metadata["source_sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source differs from run metadata for {path}")
    with np.load(states_path) as states:
        np.testing.assert_allclose(measurement.directions.numpy(), states["directions"], atol=1e-12, rtol=0)
        reverse = model(torch.tensor(DIGITS))
        np.testing.assert_allclose(reverse.numpy(), states["reverse"], atol=1e-10, rtol=0)
        for key in ("forward", "reverse", "validation"):
            audit_density(states[key])
        weights = {key: states[f"{key}_probabilities"] for key in ("generated", "validation")}
        audit_born(frame, states["reverse"][:, -1], weights["generated"])
        audit_born(frame, states["validation"], weights["validation"])
        audit_born(frame, states["forward"][:, 0], states["training_probabilities"])

    decoded = torch.cat([
        codec.logits(batch).sigmoid() for batch in measurement.directions.split(256)
    ])
    predictions = torch.cat([classifier(batch).argmax(dim=1) for batch in decoded.split(256)]).numpy()
    pixels = decoded.flatten(1).double().numpy()
    metrics = json.loads(metrics_path.read_text())
    result = {"directory": str(directory.relative_to(ROOT)), "digits": {}}
    for row, digit in enumerate(DIGITS):
        match = predictions == digit
        result["digits"][str(digit)] = {
            "generated_accuracy": float(weights["generated"][row, match].sum()),
            "target_readout_accuracy": float(weights["validation"][row, match].sum()),
            "generated_correct_class_pixel_rms": pixel_rms(pixels[match], weights["generated"][row, match]),
            "target_readout_correct_class_pixel_rms": pixel_rms(pixels[match], weights["validation"][row, match]),
            **metrics["validation"][str(digit)],
        }
    result["macro_accuracy"] = float(np.mean([digit["generated_accuracy"] for digit in result["digits"].values()]))
    result["mean_born_tv"] = float(np.abs(weights["generated"] - weights["validation"]).sum(axis=1).mean() / 2)
    result["training_seconds"] = metrics["codec_seconds"] + metrics["reverse_seconds"]
    result["sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (checkpoint, states_path, metrics_path)}
    rng = torch.Generator().manual_seed(PREVIEW_SEED)
    samples, _, _ = generate_images(codec, measurement, reverse[:, -1], 10, rng)
    scores = classifier(samples.reshape(-1, 1, 28, 28)).softmax(dim=1)
    return result, (samples.numpy(), scores.reshape(2, 10, 2).numpy())


def render_comparison(path, results, previews):
    background = "#f8f9fb"
    fig = plt.figure(figsize=(13.8, 8.6), facecolor=background)
    fig.text(0.055, 0.951, "Cifre generate da SpinDiffusion", weight="bold", fontsize=24, color="#182331")
    fig.text(0.055, 0.918, "Seed di training 7 · dieci estrazioni per classe · il numero sotto l’immagine è la cifra letta dal classificatore",
             fontsize=11, color="#445265")
    for digit, title_y, top, bottom in ((0, 0.871, 0.843, 0.516), (1, 0.445, 0.417, 0.09)):
        fig.text(0.055, title_y, f"Cifra richiesta  {digit}", weight="bold", fontsize=15, color="#182331")
        slots = fig.add_gridspec(3, 10, left=0.185, right=0.955, top=top, bottom=bottom, wspace=0.20, hspace=0.34)
        for row, (setting, label) in enumerate(SETTINGS.items()):
            samples, scores = previews[setting, 7]
            accuracy = results[setting]["7"]["digits"][str(digit)]["generated_accuracy"]
            for col, pixels in enumerate(samples[digit]):
                ax = fig.add_subplot(slots[row, col])
                prediction = scores[digit, col].argmax()
                confidence = scores[digit, col, prediction]
                color = "#875408" if confidence < 0.9 else ("#11654c" if prediction == digit else "#aa2436")
                ax.imshow(pixels, vmin=0, vmax=1, cmap="gray", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for border in ax.spines.values():
                    border.set_color(color)
                    border.set_linewidth(1.5)
                ax.text(0.5, -0.075, str(prediction), transform=ax.transAxes,
                        ha="center", va="top", color=color, weight="bold", fontsize=9)
                if col == 0:
                    center = ax.get_position().y0 + ax.get_position().height / 2
                    formatted = f"{accuracy:.1%}".replace(".", ",")
                    fig.text(0.055, center, f"{label}\n{formatted} corrisponde", fontsize=10,
                             color="#182331", va="center")
    fig.text(0.055, 0.032, "Verde  cifra richiesta     ·     Rosso  altra cifra     ·     Ocra  lettura incerta (< 90% di confidenza)",
             fontsize=10, color="#445265")
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    directory = new_output(args.outdir)
    torch.set_num_threads(4)
    images, labels, train_rows, validation_rows, heldout = load_training_data(DATA, 20261110)
    classifier = fit_classifier(images[~heldout], labels[~heldout])
    torch.save(classifier.state_dict(), directory / "classifier.pt")
    with torch.no_grad():
        validation_accuracy = (classifier(images[heldout]).argmax(1) == labels[heldout]).float().mean().item()
    results = {setting: {} for setting in SETTINGS}
    previews = {}
    for setting in SETTINGS:
        for seed in SEEDS:
            run = args.study_dir / f"{setting}_s{seed}"
            if (setting, seed) == ("classic_short", 7):
                run = OUTPUTS / "mnist_binary_01_v1"
            result, preview = evaluate_run(run.resolve(), classifier, train_rows, validation_rows)
            results[setting][str(seed)] = result
            previews[setting, seed] = preview
            print(f"{setting} seed {seed}: {result['macro_accuracy']:.2%}", flush=True)
    models = [(setting, seed, result) for setting, runs in results.items() for seed, result in runs.items()]
    best_setting, best_seed, best = max(models, key=lambda row: (row[2]["macro_accuracy"], -row[2]["mean_born_tv"]))
    write_json(directory / "comparison.json", {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classifier_seed": CLASSIFIER_SEED,
        "classifier_real_validation_accuracy": validation_accuracy,
        "preview_seed": PREVIEW_SEED,
        "real_validation_pixel_rms": {str(digit): pixel_rms(
            images[heldout & (labels == digit)].flatten(1).double().numpy(), np.ones(500)
        ) for digit in DIGITS},
        "models": results,
        "demonstration_checkpoint": {"setting": best_setting, "seed": int(best_seed),
                                     "path": best["directory"] + "/checkpoint.pt"},
    })
    render_comparison(directory / "comparison.png", results, previews)


if __name__ == "__main__":
    main()
