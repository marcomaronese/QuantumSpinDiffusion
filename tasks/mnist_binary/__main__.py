"""Train a binary MNIST model and sample its saved checkpoint."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

from matplotlib.image import imsave
import numpy as np
import torch

from .data import BASE_URL, TRAIN_FILES, load_training_data
from .model import (
    AZIMUTH_COUNT, DIGITS, POLAR_ORDER, PRESETS, DigitCodec, DigitSpinDiffusion,
    ModelConfig, born_probabilities,
    encoded_states, generate_images, local_measurement, measured_directions,
    train_codec, train_reverse,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs" / "experiments"
DATA = OUTPUTS / "mnist_binary_data"
CHECKPOINT_VERSION = 2


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def new_output(directory):
    directory = directory.resolve()
    if OUTPUTS not in directory.parents or directory == DATA or DATA in directory.parents:
        raise ValueError(f"Use a fresh output directory under {OUTPUTS}")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    task = Path(__file__).resolve().parent
    quantum = ROOT / "src" / "spin_quantum_diffusion" / "quantum"
    sources = [task / name for name in ("__init__.py", "__main__.py", "model.py", "data.py", "README.md")] + [
        quantum / name for name in ("coherent_readout.py", "correlated.py", "reverse.py", "single_spin.py")
    ]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}


def save_samples(directory, name, digits, images, outcomes):
    np.savez_compressed(directory / f"{name}.npz", digits=digits, images=images, outcomes=outcomes)
    # Use at most 12 samples per row in the preview. The NPZ preserves all samples.
    panels = images[:, :12].numpy()
    panels = np.pad(panels, ((0, 0), (0, 0), (0, 2), (0, 2)))
    rows, columns, height, width = panels.shape
    grid = panels.transpose(0, 2, 1, 3).reshape(rows * height, columns * width)
    imsave(directory / f"{name}.png", grid, cmap="gray", vmin=0, vmax=1)


def save_comparison(directory, images):
    originals, direct, measured, target, generated = (group[:, :8] for group in images)
    combined = torch.stack([originals, direct, measured, target, generated], dim=1)
    panels = np.pad(combined.flatten(0, 1).numpy(), ((0, 0), (0, 0), (0, 2), (0, 2)))
    grid = panels.transpose(0, 2, 1, 3).reshape(10 * 30, 8 * 30)
    imsave(directory / "comparison.png", grid, cmap="gray", vmin=0, vmax=1)


@torch.no_grad()
def evaluate(codec, model, forward, images, labels, batch_size, directory, seed):
    rng = torch.Generator().manual_seed(seed)
    digits = torch.tensor(DIGITS)
    local = local_measurement(model.config)
    joint = local.product()
    reverse = model(digits)
    states = encoded_states(codec, images, labels, batch_size)
    generated, outcomes, probabilities = generate_images(codec, joint, reverse[:, -1], 32, rng)
    target_images, target_outcomes, heldout_probabilities = generate_images(codec, joint, states, 32, rng)
    save_samples(directory, "generated", digits, generated, outcomes)
    save_samples(directory, "target_readout", digits, target_images, target_outcomes)

    raw = []
    direct = []
    measured = []
    report = {}
    for row, digit in enumerate(DIGITS):
        group = images[labels == digit]
        latent = torch.cat([codec.encode(batch) for batch in group.split(batch_size)])
        direct_reconstruction = codec.logits(latent).sigmoid()
        observed = measured_directions(latent, local, rng)
        readout_reconstruction = codec.logits(observed).sigmoid()
        originals = torch.randperm(len(group), generator=rng)[:8]
        raw.append(group[originals, 0])
        direct.append(direct_reconstruction[originals, 0])
        measured.append(readout_reconstruction[originals, 0])
        prior_error = reverse[row, 0] - forward[row, -1]
        state_error = reverse[row, -1] - states[row]
        report[str(digit)] = {
            "direct_reconstruction_mse": float((direct_reconstruction - group).square().mean()),
            "measured_reconstruction_mse": float((readout_reconstruction - group).square().mean()),
            "trace_distance_prior_to_forward_end": float(torch.linalg.eigvalsh(prior_error).abs().sum() / 2),
            "trace_distance_to_validation_state": float(torch.linalg.eigvalsh(state_error).abs().sum() / 2),
            "total_variation_to_validation_born_distribution": float(
                (probabilities[row] - heldout_probabilities[row]).abs().sum() / 2
            ),
        }
    save_comparison(directory, (torch.stack(raw), torch.stack(direct), torch.stack(measured), target_images, generated))
    np.savez_compressed(
        directory / "states.npz", digits=digits, forward=forward, reverse=reverse,
        validation=states, generated_probabilities=probabilities,
        validation_probabilities=heldout_probabilities, directions=joint.directions,
        training_probabilities=torch.stack([born_probabilities(joint, rho) for rho in forward[:, 0]]),
    )
    return report


def train(args):
    directory = new_output(args.outdir)
    images, labels, train_rows, validation_rows, heldout = load_training_data(DATA, args.split_seed)
    config = PRESETS[args.preset]
    codec = DigitCodec(config)
    model = DigitSpinDiffusion(config)
    np.savez(directory / "split.npz", train=train_rows, validation=validation_rows)
    write_json(directory / "metadata.json", {
        "dataset": {"source": BASE_URL, "sha256": TRAIN_FILES},
        "digits": DIGITS,
        "counts": {"train": len(train_rows), "validation": len(validation_rows)},
        "seeds": {"split": args.split_seed, "model_and_training": args.seed, "evaluation": args.seed + 1},
        "budget": {"batch_size": args.batch_size, "codec_epochs": args.codec_epochs,
                   "readout_epochs": args.readout_epochs, "reverse_updates": args.reverse_updates},
        "model": {**asdict(config), "preset": args.preset,
                  "codec_parameters": sum(p.numel() for p in codec.parameters()),
                  "controller_parameters": sum(p.numel() for p in model.parameters()),
                  "polar_order": POLAR_ORDER, "azimuth_count": AZIMUTH_COUNT},
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "codec_threads": args.threads,
                    "reverse_threads": 1},
        "source_sha256": source_hashes(),
    })
    train_images = images[~heldout]
    rng = torch.Generator().manual_seed(args.seed)
    start = perf_counter()
    codec_history = train_codec(codec, train_images, args.batch_size, args.codec_epochs, args.readout_epochs, rng)
    codec_seconds = perf_counter() - start
    start = perf_counter()
    targets = encoded_states(codec, train_images, labels[~heldout], args.batch_size)
    forward = model.forward_targets(targets)
    torch.set_num_threads(1)
    reverse_history = train_reverse(model, forward, args.reverse_updates)
    reverse_seconds = perf_counter() - start
    torch.set_num_threads(args.threads)
    torch.save({"version": CHECKPOINT_VERSION, "config": asdict(config), "codec": codec.state_dict(),
                "model": model.state_dict()}, directory / "checkpoint.pt")
    metrics = evaluate(codec, model, forward, images[heldout], labels[heldout], args.batch_size, directory, args.seed + 1)
    write_json(directory / "metrics.json", {
        "codec_seconds": codec_seconds, "reverse_seconds": reverse_seconds,
        "codec_training_bce": [row["bce"] for row in codec_history],
        "codec_training_loss": [row["loss"] for row in codec_history],
        "reverse_training_loss": reverse_history,
        "validation": metrics,
    })
    print(f"Saved checkpoint and validation samples in {directory}")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    version = checkpoint.get("version")
    if version == 1:
        config = PRESETS["classic"]
    elif version == CHECKPOINT_VERSION:
        config = ModelConfig(**checkpoint["config"])
    else:
        raise ValueError("Unsupported MNIST binary checkpoint version")
    codec = DigitCodec(config)
    codec.load_state_dict(checkpoint["codec"])
    model = DigitSpinDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    return codec.eval(), model.eval()


@torch.no_grad()
def sample(args):
    codec, model = load_checkpoint(args.checkpoint)
    digits = torch.tensor(DIGITS if args.digit is None else [args.digit])
    rng = torch.Generator().manual_seed(args.seed)
    measurement = local_measurement(model.config).product()
    images, outcomes, _ = generate_images(codec, measurement, model(digits)[:, -1], args.count, rng)
    directory = new_output(args.outdir)
    save_samples(directory, "generated", digits, images, outcomes)
    print(f"Saved {len(digits) * args.count} samples in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train on MNIST digits 0 and 1")
    sampling = commands.add_parser("sample", help="Generate new digits from a checkpoint")
    for command in (training, sampling):
        command.add_argument("--outdir", type=Path, required=True)
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--threads", type=positive_integer, default=4)
    training.add_argument("--split-seed", type=int, default=20261110)
    training.add_argument("--preset", choices=PRESETS, default="classic")
    training.add_argument("--batch-size", type=positive_integer, default=128)
    training.add_argument("--codec-epochs", type=positive_integer, default=18)
    training.add_argument("--readout-epochs", type=positive_integer, default=4)
    training.add_argument("--reverse-updates", type=positive_integer, default=900)
    sampling.add_argument("--checkpoint", type=Path, required=True)
    sampling.add_argument("--digit", type=int, choices=DIGITS)
    sampling.add_argument("--count", type=positive_integer, default=32, help="Samples per digit")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.command == "train":
        train(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
