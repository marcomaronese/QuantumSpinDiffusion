"""Classical image maps and the conditional SpinDiffusion model for 0 and 1."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from spin_quantum_diffusion.quantum.coherent_readout import (
    coherent_kets,
    coherent_povm,
    sample_coherent_directions,
)
from spin_quantum_diffusion.quantum.correlated import (
    build_two_spin_forward_trajectory,
    build_two_spin_reverse_generators,
)
from spin_quantum_diffusion.quantum.reverse import reverse_collision_channel
from spin_quantum_diffusion.quantum.single_spin import assert_physical_density, spin_operators


DIGITS = (0, 1)
POLAR_ORDER = 6
AZIMUTH_COUNT = 12


@dataclass(frozen=True)
class ModelConfig:
    two_j: int = 3
    steps: int = 3
    layers: int = 2
    codec_width: int = 16
    codec_hidden: int = 0
    controller_width: int = 32
    separate_ancilla_couplings: bool = False
    angle_scale: float = 1.0
    foreground_weight: float = 0.0
    readout_weight: float = 0.0
    reverse_lr: float = 0.025
    path_weight: float = 0.15
    diffusion_rate: float = 1.0
    diffusion_time: float = 2.25


PRESETS = {
    "classic": ModelConfig(),
    "rich": ModelConfig(
        two_j=5, steps=6, layers=3, codec_width=24, codec_hidden=96,
        controller_width=64, separate_ancilla_couplings=True, angle_scale=4,
        foreground_weight=0.15, readout_weight=0.5, reverse_lr=0.008,
    ),
}


class DigitCodec(nn.Module):
    """Encode images on S² × S² and decode a pair of measured directions."""

    def __init__(self, config: ModelConfig = PRESETS["classic"]):
        super().__init__()
        self.config = config
        width = config.codec_width
        flattened = 2 * width * 7 * 7
        hidden = config.codec_hidden
        encoder_hidden = [nn.Linear(flattened, hidden), nn.SiLU()] if hidden else []
        decoder_hidden = [nn.Linear(6, hidden), nn.SiLU()] if hidden else []
        self.encoder = nn.Sequential(
            nn.Conv2d(1, width, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(width, 2 * width, 4, 2, 1), nn.SiLU(),
            nn.Flatten(), *encoder_hidden, nn.Linear(hidden or flattened, 6),
        ).float()
        self.decoder = nn.Sequential(
            *decoder_hidden, nn.Linear(hidden or 6, flattened), nn.SiLU(),
            nn.Unflatten(1, (2 * width, 7, 7)),
            nn.ConvTranspose2d(2 * width, width, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(width, 1, 4, 2, 1),
        ).float()

    def encode(self, images):
        return F.normalize(self.encoder(images).reshape(-1, 2, 3), dim=-1)

    def logits(self, directions):
        return self.decoder(directions.reshape(-1, 6).float())


class DigitSpinDiffusion(nn.Module):
    """A shared MLP maps the digit and time to physical collision controls."""

    def __init__(self, config: ModelConfig = PRESETS["classic"]):
        super().__init__()
        self.config = config
        *self.operators, self.identity = spin_operators(config.two_j / 2)
        self.generators, _ = build_two_spin_reverse_generators(
            self.operators, self.identity, include_direct_interactions=True,
            separate_ancilla_couplings=config.separate_ancilla_couplings,
        )
        dimension = self.identity.shape[0] ** 2
        self.prior = torch.eye(dimension, dtype=self.identity.dtype) / dimension
        self.controls = nn.Sequential(
            nn.Linear(len(DIGITS) + config.steps, config.controller_width, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(config.controller_width, config.layers * len(self.generators), dtype=torch.float64),
        )

    def forward(self, digits):
        config = self.config
        classes = F.one_hot(digits, num_classes=len(DIGITS))[:, None, :].expand(-1, config.steps, -1)
        times = torch.eye(config.steps, dtype=classes.dtype).expand(len(digits), -1, -1)
        features = torch.cat([classes, times], dim=-1).to(torch.float64)
        angles = self.controls(features).reshape(len(digits), config.steps, config.layers, -1)
        angles = angles * config.angle_scale
        trajectories = []
        for class_angles in angles:
            states = [self.prior]
            for step_angles in class_angles:
                states.append(reverse_collision_channel(states[-1], step_angles, self.generators))
            trajectories.append(torch.stack(states))
        return torch.stack(trajectories)

    def forward_targets(self, states):
        config = self.config
        return torch.stack([
            torch.stack(build_two_spin_forward_trajectory(
                rho, self.operators, self.identity, config.diffusion_rate,
                config.diffusion_time / config.steps, config.steps,
            ))
            for rho in states
        ])


def local_measurement(config):
    return coherent_povm(config.two_j, POLAR_ORDER, AZIMUTH_COUNT)


def measured_directions(directions, measurement, rng):
    kets = coherent_kets(directions, measurement.frame.shape[1] - 1)
    probabilities = (kets @ measurement.frame.conj().T).abs().square()
    outcomes = torch.multinomial(probabilities.flatten(0, 1), 1, generator=rng)
    return measurement.directions[outcomes.reshape(-1, 2)]


@torch.no_grad()
def encoded_states(codec, images, labels, batch_size):
    directions = torch.cat([codec.encode(batch) for batch in images.split(batch_size)])
    kets = coherent_kets(directions, codec.config.two_j)
    joint = torch.einsum("ni,nj->nij", kets[:, 0], kets[:, 1]).flatten(1)
    states = []
    for digit in DIGITS:
        group = joint[labels == digit]
        states.append(group.T @ group.conj() / len(group))
    return torch.stack(states)


def train_codec(codec, images, batch_size, epochs, readout_epochs, rng):
    """Pretrain through the continuous POVM and calibrate on the finite grid."""
    config = codec.config
    measurement = local_measurement(config)
    history = []
    for finite, num_epochs in ((False, epochs), (True, readout_epochs)):
        parameters = codec.decoder.parameters() if finite else codec.parameters()
        optimizer = torch.optim.Adam(parameters, lr=0.001 if finite else 0.002)
        total_pixels = (1, 2, 3)
        for epoch in range(num_epochs):
            total_bce = 0.0
            total_loss = 0.0
            for indices in torch.randperm(len(images), generator=rng).split(batch_size):
                batch = images[indices]
                if finite:
                    with torch.no_grad():
                        readout = measured_directions(codec.encode(batch), measurement, rng)
                else:
                    readout = sample_coherent_directions(codec.encode(batch), config.two_j, rng)
                optimizer.zero_grad(set_to_none=True)
                logits = codec.logits(readout)
                bce = F.binary_cross_entropy_with_logits(logits, batch)
                loss = bce
                if config.foreground_weight:
                    predicted = logits.sigmoid()
                    overlap = 2 * (predicted * batch).sum(dim=total_pixels)
                    mass = (predicted + batch).sum(dim=total_pixels)
                    loss = bce + config.foreground_weight * (1 - overlap / mass).mean()
                loss.backward()
                optimizer.step()
                total_bce += float(bce.detach()) * len(batch)
                total_loss += float(loss.detach()) * len(batch)
            history.append({"bce": total_bce / len(images), "loss": total_loss / len(images)})
            phase = "finite readout" if finite else "continuous readout"
            print(f"Codec {phase}, epoch {epoch + 1}/{num_epochs}: BCE {history[-1]['bce']:.4f}", flush=True)
    return history


def train_reverse(model, forward, updates):
    config = model.config
    optimizer = torch.optim.Adam(model.parameters(), lr=config.reverse_lr)
    digits = torch.tensor(DIGITS)
    # In reverse order, exclude the fixed input prior from the training loss.
    targets = forward.flip(1)[:, 1:]
    if config.readout_weight:
        measurement = local_measurement(config).product()
        target_roots = torch.stack([measurement.probabilities(state).clamp_min(1e-15).sqrt()
                                    for state in targets[:, -1]])
    history = []
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        trajectory = model(digits)[:, 1:]
        errors = (trajectory - targets).abs().square().sum(dim=(-1, -2))
        loss = errors[:, -1].mean() + config.path_weight * errors[:, :-1].mean()
        if config.readout_weight:
            roots = torch.stack([measurement.probabilities(state).clamp_min(1e-15).sqrt()
                                 for state in trajectory[:, -1]])
            hellinger_squared = 0.5 * (roots - target_roots).square().sum(dim=1).mean()
            loss = loss + config.readout_weight * hellinger_squared
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if update == 0 or (update + 1) % 100 == 0 or update + 1 == updates:
            print(f"Reverse update {update + 1}/{updates}: loss {history[-1]:.6f}", flush=True)
    return history


def born_probabilities(measurement, rho):
    assert_physical_density(rho, "MNIST readout state")
    probabilities = measurement.probabilities(rho)
    if probabilities.min() < -1e-10 or abs(float(probabilities.sum()) - 1) > 1e-10:
        raise ValueError("Invalid Born probabilities for the product POVM")
    return probabilities.clamp_min(0)


@torch.no_grad()
def generate_images(codec, measurement, states, count, rng):
    probabilities = torch.stack([born_probabilities(measurement, rho) for rho in states])
    outcomes = torch.multinomial(probabilities, count, replacement=True, generator=rng)
    directions = measurement.directions[outcomes]
    images = codec.logits(directions).sigmoid().reshape(len(states), count, 28, 28)
    return images, outcomes, probabilities
