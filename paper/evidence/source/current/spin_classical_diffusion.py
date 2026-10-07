"""Classical Cartesian DDPM baseline for directions on (S^2)^M.

Uses the epsilon-prediction objective and Gaussian posterior of Ho et al.
(https://arxiv.org/abs/2006.11239), a cosine noise schedule, and a time-conditioned
MLP. Diffusion takes place in R^(3M); only the final samples are projected onto
the spheres. This is not intrinsic Riemannian diffusion or a parameter-matched
quantum ansatz. All stochastic operations use isolated CPU random generators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ClassicalDiffusionConfig:
    steps: int = 64
    epochs: int = 300
    batch_size: int = 256
    hidden_width: int = 64
    learning_rate: float = 1e-3

    def validate(self):
        for name in ("steps", "epochs", "batch_size", "hidden_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"classical {name} must be a positive integer")
        if self.steps < 2:
            raise ValueError("classical steps must be at least two")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("classical learning_rate must be finite and positive")


def add_classical_arguments(parser):
    """Shared explicit budget controls; the benchmark is enabled by default."""
    defaults = ClassicalDiffusionConfig()
    for name in ("steps", "epochs", "batch_size", "hidden_width"):
        parser.add_argument("--classical-" + name.replace("_", "-"), type=int,
                            default=getattr(defaults, name))
    parser.add_argument("--classical-learning-rate", type=float,
                        default=defaults.learning_rate)


def config_from_arguments(args):
    config = ClassicalDiffusionConfig(**{
        name: getattr(args, "classical_" + name)
        for name in ClassicalDiffusionConfig.__dataclass_fields__
    })
    config.validate()
    return config


def angles_to_cartesian(points):
    """Accept (N, 2) directions or (N, M, 2) joint directions."""
    angles = np.asarray(points, dtype=np.float64)
    if (angles.ndim not in (2, 3) or angles.shape[-1] != 2
            or angles.size == 0 or not np.isfinite(angles).all()):
        raise ValueError("expected finite, nonempty (N, 2) or (N, M, 2) angles")
    theta, phi = angles[..., 0], angles[..., 1]
    xyz = np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                    np.cos(theta)), axis=-1)
    return torch.tensor(xyz.reshape(len(angles), -1), dtype=torch.float64)


def cartesian_to_angles(samples):
    """Project each Cartesian triple; return (N, 2) or (N, M, 2)."""
    xyz = torch.as_tensor(samples, dtype=torch.float64).detach().cpu()
    if (xyz.ndim != 2 or xyz.shape[1] < 3 or xyz.shape[1] % 3
            or not torch.isfinite(xyz).all()):
        raise ValueError("expected finite (N, 3*M) Cartesian samples")
    xyz = xyz.reshape(len(xyz), -1, 3)
    norms = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    if torch.any(norms <= torch.finfo(xyz.dtype).eps):
        raise ValueError("cannot project a zero direction onto the sphere")
    xyz = xyz / norms
    angles = torch.stack((torch.acos(xyz[..., 2].clamp(-1, 1)),
                          torch.atan2(xyz[..., 1], xyz[..., 0]) % (2 * math.pi)), dim=-1)
    return (angles[:, 0] if angles.shape[1] == 1 else angles).numpy()


class ClassicalDDPM(nn.Module):
    """CPU float64 denoiser with a serializable diffusion schedule."""

    def __init__(self, dimensions=3, config=None):
        super().__init__()
        self.config = config or ClassicalDiffusionConfig()
        self.config.validate()
        if not isinstance(dimensions, int) or dimensions < 3 or dimensions % 3:
            raise ValueError("dimensions must be a positive multiple of three")
        self.dimensions = dimensions
        width = self.config.hidden_width
        self.network = nn.Sequential(
            nn.Linear(dimensions + 16, width, dtype=torch.float64), nn.SiLU(),
            nn.Linear(width, width, dtype=torch.float64), nn.SiLU(),
            nn.Linear(width, dimensions, dtype=torch.float64),
        )
        grid = torch.linspace(0, 1, self.config.steps + 1, dtype=torch.float64)
        cumulative = torch.cos((grid + .008) / 1.008 * math.pi / 2).square()
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=.999)
        alpha_bar = torch.cumprod(1 - beta, dim=0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("posterior_variance", beta * (1 - previous) / (1 - alpha_bar))
        self.register_buffer("posterior_x0", beta * previous.sqrt() / (1 - alpha_bar))
        self.register_buffer("posterior_xt", (1 - previous) * (1 - beta).sqrt() / (1 - alpha_bar))
        self.register_buffer("frequencies", 2 ** torch.arange(8, dtype=torch.float64) * math.pi)

    def forward(self, noisy, steps):
        phase = (steps.to(torch.float64)[:, None] + 1) / self.config.steps * self.frequencies
        return self.network(torch.cat((noisy, phase.sin(), phase.cos()), dim=-1))

    def forward_sample(self, clean, steps, noise):
        """Closed-form q(x_t | x_0); zero-based t indexes one noising step."""
        cumulative = self.alpha_bar[steps, None]
        return cumulative.sqrt() * clean + (1 - cumulative).sqrt() * noise

    def reverse_moments(self, noisy, steps, predicted_noise):
        cumulative = self.alpha_bar[steps, None]
        clean = ((noisy - (1 - cumulative).sqrt() * predicted_noise)
                 / cumulative.sqrt()).clamp(-1, 1)
        mean = self.posterior_x0[steps, None] * clean + self.posterior_xt[steps, None] * noisy
        return mean, self.posterior_variance[steps, None]

    @torch.no_grad()
    def sample(self, n_samples, seed):
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be positive")
        rng = torch.Generator(device="cpu").manual_seed(seed)
        values = torch.randn(n_samples, self.dimensions, generator=rng, dtype=torch.float64)
        for step in reversed(range(self.config.steps)):
            steps = torch.full((n_samples,), step, dtype=torch.long)
            mean, variance = self.reverse_moments(values, steps, self(values, steps))
            values = mean
            if step > 0:
                values = values + variance.sqrt() * torch.randn(
                    values.shape, generator=rng, dtype=values.dtype)
        return cartesian_to_angles(values)


def train_classical_diffusion(points, config=None, seed=7):
    """One epoch visits each training example once with fresh time/noise draws."""
    config = config or ClassicalDiffusionConfig()
    config.validate()
    data = angles_to_cartesian(points)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ClassicalDDPM(data.shape[1], config)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for _ in range(config.epochs):
        permutation = torch.randperm(len(data), generator=rng)
        total = 0.0
        for indices in permutation.split(config.batch_size):
            clean = data[indices]
            steps = torch.randint(config.steps, (len(clean),), generator=rng)
            noise = torch.randn(clean.shape, generator=rng, dtype=clean.dtype)
            noisy = model.forward_sample(clean, steps, noise)
            loss = (model(noisy, steps) - noise).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite classical diffusion training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(clean)
        history.append(total / len(data))
    return model.eval(), history


def save_classical_model(model, path: Path):
    torch.save({"format_version": 1, "dimensions": model.dimensions,
                "config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_classical_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("unsupported classical diffusion checkpoint version")
    with torch.random.fork_rng(devices=[]):
        model = ClassicalDDPM(checkpoint["dimensions"], ClassicalDiffusionConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
