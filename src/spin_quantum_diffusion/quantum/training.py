"""Loss semantics for composed and teacher-forced collision-channel training."""

from collections.abc import Sequence

import torch

from .losses import objective_loss
from .reverse import parameter_schedule, reverse_collision_channel


SUPERVISION_MODES = ("path", "path-normalized", "final", "stepwise")


def reverse_training_loss(
    forward_states: Sequence[torch.Tensor],
    parameters: torch.Tensor,
    generators: Sequence[torch.Tensor],
    prior: torch.Tensor,
    *,
    supervision: str = "path",
    objective: str = "frobenius",
    **loss_options,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the objective and last prediction, retaining the full gradient.

    ``path`` sums losses along the composed reverse chain; ``path-normalized``
    divides that sum by the number of steps. ``final`` uses the composed final
    state only. ``stepwise`` averages one-step losses evaluated on exact forward
    inputs, with one joint optimizer update per epoch. Its last prediction is
    teacher-forced, not the generated state of the assembled chain.
    """
    if supervision not in SUPERVISION_MODES:
        raise ValueError(f"unknown supervision mode: {supervision}")
    steps = len(forward_states) - 1
    if steps < 1:
        raise ValueError("at least one forward step is required")
    rho = prior
    loss = torch.zeros((), dtype=prior.real.dtype, device=prior.device)
    schedule = parameter_schedule(parameters, n_steps=steps)
    for theta, t in zip(schedule, range(steps, 0, -1), strict=True):
        channel_input = forward_states[t] if supervision == "stepwise" else rho
        rho = reverse_collision_channel(channel_input, theta, generators)
        if supervision == "final" and t != 1:
            continue
        loss = loss + objective_loss(
            rho, forward_states[t - 1], objective, **loss_options
        )
    if supervision in {"path-normalized", "stepwise"}:
        loss = loss / steps
    return loss, rho
