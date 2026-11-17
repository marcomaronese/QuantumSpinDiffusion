from dataclasses import asdict

import pytest
import torch

from spin_quantum_diffusion.quantum.coherent_readout import coherent_kets, coherent_povm
from spin_quantum_diffusion.quantum.single_spin import spin_coherent_state
from tasks.mnist_binary.__main__ import CHECKPOINT_VERSION, load_checkpoint
from tasks.mnist_binary.model import PRESETS, DigitCodec, DigitSpinDiffusion, generate_images


def test_product_povm_is_complete_and_uses_the_same_direction_order_as_the_decoder():
    measurement = coherent_povm(two_j=3, polar_order=3, azimuth_count=5)
    joint = measurement.product()
    torch.testing.assert_close(joint.frame.T @ joint.frame.conj(), torch.eye(16, dtype=torch.complex128))
    directions = torch.tensor([[0., 0., 1.], [0., 1., 0.]], dtype=torch.float64)
    first, second = coherent_kets(directions, two_j=3)
    torch.testing.assert_close(second, spin_coherent_state(1.5, torch.pi / 2, torch.pi / 2))
    first_rho = torch.outer(first, first.conj())
    second_rho = torch.outer(second, second.conj())
    rho = torch.kron(first_rho, second_rho)
    expected = torch.kron(measurement.probabilities(first_rho), measurement.probabilities(second_rho))
    torch.testing.assert_close(joint.probabilities(rho), expected)
    torch.testing.assert_close(joint.directions[7], measurement.directions[[0, 7]])


@pytest.mark.parametrize("preset", PRESETS)
def test_digit_model_starts_at_mixed_prior_and_passes_born_outcomes_to_decoder(preset):
    torch.manual_seed(7)
    config = PRESETS[preset]
    model = DigitSpinDiffusion(config)
    codec = DigitCodec(config)
    joint = coherent_povm(config.two_j, polar_order=3, azimuth_count=6).product()
    digits = torch.tensor([0, 1])
    with torch.no_grad():
        trajectories = model(digits)
        torch.testing.assert_close(trajectories[:, 0], model.prior.expand(2, -1, -1))
        images, outcomes, probabilities = generate_images(
            codec, joint, trajectories[:, -1], 8, torch.Generator().manual_seed(42)
        )
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2, dtype=torch.float64))
        expected = codec.logits(joint.directions[outcomes]).sigmoid().reshape(2, 8, 28, 28)
        torch.testing.assert_close(images, expected)


def test_extended_checkpoint_restores_the_spin_and_codec_configuration(tmp_path):
    config = PRESETS["rich"]
    codec = DigitCodec(config)
    model = DigitSpinDiffusion(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"version": CHECKPOINT_VERSION, "config": asdict(config),
                "codec": codec.state_dict(), "model": model.state_dict()}, checkpoint)
    restored_codec, restored_model = load_checkpoint(checkpoint)
    assert restored_codec.config == restored_model.config == config
    assert restored_model.prior.shape == (36, 36)
    assert len(restored_model.generators) == 16
