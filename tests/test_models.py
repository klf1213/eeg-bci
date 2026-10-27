from pathlib import Path

import torch
import yaml

from eeg_bci.models import (
    MODEL_NAMES,
    EEGNet,
    EEGNetTemporalAttention,
    EEGNetTemporalMean,
    build_model,
)


def test_models_produce_four_class_logits():
    x = torch.randn(3, 1, 22, 256)
    models = [
        EEGNet(n_time_points=256),
        EEGNetTemporalMean(),
        EEGNetTemporalAttention(),
    ]

    for model in models:
        assert model(x).shape == (3, 4)


def test_attention_is_normalized_over_time():
    model = EEGNetTemporalAttention()
    logits, attention = model(torch.randn(3, 1, 22, 256), return_attention=True)

    assert logits.shape == (3, 4)
    assert attention.shape == (3, 8)
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))


def test_high_resolution_attention_has_more_time_bins():
    model = EEGNetTemporalAttention(n_time_points=256, second_pool_size=4)
    logits, attention = model(torch.randn(3, 1, 22, 256), return_attention=True)

    assert logits.shape == (3, 4)
    assert attention.shape == (3, 16)
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))


def test_positional_residual_attention_supports_wide_window():
    model = EEGNetTemporalAttention(
        n_time_points=512, second_pool_size=4, use_position=True, residual_mean=True,
    )
    logits, attention = model(torch.randn(2, 1, 22, 512), return_attention=True)

    assert logits.shape == (2, 4)
    assert attention.shape == (2, 32)
    assert 0 < torch.sigmoid(model.attention_gate).item() < 1


def test_model_factory_builds_all_cli_variants():
    config_path = Path(__file__).parents[1] / "configs/default.yaml"
    config = yaml.safe_load(config_path.read_text())

    for name in MODEL_NAMES:
        model = build_model(name, n_time_points=256, config=config)
        assert model(torch.randn(1, 1, 22, 256)).shape == (1, 4)
