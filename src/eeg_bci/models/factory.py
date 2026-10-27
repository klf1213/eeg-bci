from __future__ import annotations

from .eegnet import EEGNet
from .temporal_attention import EEGNetTemporalAttention, EEGNetTemporalMean


MODEL_NAMES = (
    "eegnet",
    "temporal-mean",
    "temporal-mean-hires",
    "attention",
    "attention-hires",
    "attention-positional-residual",
)


def build_model(name: str, n_time_points: int, config: dict):
    """Build a configured EEGNet variant by its command-line name."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")

    model_config = config["model"]
    common = {
        "n_channels": config["data"]["n_channels"],
        "n_classes": model_config["n_classes"],
        "f1": model_config["f1"],
        "depth_multiplier": model_config["depth_multiplier"],
        "f2": model_config["f2"],
        "temporal_kernel_length": model_config["temporal_kernel_length"],
        "separable_kernel_length": model_config["separable_kernel_length"],
        "dropout": model_config["dropout"],
    }
    if name == "eegnet":
        return EEGNet(n_time_points=n_time_points, **common)

    high_resolution = name in {
        "temporal-mean-hires",
        "attention-hires",
        "attention-positional-residual",
    }
    pool_key = (
        "high_resolution_second_pool_size"
        if high_resolution
        else "attention_second_pool_size"
    )
    second_pool_size = model_config[pool_key]

    if name in {"temporal-mean", "temporal-mean-hires"}:
        return EEGNetTemporalMean(second_pool_size=second_pool_size, **common)

    positional_residual = name == "attention-positional-residual"
    return EEGNetTemporalAttention(
        attention_hidden_size=model_config["attention_hidden_size"],
        n_time_points=n_time_points,
        second_pool_size=second_pool_size,
        use_position=positional_residual,
        residual_mean=positional_residual,
        **common,
    )
