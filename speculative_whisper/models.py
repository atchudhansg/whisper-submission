"""Model loading, caching, and device management for Whisper model pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import whisper

from speculative_whisper.config import DecodingConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelPair:
    """Holds both the draft and final Whisper models on the target device.

    Attributes:
        draft: The lightweight draft model (e.g. Whisper Tiny).
        final: The high-quality verification model (e.g. Whisper Large V3).
        device: Resolved device string ('cuda' or 'cpu').
        dtype: Tensor dtype matching the device (float16 for cuda, float32 for cpu).
    """

    draft: whisper.Whisper
    final: whisper.Whisper
    device: str
    dtype: torch.dtype


def load_models(config: DecodingConfig) -> ModelPair:
    """Load both Whisper models and move them to the configured device.

    Uses the official ``openai-whisper`` package to download and cache models.
    Both models are set to eval mode with inference_mode gradients disabled.

    Args:
        config: Decoded configuration specifying model names and device.

    Returns:
        A ``ModelPair`` ready for inference.
    """
    raise NotImplementedError("TODO: implement load_models")


def get_encoder_features(
    model_pair: ModelPair,
    mel: torch.Tensor,
) -> torch.Tensor:
    """Run the audio encoder and return cached encoder features.

    Uses the **final** (Large V3) model's encoder only — both decoders
    operate on the same high-quality encoder output.  This avoids a
    redundant forward pass through the draft encoder.

    Args:
        model_pair: Loaded model pair.
        mel: Log-mel spectrogram tensor of shape ``(batch, n_mels, T)``.

    Returns:
        Encoder output tensor of shape ``(batch, T', d_model)``.
    """
    raise NotImplementedError("TODO: implement get_encoder_features")
