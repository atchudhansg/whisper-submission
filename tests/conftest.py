"""Shared pytest fixtures for speculative_whisper tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def dummy_audio() -> np.ndarray:
    """1 second of silence at 16 kHz."""
    return np.zeros(16_000, dtype=np.float32)


@pytest.fixture
def dummy_mel() -> torch.Tensor:
    """Fake mel spectrogram matching Whisper's expected shape.

    Shape: (1, 80, 3000) — batch=1, 80 mels, 30s at 100 frames/s.
    """
    return torch.zeros(1, 80, 3000, dtype=torch.float32)


@pytest.fixture
def dummy_encoder_output() -> torch.Tensor:
    """Fake encoder output for decoder tests.

    Shape: (1, 1500, 1280) — batch=1, T'=1500, d_model=1280 (Large V3 dims).
    """
    return torch.zeros(1, 1500, 1280, dtype=torch.float32)
