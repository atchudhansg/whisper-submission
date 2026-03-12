"""Audio loading, preprocessing, and batching for Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch


def load_audio(path: Union[str, Path], sr: int = 16_000) -> np.ndarray:
    """Load an audio file and resample to the target sample rate.

    Args:
        path: Path to the audio file (wav, mp3, flac, etc.).
        sr: Target sample rate in Hz. Whisper expects 16 000.

    Returns:
        1-D float32 numpy array of audio samples.
    """
    raise NotImplementedError("TODO: implement load_audio")


def compute_mel(
    audio: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute a log-mel spectrogram from raw audio.

    Pads or trims to Whisper's fixed 30-second window before computing
    the spectrogram.

    Args:
        audio: 1-D float32 numpy array at 16 kHz.
        device: Target device for the output tensor.
        dtype: Target dtype (float16 on cuda, float32 on cpu).

    Returns:
        Tensor of shape ``(n_mels, T)`` — a single-sample mel spectrogram.
    """
    raise NotImplementedError("TODO: implement compute_mel")


def batch_mels(
    paths: Sequence[Union[str, Path]],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load multiple audio files and return a batched mel spectrogram tensor.

    Each file is loaded, resampled, padded/trimmed to 30 s, converted to
    a log-mel spectrogram, and stacked into a single batch tensor.

    Args:
        paths: Sequence of audio file paths.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape ``(batch, n_mels, T)``.
    """
    raise NotImplementedError("TODO: implement batch_mels")
