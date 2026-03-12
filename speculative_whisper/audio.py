"""Audio loading, preprocessing, and batching for Whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
import whisper
from whisper.audio import SAMPLE_RATE, N_FRAMES, log_mel_spectrogram, pad_or_trim

logger = logging.getLogger(__name__)


def load_audio(path: Union[str, Path], sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to the target sample rate.

    Delegates to ``whisper.load_audio`` which uses ffmpeg under the hood,
    supporting wav, mp3, flac, ogg, and most other audio formats.

    Args:
        path: Path to the audio file.
        sr: Target sample rate in Hz. Whisper expects 16 000.

    Returns:
        1-D float32 numpy array of audio samples.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If ffmpeg fails to decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return whisper.load_audio(str(path), sr=sr)


def compute_mel(
    audio: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    n_mels: int = 80,
) -> torch.Tensor:
    """Compute a log-mel spectrogram from raw audio.

    Pads or trims to Whisper's fixed 30-second window before computing
    the spectrogram.

    Args:
        audio: 1-D float32 numpy array at 16 kHz.
        device: Target device for the output tensor.
        dtype: Target dtype (float16 on cuda, float32 on cpu).
        n_mels: Number of mel frequency bins (80 for most models, 128 for large-v3).

    Returns:
        Tensor of shape ``(n_mels, N_FRAMES)`` — a single-sample mel spectrogram.
    """
    # Pad or trim to exactly 30 seconds (480 000 samples).
    audio = pad_or_trim(audio)
    # Compute log-mel spectrogram and move to target device/dtype.
    mel = log_mel_spectrogram(audio, n_mels=n_mels, device=device)
    return mel.to(dtype=dtype)


def batch_mels(
    paths: Sequence[Union[str, Path]],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    n_mels: int = 80,
) -> torch.Tensor:
    """Load multiple audio files and return a batched mel spectrogram tensor.

    Each file is loaded, resampled to 16 kHz, padded/trimmed to 30 s,
    converted to a log-mel spectrogram, and stacked into a single batch.

    Args:
        paths: Sequence of audio file paths.
        device: Target device.
        dtype: Target dtype.
        n_mels: Number of mel frequency bins.

    Returns:
        Tensor of shape ``(batch, n_mels, N_FRAMES)``.

    Raises:
        ValueError: If ``paths`` is empty.
    """
    if not paths:
        raise ValueError("paths must be a non-empty sequence")

    mels = []
    for p in paths:
        audio = load_audio(p)
        mel = compute_mel(audio, device=device, dtype=dtype, n_mels=n_mels)
        mels.append(mel)

    return torch.stack(mels, dim=0)
