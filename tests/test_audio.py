"""Unit tests for audio loading and preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speculative_whisper.audio import batch_mels, compute_mel, load_audio


class TestLoadAudio:
    """Tests for audio file loading and resampling."""

    def test_load_audio_resamples(self, tmp_path) -> None:
        """Loaded audio should be a 1-D float32 array at 16 kHz."""
        pytest.skip("TODO: implement after load_audio is written")

    def test_load_audio_missing_file(self) -> None:
        """Should raise FileNotFoundError for nonexistent paths."""
        pytest.skip("TODO: implement after load_audio is written")


class TestComputeMel:
    """Tests for mel spectrogram computation."""

    def test_mel_shape(self, dummy_audio) -> None:
        """Output should have shape (80, 3000) for 30s of audio."""
        pytest.skip("TODO: implement after compute_mel is written")

    def test_mel_dtype(self, dummy_audio) -> None:
        """Output dtype should match the requested dtype."""
        pytest.skip("TODO: implement after compute_mel is written")


class TestBatchMels:
    """Tests for batched mel spectrogram creation."""

    def test_batch_padding(self, tmp_path) -> None:
        """All samples in a batch should be padded to the same length."""
        pytest.skip("TODO: implement after batch_mels is written")

    def test_batch_shape(self, tmp_path) -> None:
        """Output shape should be (batch, 80, 3000)."""
        pytest.skip("TODO: implement after batch_mels is written")
