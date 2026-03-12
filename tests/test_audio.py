"""Unit tests for audio loading and preprocessing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from speculative_whisper.audio import batch_mels, compute_mel, load_audio


class TestLoadAudio:
    """Tests for audio file loading and resampling."""

    def test_load_audio_returns_float32(self) -> None:
        """Loaded audio should be float32 numpy array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple test audio file
            sr = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            audio = 0.1 * np.sin(2 * np.pi * 440 * t)
            
            audio_path = Path(tmpdir) / "test.wav"
            sf.write(audio_path, audio, sr)
            
            loaded = load_audio(str(audio_path))
            assert isinstance(loaded, np.ndarray)
            assert loaded.dtype == np.float32
            assert len(loaded.shape) == 1

    def test_load_audio_resamples_to_16khz(self) -> None:
        """Loaded audio should be resampled to 16 kHz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create at different sample rate
            sr_orig = 22050
            duration = 1.0
            t = np.linspace(0, duration, int(sr_orig * duration), dtype=np.float32)
            audio = 0.1 * np.sin(2 * np.pi * 440 * t)
            
            audio_path = Path(tmpdir) / "test.wav"
            sf.write(audio_path, audio, sr_orig)
            
            loaded = load_audio(str(audio_path))
            # 1 second at 16 kHz = 16000 samples
            assert len(loaded) == 16000

    def test_load_audio_missing_file(self) -> None:
        """Should raise FileNotFoundError for nonexistent paths."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")


class TestComputeMel:
    """Tests for mel spectrogram computation."""

    def test_mel_shape_80_bins(self) -> None:
        """80-bin mel output should have shape (80, 3000) for 30s of audio."""
        audio = np.zeros(16000 * 30, dtype=np.float32)  # 30 seconds at 16 kHz
        mel = compute_mel(audio, n_mels=80)
        
        assert isinstance(mel, torch.Tensor)
        assert mel.shape == (80, 3000)  # 80 mels, 3000 frames (100/sec)

    def test_mel_shape_128_bins(self) -> None:
        """128-bin mel output should have shape (128, 3000) for 30s of audio."""
        audio = np.zeros(16000 * 30, dtype=np.float32)  # 30 seconds at 16 kHz
        mel = compute_mel(audio, n_mels=128)
        
        assert isinstance(mel, torch.Tensor)
        assert mel.shape == (128, 3000)

    def test_mel_short_audio(self, dummy_audio) -> None:
        """1-second audio should produce 100 frames."""
        mel = compute_mel(dummy_audio, n_mels=80)
        assert mel.shape == (80, 100)  # 80 mels, 100 frames (100/sec)

    def test_mel_is_float32_tensor(self) -> None:
        """Mel spectrogram should be float32 tensor."""
        audio = np.zeros(16000, dtype=np.float32)
        mel = compute_mel(audio, n_mels=80)
        
        assert isinstance(mel, torch.Tensor)
        assert mel.dtype == torch.float32


class TestBatchMels:
    """Tests for batch mel computation."""

    def test_batch_mels_shape(self) -> None:
        """Batch mels should stack correctly."""
        audio1 = np.zeros(16000, dtype=np.float32)  # 1 second
        audio2 = np.zeros(16000, dtype=np.float32)  # 1 second
        
        mel_batch = batch_mels([audio1, audio2], n_mels=80)
        
        assert isinstance(mel_batch, torch.Tensor)
        assert mel_batch.shape[0] == 2  # 2 items in batch
        assert mel_batch.shape[1] == 80  # 80 mel bins
        assert mel_batch.shape[2] == 100  # 100 frames (1s at 100fps)

    def test_batch_mels_consistent_with_single(self) -> None:
        """Single mel should match batch_mels output for single item."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        
        single = compute_mel(audio, n_mels=80)
        batch = batch_mels([audio], n_mels=80)
        
        assert torch.allclose(single.unsqueeze(0), batch, atol=1e-5)


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
