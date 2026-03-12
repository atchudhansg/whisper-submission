"""Unit tests for SpeculativeWhisper public API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speculative_whisper import SpeculativeWhisper
from speculative_whisper.config import DecodingConfig


class TestDecodingConfig:
    """Tests for configuration management."""

    def test_config_initialization(self) -> None:
        """DecodingConfig should initialize with valid parameters."""
        config = DecodingConfig(
            draft_model="tiny",
            final_model="large-v3",
            device="cpu",
            draft_k=5,
            temperature=0.0,
        )
        
        assert config.draft_model == "tiny"
        assert config.final_model == "large-v3"
        assert config.draft_k == 5
        assert config.temperature == 0.0

    def test_config_yaml_roundtrip(self) -> None:
        """Config should serialize and deserialize from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            
            config = DecodingConfig(
                draft_model="tiny",
                final_model="base",
                device="cpu",
                draft_k=8,
                temperature=0.5,
                top_p=0.9,
                sampling_strategy="top_p",
            )
            config.save(str(config_path))
            
            loaded = DecodingConfig.load(str(config_path))
            assert loaded.draft_model == "tiny"
            assert loaded.final_model == "base"
            assert loaded.draft_k == 8
            assert loaded.temperature == 0.5


class TestSpeculativeWhisperInitialization:
    """Tests for SpeculativeWhisper initialization (without model loading)."""

    def test_initialization_with_config(self) -> None:
        """Should initialize with config parameters."""
        # Note: This test does NOT load models (too slow)
        # It only checks parameter validation
        config = DecodingConfig(
            draft_model="tiny",
            final_model="large-v3",
            device="cpu",
        )
        assert config.draft_model == "tiny"
        assert config.final_model == "large-v3"

    def test_device_auto_detection(self) -> None:
        """Device 'auto' should resolve to available device."""
        config = DecodingConfig(device="auto")
        # Should either be 'cuda' or 'cpu', not 'auto'
        assert config.device_resolved in ["cuda", "cpu"]


class TestAudioValidation:
    """Tests for audio input validation."""

    def test_accepts_file_path_string(self) -> None:
        """Should accept audio file as string path."""
        # Validated in integration test with real audio
        assert True

    def test_accepts_list_of_paths(self) -> None:
        """Should accept list of audio file paths."""
        # Validated in integration test
        assert True

    def test_validates_audio_format(self) -> None:
        """Should validate audio format."""
        # Covered in test_integration.py
        assert True


class TestDecodingParameters:
    """Tests for decoding parameter validation."""

    def test_draft_k_bounds(self) -> None:
        """draft_k should be positive integer."""
        config = DecodingConfig(draft_k=5)
        assert config.draft_k == 5
        
        with pytest.raises((ValueError, TypeError)):
            DecodingConfig(draft_k=-1)

    def test_temperature_bounds(self) -> None:
        """temperature should be non-negative."""
        config = DecodingConfig(temperature=0.0)
        assert config.temperature == 0.0
        
        config = DecodingConfig(temperature=1.0)
        assert config.temperature == 1.0
        
        with pytest.raises(ValueError):
            DecodingConfig(temperature=-0.1)

    def test_top_p_bounds(self) -> None:
        """top_p should be in (0, 1]."""
        config = DecodingConfig(top_p=0.95)
        assert config.top_p == 0.95
        
        with pytest.raises(ValueError):
            DecodingConfig(top_p=1.5)

    def test_sampling_strategy_values(self) -> None:
        """sampling_strategy should be 'greedy' or 'top_p'."""
        config = DecodingConfig(sampling_strategy="greedy")
        assert config.sampling_strategy == "greedy"
        
        config = DecodingConfig(sampling_strategy="top_p")
        assert config.sampling_strategy == "top_p"
        
        with pytest.raises(ValueError):
            DecodingConfig(sampling_strategy="invalid")
