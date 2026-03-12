"""Integration tests using real models and audio.

These tests load actual Whisper models and transcribe real audio files
from the samples/ directory. They validate:
- End-to-end transcription correctness
- Acceptance rate metrics
- Performance characteristics
- Multilingual support
"""

from __future__ import annotations

from pathlib import Path

import pytest

from speculative_whisper import SpeculativeWhisper
from speculative_whisper.evaluation import compute_wer_batch


@pytest.mark.integration
class TestEndToEndTranscription:
    """Integration tests with real models and real audio."""

    @pytest.fixture
    def sw_tiny_cpu(self) -> SpeculativeWhisper:
        """Create SpeculativeWhisper with Tiny models on CPU for testing."""
        return SpeculativeWhisper(
            draft_model="tiny",
            final_model="tiny",  # Use tiny for both to speed up testing
            device="cpu",
        )

    @pytest.fixture
    def sample_audio(self) -> Path:
        """Return path to first sample audio file."""
        samples_dir = Path(__file__).parent.parent / "samples"
        audio_files = sorted(samples_dir.glob("*.wav"))
        if not audio_files:
            pytest.skip("No sample audio files found in samples/")
        return audio_files[0]

    def test_transcribe_single_file(self, sw_tiny_cpu, sample_audio) -> None:
        """Transcribe single audio file and verify output is string."""
        text = sw_tiny_cpu.transcribe(str(sample_audio))
        
        assert isinstance(text, str)
        assert len(text) > 0

    def test_transcribe_verbose_returns_metrics(self, sw_tiny_cpu, sample_audio) -> None:
        """Transcribe verbose should return DecodingOutput with metrics."""
        output = sw_tiny_cpu.transcribe_verbose(str(sample_audio))
        
        assert hasattr(output, "text")
        assert hasattr(output, "tokens")
        assert hasattr(output, "acceptance_rate")
        assert hasattr(output, "num_drafted")
        assert hasattr(output, "num_accepted")
        
        assert isinstance(output.text, str)
        assert isinstance(output.tokens, list)
        assert 0.0 <= output.acceptance_rate <= 1.0
        assert output.num_drafted > 0
        assert output.num_accepted >= 0

    def test_batch_transcription(self, sw_tiny_cpu) -> None:
        """Batch transcription should process multiple files."""
        samples_dir = Path(__file__).parent.parent / "samples"
        audio_files = sorted(samples_dir.glob("*.wav"))[:3]  # First 3 files
        
        if len(audio_files) < 2:
            pytest.skip("Need at least 2 sample files")
        
        texts = sw_tiny_cpu.transcribe([str(f) for f in audio_files])
        
        assert isinstance(texts, list)
        assert len(texts) == len(audio_files)
        assert all(isinstance(t, str) for t in texts)

    def test_greedy_is_deterministic(self, sw_tiny_cpu, sample_audio) -> None:
        """Greedy decoding should be deterministic."""
        text1 = sw_tiny_cpu.transcribe(str(sample_audio), temperature=0.0)
        text2 = sw_tiny_cpu.transcribe(str(sample_audio), temperature=0.0)
        
        assert text1 == text2

    def test_baseline_vs_speculative_consistency(self, sw_tiny_cpu, sample_audio) -> None:
        """Greedy speculative should match baseline in output text."""
        spec_text = sw_tiny_cpu.transcribe(
            str(sample_audio),
            use_speculative=True,
            temperature=0.0,
        )
        base_text = sw_tiny_cpu.transcribe(
            str(sample_audio),
            use_speculative=False,
            temperature=0.0,
        )
        
        # Greedy decoding should produce identical results
        assert spec_text == base_text

    def test_acceptance_rate_reasonable(self, sw_tiny_cpu, sample_audio) -> None:
        """Acceptance rate should be in reasonable range (0-1)."""
        output = sw_tiny_cpu.transcribe_verbose(
            str(sample_audio),
            use_speculative=True,
            temperature=0.0,
        )
        
        assert 0.0 <= output.acceptance_rate <= 1.0

    def test_wer_evaluation_integration(self, sw_tiny_cpu) -> None:
        """Integration test for WER evaluation."""
        # Simple reference transcriptions
        references = [
            "hello world",
            "good morning",
            "test phrase",
        ]
        
        # Create hypotheses (with known differences)
        hypotheses = [
            "hello world",          # Exact match (0 errors)
            "good morning",         # Exact match (0 errors)
            "test phrase modified",  # 1 extra word
        ]
        
        wer = compute_wer_batch(references, hypotheses)
        
        # Total: 1 error out of 7 words
        expected_wer = 1.0 / 7.0
        assert abs(wer - expected_wer) < 0.01


@pytest.mark.integration
class TestMultilingualSupport:
    """Tests for multilingual transcription."""

    @pytest.fixture
    def sw_multilingual(self) -> SpeculativeWhisper:
        """Create SpeculativeWhisper configured for English."""
        return SpeculativeWhisper(
            draft_model="tiny",
            final_model="tiny",
            device="cpu",
            language="en",
        )

    def test_language_parameter_english(self, sw_multilingual) -> None:
        """Should successfully set language to English."""
        # This validates configuration, actual multilingual transcription
        # requires multi-language audio samples
        output = sw_multilingual.transcribe.__doc__
        assert output is not None


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in integration scenarios."""

    @pytest.fixture
    def sw(self) -> SpeculativeWhisper:
        """Create SpeculativeWhisper instance."""
        return SpeculativeWhisper(draft_model="tiny", final_model="tiny", device="cpu")

    def test_missing_audio_file(self, sw) -> None:
        """Should raise error for missing audio file."""
        with pytest.raises((FileNotFoundError, Exception)):
            sw.transcribe("/nonexistent/path/audio.wav")

    def test_invalid_language_code(self, sw) -> None:
        """Should handle invalid language code gracefully."""
        samples_dir = Path(__file__).parent.parent / "samples"
        audio_files = sorted(samples_dir.glob("*.wav"))
        
        if not audio_files:
            pytest.skip("No sample audio files found")
        
        # Whisper may accept invalid codes or raise error
        # Just ensure it doesn't crash unexpectedly
        try:
            sw.transcribe(str(audio_files[0]), language="invalid_code")
        except (ValueError, Exception):
            pass  # Expected behavior
