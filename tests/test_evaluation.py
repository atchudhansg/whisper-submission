"""Unit tests for WER computation and evaluation utilities."""

from __future__ import annotations

import pytest

from speculative_whisper.evaluation import compute_wer, compute_wer_batch


class TestComputeWer:
    """Tests for single-utterance WER."""

    def test_wer_identical(self) -> None:
        """WER should be 0.0 when reference and hypothesis match exactly."""
        ref = "hello world"
        hyp = "hello world"
        assert compute_wer(ref, hyp) == 0.0

    def test_wer_known_value(self) -> None:
        """WER should match a hand-computed value for a known edit distance."""
        ref = "hello world"
        hyp = "hello earth"  # 1 substitution out of 2 words = 0.5 WER
        wer = compute_wer(ref, hyp)
        assert wer == 0.5

    def test_wer_empty_reference(self) -> None:
        """Should handle empty reference string gracefully."""
        assert compute_wer("", "") == 0.0
        assert compute_wer("", "hello") == 1.0
        assert compute_wer("hello", "") == 1.0

    def test_wer_case_insensitive(self) -> None:
        """WER should be case-insensitive."""
        ref = "Hello World"
        hyp = "hello world"
        assert compute_wer(ref, hyp) == 0.0


class TestComputeWerBatch:
    """Tests for corpus-level WER."""

    def test_batch_wer_perfect(self) -> None:
        """Corpus-level WER should be 0.0 when all pairs match."""
        refs = ["hello world", "goodbye moon"]
        hyps = ["hello world", "goodbye moon"]
        assert compute_wer_batch(refs, hyps) == 0.0

    def test_batch_wer_consistent(self) -> None:
        """Corpus WER should be consistent with individual WER values."""
        refs = ["hello world", "goodbye moon"]
        hyps = ["hello earth", "goodbye moon"]  # 1 error out of 4 total words = 0.25 WER
        wer = compute_wer_batch(refs, hyps)
        assert wer == 0.25

    def test_batch_wer_empty(self) -> None:
        """Should handle empty batch gracefully."""
        assert compute_wer_batch([], []) == 0.0

    def test_batch_wer_mismatched_lengths(self) -> None:
        """Should raise ValueError for mismatched input lengths."""
        refs = ["hello"]
        hyps = ["hello", "world"]
        with pytest.raises(ValueError, match="Mismatched lengths"):
            compute_wer_batch(refs, hyps)
