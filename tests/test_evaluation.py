"""Unit tests for WER computation and evaluation utilities."""

from __future__ import annotations

import pytest

from speculative_whisper.evaluation import compute_wer, compute_wer_batch


class TestComputeWer:
    """Tests for single-utterance WER."""

    def test_wer_identical(self) -> None:
        """WER should be 0.0 when reference and hypothesis match exactly."""
        pytest.skip("TODO: implement after compute_wer is written")

    def test_wer_known_value(self) -> None:
        """WER should match a hand-computed value for a known edit distance."""
        pytest.skip("TODO: implement after compute_wer is written")

    def test_wer_empty_reference(self) -> None:
        """Should handle empty reference string gracefully."""
        pytest.skip("TODO: implement after compute_wer is written")


class TestComputeWerBatch:
    """Tests for corpus-level WER."""

    def test_batch_wer_perfect(self) -> None:
        """Corpus-level WER should be 0.0 when all pairs match."""
        pytest.skip("TODO: implement after compute_wer_batch is written")

    def test_batch_wer_consistent(self) -> None:
        """Corpus WER should be consistent with individual WER values."""
        pytest.skip("TODO: implement after compute_wer_batch is written")
