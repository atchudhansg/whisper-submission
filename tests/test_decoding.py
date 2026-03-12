"""Unit tests for the speculative decoding logic."""

from __future__ import annotations

import pytest
import torch

from speculative_whisper.decoding import (
    DraftResult,
    accept_reject,
    baseline_decode,
    draft_step,
    score_with_final,
    speculative_decode,
)


class TestAcceptReject:
    """Tests for the rejection sampling accept/reject logic."""

    def test_accept_all_tokens(self) -> None:
        """When the final model assigns equal or higher prob to every draft token,
        all tokens should be accepted."""
        pytest.skip("TODO: implement after accept_reject is written")

    def test_reject_first_token(self) -> None:
        """When the final model strongly disagrees on the first token,
        only one resampled token should be returned."""
        pytest.skip("TODO: implement after accept_reject is written")

    def test_partial_acceptance(self) -> None:
        """Draft of K tokens: first N accepted, rest rejected at position N+1."""
        pytest.skip("TODO: implement after accept_reject is written")

    def test_greedy_temperature_zero(self) -> None:
        """At temperature=0, acceptance should be deterministic (no randomness)."""
        pytest.skip("TODO: implement after accept_reject is written")


class TestDraftStep:
    """Tests for the draft token generation."""

    def test_draft_returns_k_tokens(self) -> None:
        """draft_step should return exactly k tokens when EOS is not hit."""
        pytest.skip("TODO: implement after draft_step is written")

    def test_draft_stops_at_eos(self) -> None:
        """draft_step should stop early if the draft model produces EOS."""
        pytest.skip("TODO: implement after draft_step is written")


class TestScoreWithFinal:
    """Tests for the final model scoring."""

    def test_output_shape(self) -> None:
        """score_with_final should return logits of shape (K, vocab_size)."""
        pytest.skip("TODO: implement after score_with_final is written")


class TestSpeculativeDecode:
    """Integration tests for the full speculative loop."""

    def test_produces_text(self) -> None:
        """speculative_decode should return a non-empty DecodingOutput."""
        pytest.skip("TODO: implement with mock models")

    def test_acceptance_rate_in_range(self) -> None:
        """Acceptance rate should be between 0 and 1."""
        pytest.skip("TODO: implement with mock models")
