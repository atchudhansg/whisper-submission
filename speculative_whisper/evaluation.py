"""Evaluation utilities — WER computation and latency benchmarking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from speculative_whisper.config import DecodingConfig
from speculative_whisper.models import ModelPair

logger = logging.getLogger(__name__)


# =====================================================================
# Data structures
# =====================================================================


@dataclass
class BenchmarkResult:
    """Aggregated metrics for a single decoding method.

    Attributes:
        method: Human-readable label (e.g. ``"speculative"`` or ``"baseline"``).
        latency_median_s: Median wall-clock latency per sample in seconds.
        latency_p95_s: 95th-percentile latency per sample in seconds.
        wer: Word Error Rate across all evaluated samples (0.0–1.0+).
        acceptance_rate: Mean acceptance rate (speculative only; None for baseline).
    """

    method: str
    latency_median_s: float
    latency_p95_s: float
    wer: float
    acceptance_rate: Optional[float] = None

    def summary(self) -> str:
        """Return a one-line summary suitable for logging."""
        ar = f"  accept={self.acceptance_rate:.2%}" if self.acceptance_rate is not None else ""
        return (
            f"[{self.method}]  median={self.latency_median_s:.3f}s  "
            f"p95={self.latency_p95_s:.3f}s  WER={self.wer:.4f}{ar}"
        )


# =====================================================================
# WER
# =====================================================================


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between a reference and hypothesis string.

    Wraps the ``jiwer`` library for a single utterance pair.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model-generated transcription.

    Returns:
        WER as a float (0.0 = perfect, >1.0 possible for very bad output).
    """
    raise NotImplementedError("TODO: implement compute_wer")


def compute_wer_batch(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> float:
    """Compute corpus-level WER across multiple utterances.

    Args:
        references: Ground-truth transcriptions.
        hypotheses: Model-generated transcriptions (same length).

    Returns:
        Corpus-level WER as a float.
    """
    raise NotImplementedError("TODO: implement compute_wer_batch")


# =====================================================================
# Benchmarking
# =====================================================================


def benchmark(
    model_pair: ModelPair,
    audio_paths: List[str],
    references: List[str],
    config: DecodingConfig,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Run speculative and baseline decoding, then compare.

    For each audio file:
      1. Decode with speculative decoding — record latency and text.
      2. Decode with standard greedy (baseline) — record latency and text.

    Then compute WER and latency statistics for both methods.

    Args:
        model_pair: Loaded draft + final models.
        audio_paths: Paths to audio files to evaluate.
        references: Ground-truth transcriptions (aligned with audio_paths).
        config: Decoding configuration.

    Returns:
        A tuple ``(speculative_result, baseline_result)``.
    """
    raise NotImplementedError("TODO: implement benchmark")
