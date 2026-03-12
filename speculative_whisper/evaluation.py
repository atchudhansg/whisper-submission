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
    try:
        import jiwer
    except ImportError:
        raise ImportError("jiwer library required for WER computation. Install with: pip install jiwer")
    
    # Normalize strings
    ref = reference.strip().lower() if reference else ""
    hyp = hypothesis.strip().lower() if hypothesis else ""
    
    # Handle edge cases
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0  # No reference but hypothesis exists
    if not hyp:
        return 1.0  # Reference exists but no hypothesis
    
    return float(jiwer.wer(ref, hyp))


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
    try:
        import jiwer
    except ImportError:
        raise ImportError("jiwer library required for WER computation. Install with: pip install jiwer")
    
    if len(references) != len(hypotheses):
        raise ValueError(f"Mismatched lengths: {len(references)} references, {len(hypotheses)} hypotheses")
    
    if not references:
        return 0.0
    
    # Normalize all strings
    refs = [ref.strip().lower() if ref else "" for ref in references]
    hyps = [hyp.strip().lower() if hyp else "" for hyp in hypotheses]
    
    # Compute corpus-level WER (concatenates all reference/hypothesis pairs)
    return float(jiwer.wer(refs, hyps))


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
    import time
    from statistics import median
    
    from speculative_whisper.audio import compute_mel, load_audio
    from speculative_whisper.decoding import baseline_decode, speculative_decode
    
    if len(audio_paths) != len(references):
        raise ValueError(f"Mismatched lengths: {len(audio_paths)} audio files, {len(references)} references")
    
    spec_latencies = []
    spec_texts = []
    spec_acceptance_rates = []
    
    base_latencies = []
    base_texts = []
    
    logger.info(f"Benchmarking {len(audio_paths)} audio files...")
    
    for i, (audio_path, reference) in enumerate(zip(audio_paths, references)):
        logger.info(f"Processing {i+1}/{len(audio_paths)}: {audio_path}")
        
        # Load audio and compute mel spectrograms
        audio_data = load_audio(audio_path)
        mel_final = compute_mel(
            audio_data,
            device=model_pair.device,
            dtype=model_pair.dtype,
            n_mels=model_pair.final.dims.n_mels,
        ).unsqueeze(0)
        
        # Handle different mel dimensions for draft model
        draft_n_mels = model_pair.draft.dims.n_mels
        final_n_mels = model_pair.final.dims.n_mels
        if draft_n_mels != final_n_mels:
            mel_draft = compute_mel(
                audio_data,
                device=model_pair.device,
                dtype=model_pair.dtype,
                n_mels=draft_n_mels,
            ).unsqueeze(0)
        else:
            mel_draft = None
        
        # Baseline decoding
        start_time = time.perf_counter()
        base_output = baseline_decode(model_pair, mel_final, config)
        base_latency = time.perf_counter() - start_time
        
        base_latencies.append(base_latency)
        base_texts.append(base_output.text)
        
        # Speculative decoding
        start_time = time.perf_counter()
        spec_output = speculative_decode(model_pair, mel_final, config, mel_draft=mel_draft)
        spec_latency = time.perf_counter() - start_time
        
        spec_latencies.append(spec_latency)
        spec_texts.append(spec_output.text)
        spec_acceptance_rates.append(spec_output.acceptance_rate)
    
    # Compute WER for both methods
    spec_wer = compute_wer_batch(references, spec_texts)
    base_wer = compute_wer_batch(references, base_texts)
    
    # Compute latency statistics (median and 95th percentile)
    def percentile_95(values):
        sorted_vals = sorted(values)
        idx = min(int(0.95 * len(sorted_vals)), len(sorted_vals) - 1)
        return sorted_vals[idx]
    
    spec_result = BenchmarkResult(
        method="speculative",
        latency_median_s=median(spec_latencies),
        latency_p95_s=percentile_95(spec_latencies),
        wer=spec_wer,
        acceptance_rate=sum(spec_acceptance_rates) / len(spec_acceptance_rates) if spec_acceptance_rates else 0.0,
    )
    
    base_result = BenchmarkResult(
        method="baseline",
        latency_median_s=median(base_latencies),
        latency_p95_s=percentile_95(base_latencies),
        wer=base_wer,
        acceptance_rate=None,  # No speculative decoding in baseline
    )
    
    logger.info("Benchmark complete:")
    logger.info(spec_result.summary())
    logger.info(base_result.summary())
    
    return spec_result, base_result
