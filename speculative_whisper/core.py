"""SpeculativeWhisper — public API wrapper.

Drop-in replacement for ``whisper.transcribe()`` that transparently uses
speculative decoding (Tiny drafts, Large V3 verifies) for faster inference
with the exact same output distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from speculative_whisper.config import DecodingConfig
from speculative_whisper.models import ModelPair, load_models

logger = logging.getLogger(__name__)


class SpeculativeWhisper:
    """High-level interface for speculative Whisper transcription.

    Example::

        sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3", device="cuda")
        texts = sw.transcribe(["audio1.wav", "audio2.wav"], batch_size=2)

    """

    def __init__(
        self,
        draft_model: str = "tiny",
        final_model: str = "large-v3",
        device: str = "auto",
        config_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        """Initialise models and configuration.

        Loads both Whisper models onto the target device, builds the shared
        tokenizer, and stores the validated configuration.

        Args:
            draft_model: Whisper model name for drafting (e.g. ``"tiny"``).
            final_model: Whisper model name for verification (e.g. ``"large-v3"``).
            device: ``"cuda"``, ``"cpu"``, or ``"auto"`` (picks CUDA when available).
            config_path: Optional path to a YAML config file.  Fields supplied
                here (``draft_model``, ``final_model``, ``device``) override the
                YAML values.
            **kwargs: Any additional :class:`DecodingConfig` fields (e.g.
                ``sampling_strategy``, ``top_p``, ``temperature``).
        """
        # Build config — YAML first (if provided), then override with explicit args.
        if config_path is not None:
            self.config = DecodingConfig.from_yaml(config_path)
        else:
            self.config = DecodingConfig()

        # Explicit constructor args take precedence over YAML values.
        overrides = {"draft_model": draft_model, "final_model": final_model, "device": device}
        overrides.update(kwargs)
        # Re-construct to ensure model validators (e.g. sampling_strategy defaults) fire.
        merged = self.config.model_dump()
        merged.update(overrides)
        self.config = DecodingConfig(**merged)

        # Load both models onto the resolved device.
        self.model_pair: ModelPair = load_models(self.config)

        logger.info(
            "SpeculativeWhisper ready — device=%s, draft=%s, final=%s",
            self.model_pair.device,
            self.config.draft_model,
            self.config.final_model,
        )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, List[Union[str, Path]]],
        max_tokens: int = 200,
        batch_size: int = 1,
        use_speculative: bool = True,
    ) -> Union[str, List[str]]:
        """Transcribe one or more audio files.

        Args:
            audio: A single file path or a list of file paths.
            max_tokens: Maximum number of tokens to generate per segment.
            batch_size: Number of files to decode in each batch.
            use_speculative: If True, use speculative decoding.
                If False, fall back to standard Large V3 greedy decoding.

        Returns:
            A single transcription string (if one file given) or a list of
            transcription strings (if multiple files given).
        """
        from speculative_whisper.audio import compute_mel, load_audio
        from speculative_whisper.decoding import baseline_decode, speculative_decode

        # Normalise input to a list.
        single_input = isinstance(audio, (str, Path))
        paths = [audio] if single_input else list(audio)

        # Override max_tokens in a per-call config copy.
        call_config = self.config.model_copy(update={"max_tokens": max_tokens})

        results: List[str] = []

        # Process in batches (decoding is per-sample, batching is for throughput).
        for batch_start in range(0, len(paths), batch_size):
            batch_paths = paths[batch_start : batch_start + batch_size]

            for p in batch_paths:
                audio_data = load_audio(p)
                # Compute mel spectrograms for each model separately.
                # Large uses n_mels=128, Tiny uses n_mels=80 — they differ.
                mel = compute_mel(
                    audio_data,
                    device=self.model_pair.device,
                    dtype=self.model_pair.dtype,
                    n_mels=self.model_pair.final.dims.n_mels,
                ).unsqueeze(0)  # (1, n_mels_final, T)

                # Pre-compute draft mel only when n_mels differ to avoid
                # the redundant forward pass on same-architecture pairs.
                draft_n_mels = self.model_pair.draft.dims.n_mels
                final_n_mels = self.model_pair.final.dims.n_mels
                if draft_n_mels != final_n_mels:
                    mel_draft = compute_mel(
                        audio_data,
                        device=self.model_pair.device,
                        dtype=self.model_pair.dtype,
                        n_mels=draft_n_mels,
                    ).unsqueeze(0)  # (1, n_mels_draft, T)
                else:
                    mel_draft = None

                if use_speculative:
                    output = speculative_decode(self.model_pair, mel, call_config, mel_draft=mel_draft)
                else:
                    output = baseline_decode(self.model_pair, mel, call_config)

                logger.info(
                    "%s → %d tokens, acceptance=%.2f",
                    Path(p).name,
                    len(output.tokens),
                    output.acceptance_rate,
                )
                results.append(output.text)

        return results[0] if single_input else results

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        audio_paths: List[Union[str, Path]],
        references: List[str],
    ) -> Dict:
        """Run both speculative and baseline decoding and return comparison metrics.

        Args:
            audio_paths: Paths to audio files.
            references: Ground-truth transcriptions aligned with ``audio_paths``.

        Returns:
            Dictionary with keys ``"speculative"`` and ``"baseline"``, each
            containing latency and WER statistics.
        """
        raise NotImplementedError("TODO: implement benchmark")
