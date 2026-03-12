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
    ) -> None:
        """Initialise models and configuration.

        Args:
            draft_model: Whisper model name for drafting (e.g. ``"tiny"``).
            final_model: Whisper model name for verification (e.g. ``"large-v3"``).
            device: ``"cuda"``, ``"cpu"``, or ``"auto"`` (picks CUDA when available).
            config_path: Optional path to a YAML config file.  Fields supplied
                here (``draft_model``, ``final_model``, ``device``) override the
                YAML values.
        """
        raise NotImplementedError("TODO: implement __init__")

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
        raise NotImplementedError("TODO: implement transcribe")

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
