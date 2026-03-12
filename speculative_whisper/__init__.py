"""Speculative Whisper — fast, exact transcription via speculative decoding.

Public API:
    - ``SpeculativeWhisper``: High-level transcription wrapper.
    - ``DecodingConfig``: Pydantic-validated configuration.
"""

from __future__ import annotations

from speculative_whisper.config import DecodingConfig
from speculative_whisper.core import SpeculativeWhisper

__version__ = "0.1.0"
__all__ = ["SpeculativeWhisper", "DecodingConfig", "__version__"]
