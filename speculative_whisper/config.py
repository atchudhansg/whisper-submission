"""Pydantic-validated configuration for speculative decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DecodingConfig(BaseSettings):
    """All tuneable knobs for speculative Whisper decoding.

    Load from YAML via ``DecodingConfig.from_yaml(path)`` or construct directly.
    """

    # Model selection
    draft_model: str = Field("tiny", description="Whisper model name for drafting.")
    final_model: str = Field("large-v3", description="Whisper model name for verification.")

    # Device
    device: str = Field("auto", description="'cuda', 'cpu', or 'auto'.")

    # Speculative decoding parameters
    draft_k: int = Field(5, ge=1, le=50, description="Tokens to draft per iteration.")
    temperature: float = Field(0.0, ge=0.0, description="Sampling temperature (0 = greedy).")
    max_tokens: int = Field(200, ge=1, description="Max tokens to generate per segment.")

    # Batching
    batch_size: int = Field(1, ge=1, description="Audio files to process together.")

    # Language
    language: str = Field("en", description="Target language code.")

    # Optional sampling
    beam_size: Optional[int] = Field(None, ge=1, description="Beam width (None = disabled).")
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0, description="Nucleus sampling p.")

    model_config = {"env_prefix": "SPEC_WHISPER_", "extra": "ignore"}

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        allowed = {"auto", "cuda", "cpu"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got {v!r}")
        return v

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device_resolved(self) -> str:
        """Resolve ``'auto'`` to the best available device."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @property
    def dtype(self) -> torch.dtype:
        """Use float16 on CUDA, float32 on CPU."""
        return torch.float16 if self.device_resolved == "cuda" else torch.float32

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> DecodingConfig:
        """Load configuration from a YAML file and validate all fields."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)
