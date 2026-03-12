"""FastAPI REST interface for speculative Whisper transcription."""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Query, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =====================================================================
# Response schemas
# =====================================================================


class TranscriptionItem(BaseModel):
    """Single file transcription result."""

    file: str
    text: str
    latency_s: float


class TranscriptionResponse(BaseModel):
    """Batch transcription response."""

    results: List[TranscriptionItem]


# =====================================================================
# Application lifespan (model loading)
# =====================================================================

# Module-level reference — populated during lifespan startup.
_whisper_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once on startup, release on shutdown."""
    global _whisper_instance  # noqa: PLW0603
    raise NotImplementedError("TODO: implement lifespan — load SpeculativeWhisper here")
    yield
    _whisper_instance = None


app = FastAPI(
    title="Speculative Whisper API",
    description="Audio transcription via speculative decoding (Tiny → Large V3).",
    version="0.1.0",
    lifespan=lifespan,
)


# =====================================================================
# Endpoints
# =====================================================================


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    files: List[UploadFile] = File(..., description="Audio files to transcribe."),
    max_tokens: int = Query(200, ge=1, description="Max tokens per segment."),
    batch_size: int = Query(1, ge=1, description="Batch size."),
    use_speculative: bool = Query(True, description="Use speculative decoding."),
) -> TranscriptionResponse:
    """Transcribe uploaded audio files and return text with latency."""
    raise NotImplementedError("TODO: implement /transcribe endpoint")


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "ok", "model_loaded": _whisper_instance is not None}
