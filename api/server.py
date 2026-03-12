"""FastAPI REST interface for speculative Whisper transcription."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from speculative_whisper.core import SpeculativeWhisper

logger = logging.getLogger(__name__)


# =====================================================================
# Response schemas
# =====================================================================


class TranscriptionItem(BaseModel):
    """Single file transcription result."""

    file: str
    text: str
    latency_s: float
    acceptance_rate: Optional[float] = Field(None, description="Draft acceptance rate (speculative only).")
    num_tokens: Optional[int] = Field(None, description="Number of tokens generated.")


class TranscriptionResponse(BaseModel):
    """Batch transcription response."""

    results: List[TranscriptionItem]
    total_files: int
    total_latency_s: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    draft_model: Optional[str] = None
    final_model: Optional[str] = None
    device: Optional[str] = None


# =====================================================================
# Application lifespan (model loading)
# =====================================================================

# Module-level reference — populated during lifespan startup.
_whisper_instance: Optional[SpeculativeWhisper] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once on startup, release on shutdown."""
    global _whisper_instance  # noqa: PLW0603

    logger.info("Loading SpeculativeWhisper models on startup...")
    draft_model = os.environ.get("WHISPER_DRAFT_MODEL", "tiny")
    final_model = os.environ.get("WHISPER_FINAL_MODEL", "large-v3")
    device = os.environ.get("WHISPER_DEVICE", "auto")

    try:
        _whisper_instance = SpeculativeWhisper(
            draft_model=draft_model,
            final_model=final_model,
            device=device,
        )
        logger.info("Models loaded successfully: draft=%s, final=%s, device=%s",
                   draft_model, final_model, _whisper_instance.model_pair.device)
    except Exception as e:
        logger.error("Failed to load models: %s", e, exc_info=True)
        _whisper_instance = None

    yield

    logger.info("Shutting down — releasing models.")
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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — returns model status."""
    if _whisper_instance is None:
        return HealthResponse(
            status="error",
            model_loaded=False,
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        draft_model=_whisper_instance.config.draft_model,
        final_model=_whisper_instance.config.final_model,
        device=_whisper_instance.model_pair.device,
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    files: List[UploadFile] = File(..., description="Audio files to transcribe."),
    max_tokens: int = Query(200, ge=1, le=1000, description="Max tokens per segment."),
    batch_size: int = Query(1, ge=1, le=32, description="Batch size."),
    use_speculative: bool = Query(True, description="Use speculative decoding."),
    draft_k: Optional[int] = Query(None, ge=1, le=50, description="Draft tokens per iteration."),
    temperature: Optional[float] = Query(None, ge=0.0, le=2.0, description="Sampling temperature."),
    top_p: Optional[float] = Query(None, gt=0.0, le=1.0, description="Nucleus sampling p."),
    sampling_strategy: Optional[str] = Query(None, pattern="^(greedy|top_p)$", description="Sampling strategy."),
) -> TranscriptionResponse:
    """Transcribe uploaded audio files and return text with latency.

    Supports batch uploads. Each file is saved temporarily, transcribed,
    and cleaned up. Query parameters allow runtime config overrides.

    Example:
        curl -X POST "http://localhost:8000/transcribe?draft_k=8&temperature=0.0" \\
             -F "files=@audio1.wav" \\
             -F "files=@audio2.wav"
    """
    if _whisper_instance is None:
        raise HTTPException(status_code=503, detail="Models not loaded — check /health.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results: List[TranscriptionItem] = []
    temp_paths: List[Path] = []
    start_time = time.perf_counter()

    try:
        # Save uploaded files to temporary directory.
        for upload_file in files:
            suffix = Path(upload_file.filename or "audio.wav").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await upload_file.read()
                tmp.write(content)
                temp_paths.append(Path(tmp.name))

        # Transcribe all files — use verbose path to get acceptance_rate and token count.
        outputs = _whisper_instance.transcribe_verbose(
            audio=[str(p) for p in temp_paths],
            max_tokens=max_tokens,
            batch_size=batch_size,
            use_speculative=use_speculative,
            draft_k=draft_k,
            temperature=temperature,
            top_p=top_p,
            sampling_strategy=sampling_strategy,
        )
        if not isinstance(outputs, list):
            outputs = [outputs]

        elapsed = time.perf_counter() - start_time
        per_file_latency = elapsed / len(files)

        for i, (upload_file, out) in enumerate(zip(files, outputs)):
            results.append(
                TranscriptionItem(
                    file=upload_file.filename or f"file_{i}",
                    text=out.text,
                    latency_s=per_file_latency,
                    acceptance_rate=out.acceptance_rate if use_speculative else None,
                    num_tokens=len(out.tokens) if out.tokens else None,
                )
            )

    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

    finally:
        # Clean up temporary files.
        for p in temp_paths:
            try:
                p.unlink()
            except OSError:
                pass

    return TranscriptionResponse(
        results=results,
        total_files=len(files),
        total_latency_s=time.perf_counter() - start_time,
    )


@app.post("/transcribe/single")
async def transcribe_single(
    file: UploadFile = File(..., description="Single audio file to transcribe."),
    max_tokens: int = Query(200, ge=1, le=1000, description="Max tokens per segment."),
    use_speculative: bool = Query(True, description="Use speculative decoding."),
    draft_k: Optional[int] = Query(None, ge=1, le=50, description="Draft tokens per iteration."),
    temperature: Optional[float] = Query(None, ge=0.0, le=2.0, description="Sampling temperature."),
    top_p: Optional[float] = Query(None, gt=0.0, le=1.0, description="Nucleus sampling p."),
    sampling_strategy: Optional[str] = Query(None, pattern="^(greedy|top_p)$", description="Sampling strategy."),
) -> TranscriptionItem:
    """Transcribe a single audio file (convenience endpoint).

    Example:
        curl -X POST "http://localhost:8000/transcribe/single?draft_k=5" \\
             -F "file=@audio.wav"
    """
    response = await transcribe(
        files=[file],
        max_tokens=max_tokens,
        batch_size=1,
        use_speculative=use_speculative,
        draft_k=draft_k,
        temperature=temperature,
        top_p=top_p,
        sampling_strategy=sampling_strategy,
    )
    return response.results[0]


# =====================================================================
# Error handlers
# =====================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all for unhandled exceptions."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )


# =====================================================================
# Entry point (for uvicorn)
# =====================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
