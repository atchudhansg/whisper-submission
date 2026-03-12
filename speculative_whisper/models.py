"""Model loading, caching, and device management for Whisper model pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import whisper
from whisper.tokenizer import get_tokenizer

from speculative_whisper.config import DecodingConfig

logger = logging.getLogger(__name__)

# Valid model names from the openai-whisper package.
VALID_MODELS = whisper.available_models()


@dataclass
class ModelPair:
    """Holds both the draft and final Whisper models on the target device.

    Attributes:
        draft: The lightweight draft model (e.g. Whisper Tiny).
        final: The high-quality verification model (e.g. Whisper Large V3).
        device: Resolved device string ('cuda' or 'cpu').
        dtype: Tensor dtype matching the device (float16 for cuda, float32 for cpu).
        tokenizer: Shared tokenizer derived from the final model.
    """

    draft: whisper.Whisper
    final: whisper.Whisper
    device: str
    dtype: torch.dtype
    tokenizer: object  # whisper.tokenizer.Tokenizer


def _load_single_model(
    name: str,
    device: str,
    fp16: bool,
) -> whisper.Whisper:
    """Load one Whisper model, set to eval mode, and optionally cast to fp16."""
    if name not in VALID_MODELS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(VALID_MODELS)}"
        )

    logger.info("Loading Whisper model %r on %s (fp16=%s)", name, device, fp16)
    model = whisper.load_model(name, device=device)

    # Cast to fp16 when running on CUDA for memory / speed.
    if fp16 and device != "cpu":
        model = model.half()
        # Whisper's LayerNorm wrapper casts its input to float32 before calling
        # PyTorch's layer_norm.  Newer PyTorch (≥2.1) strictly requires that the
        # input dtype matches the weight/bias dtype, so we restore all LayerNorm
        # parameters to float32 after the half-precision cast.
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.float()

    model.eval()
    return model


def load_models(config: DecodingConfig) -> ModelPair:
    """Load both Whisper models and move them to the configured device.

    Uses the official ``openai-whisper`` package to download and cache models.
    Both models are set to eval mode with gradients disabled.

    Args:
        config: Decoded configuration specifying model names and device.

    Returns:
        A ``ModelPair`` ready for inference.
    """
    device = config.device_resolved
    fp16 = device != "cpu"

    final_model = _load_single_model(config.final_model, device, fp16)
    draft_model = _load_single_model(config.draft_model, device, fp16)

    # Disable gradient computation globally for both models.
    for param in final_model.parameters():
        param.requires_grad_(False)
    for param in draft_model.parameters():
        param.requires_grad_(False)

    # CUDA-specific optimisations.
    if device != "cpu":
        torch.backends.cudnn.benchmark = True
        # SDPA (scaled-dot-product attention) is enabled by default in
        # newer PyTorch; ensure Whisper uses it.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Build a shared tokenizer from the final (authoritative) model.
    tokenizer = get_tokenizer(
        multilingual=final_model.is_multilingual,
        num_languages=final_model.num_languages,
        language=config.language,
        task="transcribe",
    )

    logger.info(
        "Models loaded — draft: %s (%s params), final: %s (%s params)",
        config.draft_model,
        f"{sum(p.numel() for p in draft_model.parameters()) / 1e6:.1f}M",
        config.final_model,
        f"{sum(p.numel() for p in final_model.parameters()) / 1e6:.1f}M",
    )

    return ModelPair(
        draft=draft_model,
        final=final_model,
        device=device,
        dtype=config.dtype,
        tokenizer=tokenizer,
    )


@torch.inference_mode()
def get_encoder_features(
    model_pair: ModelPair,
    mel: torch.Tensor,
) -> torch.Tensor:
    """Run the audio encoder and return encoder features.

    Uses the **final** (Large V3) model's encoder only — both decoders
    operate on the same high-quality encoder output.  This avoids a
    redundant forward pass through the draft encoder.

    Args:
        model_pair: Loaded model pair.
        mel: Log-mel spectrogram tensor of shape ``(batch, n_mels, T)``.
            Will be cast to the model's dtype/device if necessary.

    Returns:
        Encoder output tensor of shape ``(batch, 1500, d_model)``.
    """
    mel = mel.to(device=model_pair.device, dtype=model_pair.dtype)
    return model_pair.final.encoder(mel)
