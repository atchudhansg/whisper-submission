"""Speculative decoding algorithm — the core novel logic.

Implements the draft → verify → accept/reject loop that produces output
*distributionally identical* to standard Whisper Large V3 decoding,
while running significantly faster by amortising draft generation
through the lightweight Tiny model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

import whisper

from speculative_whisper.config import DecodingConfig
from speculative_whisper.models import ModelPair

logger = logging.getLogger(__name__)


# =====================================================================
# Data structures
# =====================================================================


@dataclass
class DraftResult:
    """Output of one draft step from the lightweight model.

    Attributes:
        tokens: List of drafted token IDs (length ≤ k).
        log_probs: Log-probability the draft model assigned to each token.
    """

    tokens: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)


@dataclass
class DecodingOutput:
    """Full result of speculative (or baseline) decoding.

    Attributes:
        text: Decoded transcription string.
        tokens: Final token ID sequence.
        num_drafted: Total tokens proposed by the draft model.
        num_accepted: Total tokens accepted without resampling.
    """

    text: str = ""
    tokens: List[int] = field(default_factory=list)
    num_drafted: int = 0
    num_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens accepted (0.0–1.0)."""
        if self.num_drafted == 0:
            return 0.0
        return self.num_accepted / self.num_drafted


# =====================================================================
# Helpers
# =====================================================================


def _get_initial_tokens(model_pair: ModelPair, config: DecodingConfig) -> List[int]:
    """Build the SOT prefix for decoding (start-of-transcript + language + task).

    For English transcription without timestamps this is typically:
        [SOT, <|en|>, <|transcribe|>, <|notimestamps|>]
    """
    tokenizer = model_pair.tokenizer
    sot_seq = list(tokenizer.sot_sequence)
    # Append <|notimestamps|> to disable timestamp token generation.
    sot_seq.append(tokenizer.no_timestamps)
    return sot_seq


def _sample_token(logits: Tensor, temperature: float) -> int:
    """Sample a single token from logits, respecting temperature."""
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        return int(Categorical(probs=probs).sample().item())


def _get_log_prob(logits: Tensor, token: int, temperature: float) -> float:
    """Compute the log-probability of a specific token under the distribution."""
    if temperature == 0.0:
        # Greedy: treat as 1.0 if argmax, 0.0 otherwise (for accept/reject).
        log_probs = F.log_softmax(logits.float(), dim=-1)
    else:
        log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    return float(log_probs[token].item())


def _build_suppress_tokens(tokenizer) -> List[int]:
    """Build the default set of token IDs to suppress (matches ``whisper.decode``)."""
    suppress = set(tokenizer.non_speech_tokens)
    for attr in ("transcribe", "translate", "sot", "sot_prev", "sot_lm"):
        suppress.add(getattr(tokenizer, attr))
    if hasattr(tokenizer, "no_speech"):
        suppress.add(tokenizer.no_speech)
    return sorted(suppress)


def _apply_top_p_filter(logits: Tensor, top_p: float) -> None:
    """Nucleus (top-p) filtering — zero out tokens outside the probability nucleus.

    Sorts tokens by descending probability, computes the cumulative sum,
    and masks out every token whose cumulative mass exceeds ``top_p``.
    At least one token is always kept.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Identify tokens to remove: those whose cumulative prob exceeds top_p.
    # Shift right so the token that *first* crosses the threshold is kept.
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    # Scatter the mask back to the original ordering.
    mask = sorted_mask.scatter(0, sorted_indices, sorted_mask)
    logits[mask] = float("-inf")


def _apply_logit_filters(
    logits: Tensor,
    tokens_so_far: int,
    sample_begin: int,
    suppress_ids: List[int],
    tokenizer,
    top_p: Optional[float] = None,
) -> None:
    """Apply Whisper's standard logit suppression in-place.

    - **SuppressTokens**: always mask out non-speech / special tokens.
    - **SuppressBlank**: at the very first decoding position only, also
      suppress the space token and EOT so the model is forced to emit
      a real content token first.
    - **Top-p**: if set, truncate the distribution to the probability nucleus.
    """
    logits[suppress_ids] = float("-inf")
    if tokens_so_far == sample_begin:
        blank_ids = tokenizer.encoding.encode(" ") + [tokenizer.eot]
        logits[blank_ids] = float("-inf")
    if top_p is not None:
        _apply_top_p_filter(logits, top_p)


# =====================================================================
# Draft step
# =====================================================================


@torch.inference_mode()
def draft_step(
    draft_model: torch.nn.Module,
    encoder_output: Tensor,
    prefix: List[int],
    k: int,
    temperature: float,
    eot_token: int,
    suppress_ids: List[int],
    sample_begin: int,
    tokenizer,
    top_p: Optional[float] = None,
) -> DraftResult:
    """Sample k tokens from the draft model using a fresh per-call KV cache.

    Installs a temporary KV cache, warms it with the full prefix in one
    parallel forward pass, then issues k single-token forward passes.
    The cache is discarded at the end of the call — no cross-iteration
    state to corrupt.

    Args:
        draft_model: The draft (Tiny) Whisper model.
        encoder_output: Encoder output for the draft model.
        prefix: Full token prefix (SOT + previously generated tokens).
        k: Number of tokens to draft.
        temperature: Sampling temperature.  0 → greedy argmax.
        eot_token: End-of-transcript token ID.
        suppress_ids: Tokens to suppress in logits.
        sample_begin: Index of the first free decoding position.
        tokenizer: Whisper tokenizer.
        top_p: Optional nucleus threshold.

    Returns:
        DraftResult with drafted tokens and their log-probs.
    """
    device = encoder_output.device
    drafted_tokens: List[int] = []
    drafted_log_probs: List[float] = []

    kv_cache, hooks = draft_model.install_kv_cache_hooks()
    try:
        # Warm the cache with the full prefix in one parallel pass.
        # out[0, -1] is conditioned on prefix[-1] -> predicts position len(prefix).
        prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
        out = draft_model.decoder(prefix_tensor, encoder_output, kv_cache=kv_cache)
        next_logits = out[0, -1].clone()  # (vocab,)
        current_len = len(prefix)

        for _ in range(k):
            _apply_logit_filters(
                next_logits, current_len, sample_begin, suppress_ids, tokenizer, top_p
            )
            token_id = _sample_token(next_logits, temperature)
            log_prob = _get_log_prob(next_logits, token_id, temperature)
            drafted_tokens.append(token_id)
            drafted_log_probs.append(log_prob)
            current_len += 1

            if token_id == eot_token:
                break

            # Single-token forward — KV cache handles full history.
            t = torch.tensor([[token_id]], dtype=torch.long, device=device)
            out = draft_model.decoder(t, encoder_output, kv_cache=kv_cache)
            next_logits = out[0, -1].clone()
    finally:
        for h in hooks:
            h.remove()

    return DraftResult(tokens=drafted_tokens, log_probs=drafted_log_probs)


# =====================================================================
# Verification step
# =====================================================================


@torch.inference_mode()
def score_with_final(
    final_model: torch.nn.Module,
    encoder_output: Tensor,
    prefix: List[int],
    draft_tokens: List[int],
) -> Tuple[Tensor, Tensor]:
    """Score all draft tokens plus a bonus position in one forward pass.

    Feeds ``prefix + draft_tokens`` through the final decoder.  The
    Transformer output at position *i* is conditioned on tokens 0..i
    (causal mask), so slicing at ``prefix_len - 1`` gives logits that
    predict each draft token in parallel — no persistent KV cache, no
    cross-iteration offset arithmetic.

    Args:
        final_model: The verification Whisper model.
        encoder_output: Encoder output for the final model.
        prefix: Token prefix before the draft sequence.
        draft_tokens: Draft token IDs to verify.

    Returns:
        ``(draft_logits, bonus_logit)`` where ``draft_logits`` has shape
        ``(k, vocab_size)`` — row *i* predicts ``draft_tokens[i]`` —
        and ``bonus_logit`` has shape ``(vocab_size,)`` predicting the
        token after all draft tokens (free when all are accepted).
    """
    device = encoder_output.device
    prefix_len = len(prefix)
    k = len(draft_tokens)

    all_tokens = prefix + draft_tokens
    token_tensor = torch.tensor([all_tokens], dtype=torch.long, device=device)

    # Single forward pass — no KV cache avoids all cross-iteration state issues.
    logits = final_model.decoder(token_tensor, encoder_output)  # (1, prefix_len+k, vocab)

    # logits[0, prefix_len-1] is conditioned on prefix[-1] -> predicts draft_tokens[0]
    # logits[0, prefix_len-1+i] predicts draft_tokens[i]
    # logits[0, prefix_len-1+k] predicts the bonus token (after all drafts)
    start = prefix_len - 1
    draft_logits = logits[0, start : start + k].clone()   # (k, vocab)
    bonus_logit  = logits[0, start + k].clone()           # (vocab,)

    return draft_logits, bonus_logit


# =====================================================================
# Accept / reject (rejection sampling)
# =====================================================================


def accept_reject(
    draft_result: DraftResult,
    final_logits: Tensor,
    temperature: float,
) -> Tuple[List[int], bool]:
    """Apply rejection sampling to decide which draft tokens to keep.

    Walks through the *K* draft tokens one-by-one.  For each token:
    - Accept deterministically if ``r >= 1``.
    - Otherwise accept with probability ``r``; on rejection resample
      from the final distribution and stop.

    At temperature=0 (greedy): accept iff final argmax == draft token.

    Args:
        draft_result: Tokens and log-probs from the draft model.
        final_logits: ``(K, vocab_size)`` logits from ``score_with_final``.
        temperature: Must match what was used for drafting.

    Returns:
        ``(accepted_tokens, all_accepted)``.
    """
    accepted: List[int] = []

    for i, draft_token in enumerate(draft_result.tokens):
        logits_i = final_logits[i]

        if temperature == 0.0:
            final_choice = int(logits_i.argmax(dim=-1).item())
            if final_choice == draft_token:
                accepted.append(draft_token)
            else:
                accepted.append(final_choice)
                return accepted, False
        else:
            final_log_prob = _get_log_prob(logits_i, draft_token, temperature)
            log_r = final_log_prob - draft_result.log_probs[i]
            r = min(1.0, torch.exp(torch.tensor(log_r)).item())
            if torch.rand(1).item() < r:
                accepted.append(draft_token)
            else:
                final_probs = F.softmax(logits_i.float() / temperature, dim=-1)
                resampled = int(Categorical(probs=final_probs).sample().item())
                accepted.append(resampled)
                return accepted, False

    return accepted, True


# =====================================================================
# Main speculative decode loop
# =====================================================================


@torch.inference_mode()
def speculative_decode(
    model_pair: ModelPair,
    mel: Tensor,
    config: DecodingConfig,
    mel_draft: Optional[Tensor] = None,
) -> DecodingOutput:
    """Run the full speculative decoding loop on a single audio segment.

    Algorithm per iteration:
      1. ``draft_step`` — k single-token passes through Tiny (fresh KV cache).
      2. ``score_with_final`` — one parallel pass through Large; returns
         draft_logits (k, vocab) and bonus_logit (vocab) for free.
      3. ``accept_reject`` — keep matching tokens, take final model's
         correction on first mismatch.
      4. Bonus token — when all k accepted, sample the free bonus_logit.

    Args:
        model_pair: Loaded draft + final models.
        mel: Log-mel spectrogram for the final model, shape ``(1, n_mels, T)``.
        config: Decoding configuration.
        mel_draft: Log-mel spectrogram for the draft model when ``n_mels``
            differs (e.g. Tiny=80 vs Large=128).

    Returns:
        ``DecodingOutput`` with text, tokens, and acceptance statistics.
    """
    tokenizer = model_pair.tokenizer
    eot = tokenizer.eot
    device = model_pair.device

    # 1. Encode audio once per model.
    mel = mel.to(device=device, dtype=model_pair.dtype)
    encoder_output_final = model_pair.final.encoder(mel)

    if mel_draft is not None:
        mel_for_draft = mel_draft.to(device=device, dtype=model_pair.dtype)
    elif model_pair.draft.dims.n_mels == model_pair.final.dims.n_mels:
        mel_for_draft = mel
    else:
        raise ValueError(
            f"Draft model expects n_mels={model_pair.draft.dims.n_mels} but "
            f"final model mel has n_mels={model_pair.final.dims.n_mels}. "
            "Pass mel_draft= with the correct spectrogram for the draft model."
        )

    if model_pair.draft.dims.n_audio_state == model_pair.final.dims.n_audio_state and mel_draft is None:
        encoder_output_draft = encoder_output_final
    else:
        encoder_output_draft = model_pair.draft.encoder(mel_for_draft)

    prefix = _get_initial_tokens(model_pair, config)
    sample_begin = len(prefix)
    suppress_ids = _build_suppress_tokens(tokenizer)

    total_drafted = 0
    total_accepted = 0
    generated_tokens: List[int] = []

    while len(generated_tokens) < config.max_tokens:
        current_prefix = prefix + generated_tokens

        max_ctx = model_pair.final.dims.n_text_ctx
        if len(current_prefix) >= max_ctx:
            logger.warning("Reached max decoder context (%d), stopping.", max_ctx)
            break

        remaining = config.max_tokens - len(generated_tokens)
        k = min(config.draft_k, remaining, max_ctx - len(current_prefix))
        if k <= 0:
            break

        # --- Draft: fresh KV cache per call ---
        draft = draft_step(
            draft_model=model_pair.draft,
            encoder_output=encoder_output_draft,
            prefix=current_prefix,
            k=k,
            temperature=config.temperature,
            eot_token=eot,
            suppress_ids=suppress_ids,
            sample_begin=sample_begin,
            tokenizer=tokenizer,
            top_p=config.top_p,
        )
        total_drafted += len(draft.tokens)

        # --- Verify: one parallel final-model forward, no persistent cache ---
        draft_logits, bonus_logit = score_with_final(
            final_model=model_pair.final,
            encoder_output=encoder_output_final,
            prefix=current_prefix,
            draft_tokens=draft.tokens,
        )

        # Apply Whisper logit filters to the final-model scores.
        for i in range(len(draft.tokens)):
            _apply_logit_filters(
                draft_logits[i],
                len(current_prefix) + i,
                sample_begin,
                suppress_ids,
                tokenizer,
                top_p=config.top_p,
            )

        # --- Accept / reject ---
        accepted_tokens, all_accepted = accept_reject(
            draft_result=draft,
            final_logits=draft_logits,
            temperature=config.temperature,
        )

        total_accepted += sum(
            1 for i, t in enumerate(accepted_tokens)
            if i < len(draft.tokens) and t == draft.tokens[i]
        )

        generated_tokens.extend(accepted_tokens)

        if eot in accepted_tokens:
            eot_idx = generated_tokens.index(eot)
            generated_tokens = generated_tokens[:eot_idx]
            break

        # Bonus token: when all k drafts accepted, score_with_final already
        # computed logits for position k+1 — sample it at zero extra cost.
        if all_accepted and len(draft.tokens) == k:
            _apply_logit_filters(
                bonus_logit,
                len(current_prefix) + k,
                sample_begin,
                suppress_ids,
                tokenizer,
                top_p=config.top_p,
            )
            bonus_token = _sample_token(bonus_logit, config.temperature)
            if bonus_token == eot:
                break
            generated_tokens.append(bonus_token)

    text = tokenizer.decode(generated_tokens).strip()

    logger.info(
        "Speculative decode: %d tokens, drafted=%d, accepted=%d, rate=%.2f",
        len(generated_tokens),
        total_drafted,
        total_accepted,
        total_accepted / max(total_drafted, 1),
    )

    return DecodingOutput(
        text=text,
        tokens=generated_tokens,
        num_drafted=total_drafted,
        num_accepted=total_accepted,
    )


# =====================================================================
# Baseline greedy decode
# =====================================================================


@torch.inference_mode()
def baseline_decode(
    model_pair: ModelPair,
    mel: Tensor,
    config: DecodingConfig,
) -> DecodingOutput:
    """Standard greedy decoding with Whisper Large (baseline).

    Uses the official ``whisper.decode()`` for an apples-to-apples comparison.

    Args:
        model_pair: Loaded model pair (only the final model is used).
        mel: Log-mel spectrogram, shape ``(1, n_mels, T)`` or ``(n_mels, T)``.
        config: Decoding configuration.

    Returns:
        A ``DecodingOutput`` with text and tokens (acceptance fields are zero).
    """
    # Squeeze batch dim if present — whisper.decode expects (n_mels, T) or (batch, n_mels, T).
    if mel.dim() == 3 and mel.shape[0] == 1:
        mel_input = mel.squeeze(0)
    else:
        mel_input = mel

    mel_input = mel_input.to(device=model_pair.device, dtype=model_pair.dtype)

    options = whisper.DecodingOptions(
        language=config.language,
        temperature=config.temperature,
        sample_len=config.max_tokens,
        without_timestamps=True,
        fp16=(model_pair.dtype == torch.float16),
    )

    result = whisper.decode(model_pair.final, mel_input, options)

    # whisper.decode returns a single DecodingResult or list.
    if isinstance(result, list):
        result = result[0]

    return DecodingOutput(
        text=result.text.strip(),
        tokens=list(result.tokens),
        num_drafted=0,
        num_accepted=0,
    )
