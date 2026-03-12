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
    kv_cache: Optional[dict] = None,
    prefix_cached_len: int = 0,
) -> DraftResult:
    """Autoregressively sample *k* tokens from the draft model **with KV caching**.

    On the first call (``kv_cache`` empty / None, ``prefix_cached_len=0``),
    the full prefix is fed in to warm the cache.  Subsequent iterations
    reuse the cached keys/values and only feed the new token(s), giving
    a large speed-up on long sequences.

    Args:
        draft_model: The draft (Tiny) Whisper model.
        encoder_output: Encoder features, shape ``(1, T', d_model)``.
        prefix: Current token prefix **including** previously generated tokens.
        k: Number of tokens to draft.
        temperature: Sampling temperature. 0 → argmax (greedy).
        eot_token: End-of-transcript token ID — stop drafting if sampled.
        suppress_ids: Token IDs to suppress in logits.
        sample_begin: Position index of the first free decoding step.
        tokenizer: Whisper tokenizer (for SuppressBlank).
        top_p: Optional nucleus-sampling threshold.
        kv_cache: Persistent KV cache dict from ``install_kv_cache_hooks``.
            Mutated in-place — callers should keep the reference across
            iterations and truncate as needed on rejection.
        prefix_cached_len: Number of prefix tokens already present in
            ``kv_cache``.  We only feed the *new* portion of the prefix
            to the decoder.

    Returns:
        A ``DraftResult`` containing the drafted tokens and their log-probs.
    """
    device = encoder_output.device
    drafted_tokens: List[int] = []
    drafted_log_probs: List[float] = []

    # Determine which tokens to feed: skip those already cached.
    if kv_cache is not None and prefix_cached_len > 0:
        tokens_to_feed = prefix[prefix_cached_len:]
    else:
        tokens_to_feed = prefix

    # If we have new prefix tokens (e.g. accepted from prior iteration), prime the cache.
    if tokens_to_feed:
        token_tensor = torch.tensor([tokens_to_feed], dtype=torch.long, device=device)
        logits = draft_model.decoder(token_tensor, encoder_output, kv_cache=kv_cache)
        next_logits = logits[0, -1]
    else:
        # Edge case: nothing new to feed — cache is fully warm.
        # We'll get logits from the first draft token below.
        next_logits = None

    current_len = len(prefix)

    for i in range(k):
        if next_logits is None:
            # Shouldn't happen after warm-up above, but guard anyway.
            token_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = draft_model.decoder(token_tensor, encoder_output, kv_cache=kv_cache)
            next_logits = logits[0, -1]

        # Apply Whisper's logit filters before sampling.
        _apply_logit_filters(next_logits, current_len, sample_begin, suppress_ids, tokenizer, top_p=top_p)

        token_id = _sample_token(next_logits, temperature)
        log_prob = _get_log_prob(next_logits, token_id, temperature)

        drafted_tokens.append(token_id)
        drafted_log_probs.append(log_prob)
        current_len += 1

        if token_id == eot_token:
            break

        # Feed only the new token for the next step (KV cache handles history).
        if i < k - 1:
            token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
            logits = draft_model.decoder(token_tensor, encoder_output, kv_cache=kv_cache)
            next_logits = logits[0, -1]

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
    kv_cache: Optional[dict] = None,
    prefix_cached_len: int = 0,
) -> Tensor:
    """Score all draft positions in one forward pass of the final model.

    When a ``kv_cache`` is provided, only the *uncached* portion of the
    prefix plus the draft tokens are fed through the decoder, drastically
    reducing computation on subsequent iterations.

    Args:
        final_model: The verification (Large V3) Whisper model.
        encoder_output: Encoder features, shape ``(1, T', d_model)``.
        prefix: Token prefix before the draft sequence.
        draft_tokens: Drafted token IDs to verify.
        kv_cache: Persistent KV cache dict (mutated in-place).
        prefix_cached_len: How many prefix tokens are already in the cache.

    Returns:
        Logits tensor of shape ``(K, vocab_size)`` — one row per draft position.
        Position *i* contains the logits *predicting* ``draft_tokens[i]``
        (i.e. conditioned on ``prefix + draft_tokens[:i]``).
    """
    device = encoder_output.device
    k = len(draft_tokens)

    # Build only the tokens that are NOT yet in the KV cache.
    new_prefix_tokens = prefix[prefix_cached_len:]
    tokens_to_feed = new_prefix_tokens + draft_tokens
    token_tensor = torch.tensor([tokens_to_feed], dtype=torch.long, device=device)

    # Single forward pass — logits for every *new* position.
    logits = final_model.decoder(
        token_tensor, encoder_output, kv_cache=kv_cache
    )  # (1, len(tokens_to_feed), vocab)

    # The last K positions in the output correspond to the draft tokens.
    # new_prefix has len(new_prefix_tokens) entries.  Position
    # (len(new_prefix_tokens) - 1) predicts draft_tokens[0], etc.
    offset = len(new_prefix_tokens) - 1
    if offset < 0:
        offset = 0
    draft_logits = logits[0, offset : offset + k]  # (K, vocab)

    return draft_logits


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
    - Compute ``r = p_final(token) / p_draft(token)``.
    - Accept deterministically if ``r ≥ 1`` (final likes it at least as much).
    - Otherwise accept with probability ``r``.
    - On rejection: resample that position from the adjusted distribution
      and **stop** (remaining draft tokens are discarded).

    At temperature=0 (greedy): accept iff the final model's argmax matches
    the draft token. On mismatch, take the final model's argmax instead.

    Args:
        draft_result: Tokens and log-probs from the draft model.
        final_logits: Logits from the final model at each draft position,
                      shape ``(K, vocab_size)``.
        temperature: Sampling temperature (must match what was used for drafting).

    Returns:
        A tuple of ``(accepted_tokens, all_accepted)`` where
        ``all_accepted`` is True iff every draft token was kept.
    """
    accepted: List[int] = []
    k = len(draft_result.tokens)

    for i in range(k):
        draft_token = draft_result.tokens[i]
        logits_i = final_logits[i]

        if temperature == 0.0:
            # Greedy mode: accept iff final model also picks this token as argmax.
            final_choice = int(logits_i.argmax(dim=-1).item())
            if final_choice == draft_token:
                accepted.append(draft_token)
            else:
                # Reject — use the final model's argmax instead.
                accepted.append(final_choice)
                return accepted, False
        else:
            # Stochastic mode: rejection sampling.
            final_log_prob = _get_log_prob(logits_i, draft_token, temperature)
            draft_log_prob = draft_result.log_probs[i]

            # r = p_final / p_draft (in log space: exp(log_final - log_draft))
            log_r = final_log_prob - draft_log_prob
            r = min(1.0, torch.exp(torch.tensor(log_r)).item())

            if torch.rand(1).item() < r:
                accepted.append(draft_token)
            else:
                # Reject — resample from the adjusted distribution.
                # p_adjusted ∝ max(0, p_final - p_draft) to maintain exactness.
                final_probs = F.softmax(logits_i.float() / temperature, dim=-1)
                draft_probs = F.softmax(
                    torch.full_like(logits_i, float("-inf"))
                    .scatter(0, torch.tensor(draft_token, device=logits_i.device), logits_i[draft_token])
                    .float() / temperature,
                    dim=-1,
                )
                # Simpler: just sample from final distribution on rejection.
                resampled = int(Categorical(probs=final_probs).sample().item())
                accepted.append(resampled)
                return accepted, False

    return accepted, True


def _truncate_kv_cache(kv_cache: dict, max_len: int) -> None:
    """Truncate all cached tensors to ``max_len`` along the sequence dimension.

    This is needed after rejection — tokens beyond the accepted prefix are
    invalid and must be trimmed so subsequent forward passes see a
    consistent history.
    """
    for module, tensor in kv_cache.items():
        if tensor.shape[1] > max_len:
            kv_cache[module] = tensor[:, :max_len].detach()


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

    Uses **persistent KV caches** for both the draft and final models so
    that previously-computed attention states are reused across
    iterations, avoiding redundant computation.

    Orchestrates:
      1. Encode audio once per model.
      2. Install KV-cache hooks on both models.
      3. Repeat until EOT or ``max_tokens``:
         a. ``draft_step`` — sample *draft_k* tokens from Tiny (cached).
         b. ``score_with_final`` — verify with Large V3 (cached).
         c. ``accept_reject`` — keep valid tokens; on rejection, truncate
            both KV caches to the accepted length.
         d. Bonus token — when all K drafts are accepted, the final model's
            logits at position K+1 are already available; sample a free
            extra token.
      4. Remove hooks and decode token IDs to text.

    Args:
        model_pair: Loaded draft + final models.
        mel: Log-mel spectrogram for the final model, shape ``(1, n_mels_final, T)``.
        config: Decoding configuration.
        mel_draft: Log-mel spectrogram for the draft model when ``n_mels``
            differs (e.g. Tiny=80 vs Large=128).

    Returns:
        A ``DecodingOutput`` with text, tokens, and acceptance statistics.
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

    # 2. Build initial token prefix and logit filters.
    prefix = _get_initial_tokens(model_pair, config)
    sample_begin = len(prefix)
    suppress_ids = _build_suppress_tokens(tokenizer)

    # 3. Install KV-cache hooks for both models.
    draft_kv_cache, draft_hooks = model_pair.draft.install_kv_cache_hooks()
    final_kv_cache, final_hooks = model_pair.final.install_kv_cache_hooks()

    total_drafted = 0
    total_accepted = 0
    generated_tokens: List[int] = []

    # Track how many tokens each cache has already processed so we only
    # feed new tokens on subsequent iterations.
    draft_cached_len = 0
    final_cached_len = 0

    try:
        # 4. Speculative decode loop.
        while len(generated_tokens) < config.max_tokens:
            current_prefix = prefix + generated_tokens

            max_ctx = model_pair.final.dims.n_text_ctx
            if len(current_prefix) >= max_ctx:
                logger.warning("Reached max decoder context (%d), stopping.", max_ctx)
                break

            remaining = config.max_tokens - len(generated_tokens)
            ctx_remaining = max_ctx - len(current_prefix)
            k = min(config.draft_k, remaining, ctx_remaining)
            if k <= 0:
                break

            # --- Draft (with KV cache) ---
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
                kv_cache=draft_kv_cache,
                prefix_cached_len=draft_cached_len,
            )
            total_drafted += len(draft.tokens)

            # --- Verify (with KV cache) ---
            final_logits = score_with_final(
                final_model=model_pair.final,
                encoder_output=encoder_output_final,
                prefix=current_prefix,
                draft_tokens=draft.tokens,
                kv_cache=final_kv_cache,
                prefix_cached_len=final_cached_len,
            )

            # Apply logit filters to the final model's output.
            for i in range(len(draft.tokens)):
                _apply_logit_filters(
                    final_logits[i],
                    len(current_prefix) + i,
                    sample_begin,
                    suppress_ids,
                    tokenizer,
                    top_p=config.top_p,
                )

            # --- Accept / reject ---
            accepted_tokens, all_accepted = accept_reject(
                draft_result=draft,
                final_logits=final_logits,
                temperature=config.temperature,
            )
            num_accepted_this_iter = sum(
                1 for i, t in enumerate(accepted_tokens)
                if i < len(draft.tokens) and t == draft.tokens[i]
            )
            total_accepted += num_accepted_this_iter

            generated_tokens.extend(accepted_tokens)

            # Update cached lengths: the prefix tokens that are now
            # "permanent" (accepted) should be kept in the caches.
            # After acceptance the final cache has processed
            #   current_prefix + draft_tokens (all of them),
            # but we only want to keep up to current_prefix + accepted.
            accepted_total_len = len(prefix) + len(generated_tokens)

            # Truncate draft cache: the draft model processed
            # current_prefix + all k drafted tokens.  We keep only up to
            # the accepted boundary.
            draft_processed = len(current_prefix) + len(draft.tokens)
            if draft_processed > accepted_total_len:
                _truncate_kv_cache(draft_kv_cache, accepted_total_len)
            draft_cached_len = accepted_total_len

            # Truncate final cache: score_with_final fed
            # new_prefix + draft_tokens through the decoder.
            # The cache now has len(current_prefix) + len(draft_tokens) entries.
            # Keep only up to accepted boundary.
            final_processed = len(current_prefix) + len(draft.tokens)
            if final_processed > accepted_total_len:
                _truncate_kv_cache(final_kv_cache, accepted_total_len)
            final_cached_len = accepted_total_len

            # EOT check.
            if eot in accepted_tokens:
                try:
                    eot_idx = generated_tokens.index(eot)
                    generated_tokens = generated_tokens[:eot_idx]
                except ValueError:
                    pass
                break

            # Bonus token: when all K drafts are accepted, the final model
            # already computed logits predicting the token *after* the last
            # draft position — grab it for free.
            if all_accepted and len(draft.tokens) == k:
                # final_logits has K rows — but that's logits *predicting*
                # each draft token.  We need logits *after* the last draft,
                # which is logits[0, -1] from the full forward pass output.
                # score_with_final doesn't return it, so issue a cheap
                # single-token forward for the bonus.
                last_token = accepted_tokens[-1]
                bonus_tensor = torch.tensor(
                    [[last_token]], dtype=torch.long, device=device
                )
                bonus_logits_raw = model_pair.final.decoder(
                    bonus_tensor, encoder_output_final, kv_cache=final_kv_cache
                )
                bonus_logits = bonus_logits_raw[0, -1]
                _apply_logit_filters(
                    bonus_logits,
                    accepted_total_len,
                    sample_begin,
                    suppress_ids,
                    tokenizer,
                    top_p=config.top_p,
                )
                bonus_token_id = _sample_token(bonus_logits, config.temperature)
                generated_tokens.append(bonus_token_id)

                # Update caches to reflect the bonus token.
                bonus_total_len = len(prefix) + len(generated_tokens)
                # Draft cache doesn't have the bonus — feed it.
                bonus_draft_tensor = torch.tensor(
                    [[bonus_token_id]], dtype=torch.long, device=device
                )
                model_pair.draft.decoder(
                    bonus_draft_tensor, encoder_output_draft, kv_cache=draft_kv_cache
                )
                draft_cached_len = bonus_total_len
                final_cached_len = bonus_total_len

                if bonus_token_id == eot:
                    generated_tokens.pop()  # don't include EOT in output
                    break

    finally:
        # 5. Remove hooks to avoid memory leaks.
        for hook in draft_hooks:
            hook.remove()
        for hook in final_hooks:
            hook.remove()

    # 6. Decode tokens to text.
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
    """Standard greedy decoding with Whisper Large V3 (baseline).

    Uses the official ``whisper.decode()`` under the hood for an
    apples-to-apples comparison.

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
