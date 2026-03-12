#!/usr/bin/env python3
"""
Benchmark: Baseline Whisper Large vs Speculative Decoding (Greedy & Top-p)
===========================================================================
Dataset : Speech Emotion Recognition EN (Crema subset)
          Source: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en
Models  : draft = whisper-tiny,  final = whisper-large
Metrics : Latency (s), Speedup (x), Acceptance Rate (%), WER vs baseline

Usage:
    python benchmark.py [audio_dir]
    python benchmark.py samples/       # Use local samples
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Optional

import torch

# ── WER ───────────────────────────────────────────────────────────────────────
try:
    from jiwer import wer as _jiwer_wer
    HAS_JIWER = True
except ImportError:
    print("WARNING: jiwer not installed — WER skipped.  pip install jiwer")
    HAS_JIWER = False

# ── Project imports ───────────────────────────────────────────────────────────
import whisper as _whisper

from speculative_whisper.audio import compute_mel, load_audio
from speculative_whisper.config import DecodingConfig
from speculative_whisper.decoding import baseline_decode, speculative_decode
from speculative_whisper.models import load_models

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # suppress model-load chatter during benchmark
    format="%(asctime)s %(name)s %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")
logging.getLogger("speculative_whisper").setLevel(logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
DRAFT_MODEL = "tiny"
FINAL_MODEL = "large"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TOP_N_FILES = 30
DRAFT_K     = 5
MAX_TOKENS  = 200


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    filename:       str
    text:           str
    latency_s:      float
    acceptance:     float = 0.0   # 0 for baseline
    num_drafted:    int   = 0
    num_accepted:   int   = 0
    wer_vs_base:    Optional[float] = None


@dataclass
class ModeResults:
    mode:    str
    samples: List[SampleResult] = field(default_factory=list)

    # ── aggregate helpers ──────────────────────────────────────────
    def latencies(self):  return [s.latency_s for s in self.samples]
    def wers(self):       return [s.wer_vs_base for s in self.samples if s.wer_vs_base is not None]
    def acceptances(self): return [s.acceptance for s in self.samples if s.acceptance > 0]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def wer(ref: str, hyp: str) -> float:
    if not HAS_JIWER:
        return float("nan")
    ref = ref.strip().lower()
    hyp = hyp.strip().lower()
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(_jiwer_wer(ref, hyp))


def _sep(char="─", width=90):
    print(char * width)


def _collect_audio_files(dataset_path: str) -> List[Path]:
    """Recursively collect .wav files from the dataset, sorted, top N."""
    files = sorted(Path(dataset_path).rglob("*.wav"))
    if not files:
        # some releases use .WAV
        files = sorted(Path(dataset_path).rglob("*.WAV"))
    if not files:
        raise FileNotFoundError(f"No .wav files found under {dataset_path}")
    return files[:TOP_N_FILES]


def _load_mel(audio_path: Path, n_mels: int, device: str, dtype):
    audio = load_audio(str(audio_path))
    mel = compute_mel(audio, device=device, dtype=dtype, n_mels=n_mels)
    return mel.unsqueeze(0)  # (1, n_mels, T)


def _warmup(model_pair, cfg_greedy):
    """Single throwaway decode to warm CUDA kernels."""
    if DEVICE != "cuda":
        return
    import numpy as np
    dummy = np.zeros(16000 * 3, dtype=np.float32)
    mel = compute_mel(dummy, DEVICE, model_pair.dtype, model_pair.final.dims.n_mels).unsqueeze(0)
    mel_d = compute_mel(dummy, DEVICE, model_pair.dtype, model_pair.draft.dims.n_mels).unsqueeze(0)
    speculative_decode(model_pair, mel, cfg_greedy, mel_draft=mel_d)
    baseline_decode(model_pair, mel, cfg_greedy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    print("CUDA warmup done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample decode wrappers
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(model_pair, audio_path: Path, cfg: DecodingConfig) -> SampleResult:
    mel = _load_mel(audio_path, model_pair.final.dims.n_mels, model_pair.device, model_pair.dtype)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = baseline_decode(model_pair, mel, cfg)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    return SampleResult(
        filename=audio_path.name,
        text=out.text,
        latency_s=latency,
    )


def run_speculative(
    model_pair,
    audio_path: Path,
    cfg: DecodingConfig,
    baseline_text: str,
) -> SampleResult:
    mel_final = _load_mel(audio_path, model_pair.final.dims.n_mels, model_pair.device, model_pair.dtype)
    need_draft_mel = model_pair.draft.dims.n_mels != model_pair.final.dims.n_mels
    mel_draft = (
        _load_mel(audio_path, model_pair.draft.dims.n_mels, model_pair.device, model_pair.dtype)
        if need_draft_mel else None
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = speculative_decode(model_pair, mel_final, cfg, mel_draft=mel_draft)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    return SampleResult(
        filename=audio_path.name,
        text=out.text,
        latency_s=latency,
        acceptance=out.acceptance_rate,
        num_drafted=out.num_drafted,
        num_accepted=out.num_accepted,
        wer_vs_base=wer(baseline_text, out.text),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_per_sample_table(baseline: ModeResults, greedy: ModeResults, topp: ModeResults):
    _sep("═")
    print("PER-SAMPLE RESULTS")
    _sep("═")

    col_w = 32
    hdr = (
        f"{'File':<{col_w}} "
        f"{'Base(s)':>7}  {'Grdy(s)':>7}  {'TopP(s)':>7}"
        f"  {'Spdp-G':>6}  {'Spdp-P':>6}"
        f"  {'AccR-G':>6}  {'AccR-P':>6}"
        f"  {'WER-G':>6}  {'WER-P':>6}"
    )
    print(hdr)
    _sep()

    for i, bs in enumerate(baseline.samples):
        gs = greedy.samples[i]
        ps = topp.samples[i]
        spd_g = bs.latency_s / gs.latency_s if gs.latency_s > 0 else 0
        spd_p = bs.latency_s / ps.latency_s if ps.latency_s > 0 else 0
        wer_g = f"{gs.wer_vs_base:.3f}" if gs.wer_vs_base is not None and not (gs.wer_vs_base != gs.wer_vs_base) else "  n/a"
        wer_p = f"{ps.wer_vs_base:.3f}" if ps.wer_vs_base is not None and not (ps.wer_vs_base != ps.wer_vs_base) else "  n/a"
        fname = bs.filename[:col_w]
        print(
            f"{fname:<{col_w}} "
            f"{bs.latency_s:>7.3f}  {gs.latency_s:>7.3f}  {ps.latency_s:>7.3f}"
            f"  {spd_g:>6.2f}x  {spd_p:>6.2f}x"
            f"  {gs.acceptance:>6.2%}  {ps.acceptance:>6.2%}"
            f"  {wer_g:>6}  {wer_p:>6}"
        )

    _sep()


def print_transcription_comparison(baseline: ModeResults, greedy: ModeResults, topp: ModeResults):
    _sep("═")
    print("TRANSCRIPTION COMPARISON (first 15 files)")
    _sep("═")
    for i in range(min(15, len(baseline.samples))):
        bs = baseline.samples[i]
        gs = greedy.samples[i]
        ps = topp.samples[i]
        print(f"\n[{i+1:02d}] {bs.filename}")
        print(f"  Baseline : {bs.text[:120]}")
        print(f"  Greedy   : {gs.text[:120]}")
        print(f"  Top-p    : {ps.text[:120]}")
        match_g = "✓ match" if gs.text.strip() == bs.text.strip() else "✗ differ"
        match_p = "✓ match" if ps.text.strip() == bs.text.strip() else "✗ differ"
        print(f"  Greedy vs Baseline: {match_g}   |   Top-p vs Baseline: {match_p}")


def _safe_stdev(vals):
    return stdev(vals) if len(vals) > 1 else 0.0


def print_aggregate_stats(baseline: ModeResults, greedy: ModeResults, topp: ModeResults):
    _sep("═")
    print("AGGREGATE STATISTICS")
    _sep("═")

    n = len(baseline.samples)

    # Speedups
    speedups_g = [b.latency_s / g.latency_s for b, g in zip(baseline.samples, greedy.samples) if g.latency_s > 0]
    speedups_p = [b.latency_s / p.latency_s for b, p in zip(baseline.samples, topp.samples) if p.latency_s > 0]

    # Match rates
    exact_g = sum(1 for b, g in zip(baseline.samples, greedy.samples) if b.text.strip() == g.text.strip())
    exact_p = sum(1 for b, p in zip(baseline.samples, topp.samples) if b.text.strip() == p.text.strip())

    def row(label, b_val, g_val, p_val, fmt=".3f"):
        print(f"  {label:<35} Base: {b_val:{fmt}}   Greedy: {g_val:{fmt}}   Top-p: {p_val:{fmt}}")

    print(f"\n  Files benchmarked : {n}")
    print(f"  Device            : {DEVICE.upper()}")
    print(f"  Draft model       : {DRAFT_MODEL}  |  Final model: {FINAL_MODEL}")
    print(f"  Draft k           : {DRAFT_K}")
    print()

    # Latency
    bl, gl, pl = baseline.latencies(), greedy.latencies(), topp.latencies()
    print("  ── Latency (seconds) ──────────────────────────────────────────────────")
    row("Mean latency (s)",   mean(bl),   mean(gl),   mean(pl))
    row("Median latency (s)", median(bl), median(gl), median(pl))
    row("Min latency (s)",    min(bl),    min(gl),    min(pl))
    row("Max latency (s)",    max(bl),    max(gl),    max(pl))
    row("Stdev latency (s)",  _safe_stdev(bl), _safe_stdev(gl), _safe_stdev(pl))
    print()

    # Speedup
    if speedups_g:
        print("  ── Speedup vs Baseline ────────────────────────────────────────────────")
        print(f"  {'Spec Greedy — mean speedup':<35} {mean(speedups_g):.3f}x")
        print(f"  {'Spec Greedy — median speedup':<35} {median(speedups_g):.3f}x")
        print(f"  {'Spec Greedy — min / max':<35} {min(speedups_g):.3f}x / {max(speedups_g):.3f}x")
        print(f"  {'Spec Top-p  — mean speedup':<35} {mean(speedups_p):.3f}x")
        print(f"  {'Spec Top-p  — median speedup':<35} {median(speedups_p):.3f}x")
        print(f"  {'Spec Top-p  — min / max':<35} {min(speedups_p):.3f}x / {max(speedups_p):.3f}x")
        print()

    # Acceptance rate
    ag = greedy.acceptances()
    ap = topp.acceptances()
    if ag:
        print("  ── Draft Acceptance Rate ──────────────────────────────────────────────")
        print(f"  {'Greedy — mean acceptance':<35} {mean(ag):.2%}")
        print(f"  {'Greedy — min / max':<35} {min(ag):.2%} / {max(ag):.2%}")
        print(f"  {'Top-p  — mean acceptance':<35} {mean(ap):.2%}")
        print(f"  {'Top-p  — min / max':<35} {min(ap):.2%} / {max(ap):.2%}")
        print()

    # Total tokens drafted / accepted
    total_drafted_g  = sum(s.num_drafted  for s in greedy.samples)
    total_accepted_g = sum(s.num_accepted for s in greedy.samples)
    total_drafted_p  = sum(s.num_drafted  for s in topp.samples)
    total_accepted_p = sum(s.num_accepted for s in topp.samples)
    print("  ── Token Stats (across all files) ─────────────────────────────────────")
    print(f"  {'Greedy — total drafted':<35} {total_drafted_g}")
    print(f"  {'Greedy — total accepted':<35} {total_accepted_g}")
    print(f"  {'Top-p  — total drafted':<35} {total_drafted_p}")
    print(f"  {'Top-p  — total accepted':<35} {total_accepted_p}")
    print()

    # WER
    wg = greedy.wers()
    wp = topp.wers()
    if wg and not all(w != w for w in wg):  # any non-NaN
        wg_clean = [w for w in wg if w == w]
        wp_clean = [w for w in wp if w == w]
        print("  ── Word Error Rate vs Baseline ────────────────────────────────────────")
        print(f"  (Reference = Baseline Whisper Large output)")
        if wg_clean:
            print(f"  {'Greedy — mean WER':<35} {mean(wg_clean):.4f}  ({mean(wg_clean)*100:.2f}%)")
            print(f"  {'Greedy — median WER':<35} {median(wg_clean):.4f}  ({median(wg_clean)*100:.2f}%)")
        if wp_clean:
            print(f"  {'Top-p  — mean WER':<35} {mean(wp_clean):.4f}  ({mean(wp_clean)*100:.2f}%)")
            print(f"  {'Top-p  — median WER':<35} {median(wp_clean):.4f}  ({median(wp_clean)*100:.2f}%)")
        print()
    else:
        print("  WER: jiwer not available — skipped.\n")

    # Exact match
    print("  ── Exact Match vs Baseline ────────────────────────────────────────────")
    print(f"  Greedy exact match : {exact_g}/{n}  ({exact_g/n:.1%})")
    print(f"  Top-p  exact match : {exact_p}/{n}  ({exact_p/n:.1%})")
    print()

    # Total wall time
    total_base  = sum(bl)
    total_grdy  = sum(gl)
    total_topp  = sum(pl)
    print("  ── Total Wall Time (all files) ────────────────────────────────────────")
    print(f"  Baseline   : {total_base:.2f}s")
    print(f"  Greedy     : {total_grdy:.2f}s    ({total_base/total_grdy:.2f}x vs baseline)")
    print(f"  Top-p      : {total_topp:.2f}s    ({total_base/total_topp:.2f}x vs baseline)")
    _sep("═")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Parse CLI args ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding vs baseline on audio files."
    )
    parser.add_argument(
        "audio_dir",
        nargs="?",
        default="samples",
        help="Directory containing .wav files (default: samples/)",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}")
        print(f"\nUsage: python benchmark.py [audio_dir]")
        print(f"       python benchmark.py samples/")
        sys.exit(1)

    # ── 1. Collect audio files ────────────────────────────────────────────────
    print(f"Loading audio files from: {audio_dir}")
    audio_files = _collect_audio_files(str(audio_dir))
    print(f"Found {len(audio_files)} audio files.\n")
    for i, f in enumerate(audio_files[:10], 1):  # show first 10
        print(f"  {i:02d}. {f.name}")
    if len(audio_files) > 10:
        print(f"  ... and {len(audio_files) - 10} more")
    print()

    # ── 2. Load models ────────────────────────────────────────────────────────
    print(f"Loading models: draft={DRAFT_MODEL}, final={FINAL_MODEL}, device={DEVICE}")
    cfg_greedy = DecodingConfig(
        draft_model=DRAFT_MODEL,
        final_model=FINAL_MODEL,
        device=DEVICE,
        draft_k=DRAFT_K,
        max_tokens=MAX_TOKENS,
        sampling_strategy="greedy",
    )
    cfg_topp = DecodingConfig(
        draft_model=DRAFT_MODEL,
        final_model=FINAL_MODEL,
        device=DEVICE,
        draft_k=DRAFT_K,
        max_tokens=MAX_TOKENS,
        sampling_strategy="top_p",
    )
    model_pair = load_models(cfg_greedy)  # single pair for all modes

    if DEVICE == "cuda":
        gpu = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu}")
    print()

    # ── 3. CUDA warmup ────────────────────────────────────────────────────────
    print("Warming up CUDA …")
    _warmup(model_pair, cfg_greedy)

    # ── 4. Baseline pass ─────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"[1/3] BASELINE  — whisper.decode (Large, greedy)")
    print(f"{'─'*60}")
    baseline_res = ModeResults(mode="Baseline (Large greedy)")
    for i, fpath in enumerate(audio_files, 1):
        result = run_baseline(model_pair, fpath, cfg_greedy)
        baseline_res.samples.append(result)
        print(f"  [{i:02d}/{len(audio_files)}] {fpath.name:<45} {result.latency_s:.3f}s  \"{result.text[:60]}\"")

    # ── 5. Speculative greedy pass ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[2/3] SPECULATIVE GREEDY  — draft={DRAFT_MODEL} k={DRAFT_K}, final={FINAL_MODEL}")
    print(f"{'─'*60}")
    greedy_res = ModeResults(mode="Speculative Greedy")
    for i, fpath in enumerate(audio_files, 1):
        ref = baseline_res.samples[i - 1].text
        result = run_speculative(model_pair, fpath, cfg_greedy, baseline_text=ref)
        greedy_res.samples.append(result)
        speedup = baseline_res.samples[i-1].latency_s / result.latency_s if result.latency_s > 0 else 0
        print(
            f"  [{i:02d}/{len(audio_files)}] {fpath.name:<45} "
            f"{result.latency_s:.3f}s  {speedup:.2f}x  acc={result.acceptance:.0%}  "
            f"wer={result.wer_vs_base:.3f}" if HAS_JIWER and result.wer_vs_base == result.wer_vs_base
            else f"  [{i:02d}/{len(audio_files)}] {fpath.name:<45} "
                 f"{result.latency_s:.3f}s  {speedup:.2f}x  acc={result.acceptance:.0%}"
        )

    # ── 6. Speculative top-p pass ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[3/3] SPECULATIVE TOP-P  — draft={DRAFT_MODEL} k={DRAFT_K}, final={FINAL_MODEL}, top_p=0.9, temp=0.6")
    print(f"{'─'*60}")
    topp_res = ModeResults(mode="Speculative Top-p")
    for i, fpath in enumerate(audio_files, 1):
        ref = baseline_res.samples[i - 1].text
        result = run_speculative(model_pair, fpath, cfg_topp, baseline_text=ref)
        topp_res.samples.append(result)
        speedup = baseline_res.samples[i-1].latency_s / result.latency_s if result.latency_s > 0 else 0
        print(
            f"  [{i:02d}/{len(audio_files)}] {fpath.name:<45} "
            f"{result.latency_s:.3f}s  {speedup:.2f}x  acc={result.acceptance:.0%}  "
            f"wer={result.wer_vs_base:.3f}" if HAS_JIWER and result.wer_vs_base == result.wer_vs_base
            else f"  [{i:02d}/{len(audio_files)}] {fpath.name:<45} "
                 f"{result.latency_s:.3f}s  {speedup:.2f}x  acc={result.acceptance:.0%}"
        )

    # ── 7. Print full stats ───────────────────────────────────────────────────
    print()
    print_per_sample_table(baseline_res, greedy_res, topp_res)
    print_transcription_comparison(baseline_res, greedy_res, topp_res)
    print_aggregate_stats(baseline_res, greedy_res, topp_res)


if __name__ == "__main__":
    main()
