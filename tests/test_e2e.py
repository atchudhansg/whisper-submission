#!/usr/bin/env python3
"""End-to-end test of speculative decoding with Whisper.

Runs speculative (KV-cached, with optional bonus tokens) and baseline
decoding, then compares text output and latency.

Usage:
    python test_speculative.py
    python test_speculative.py --draft tiny --final large --device cuda --audio audio.wav
"""

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)

import numpy as np
import soundfile as sf  # noqa: E402
import torch


def generate_test_audio(path: str, duration: float = 3.0, sr: int = 16000):
    """Generate a simple test audio file (sine wave)."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr)
    return path


def main():
    parser = argparse.ArgumentParser(description="Test speculative decoding.")
    parser.add_argument("--draft", default="tiny", help="Draft model (default: tiny)")
    parser.add_argument("--final", default="tiny", help="Final model (default: tiny)")
    parser.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--audio", default=None, help="Path to audio file (optional)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens")
    parser.add_argument("--strategy", default="greedy", choices=["greedy", "top_p"])
    args = parser.parse_args()

    from speculative_whisper import SpeculativeWhisper

    # Generate test audio if none provided.
    if args.audio:
        audio_path = args.audio
    else:
        audio_path = "/tmp/test_speculative.wav"
        generate_test_audio(audio_path)
        print(f"Generated test audio: {audio_path}")

    print(f"\nLoading models: draft={args.draft}, final={args.final}, device={args.device}")
    sw = SpeculativeWhisper(
        draft_model=args.draft,
        final_model=args.final,
        device=args.device,
        sampling_strategy=args.strategy,
    )

    # Show active optimizations.
    cfg = sw.config
    print(f"\n--- Configuration ---")
    print(f"Sampling: {cfg.sampling_strategy} (temp={cfg.temperature}, top_p={cfg.top_p})")
    print(f"Draft k: {cfg.draft_k}")
    print(f"Device:  {sw.model_pair.device} (dtype={sw.model_pair.dtype})")
    if sw.model_pair.device != "cpu":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Optimizations: KV cache, flash SDPA, bonus token")

    # --- Warmup (first pass warms CUDA kernels) ---
    if sw.model_pair.device != "cpu":
        print("\nWarming up CUDA...")
        _ = sw.transcribe(audio_path, use_speculative=True, max_tokens=5)

    # --- Speculative decoding ---
    print("\n--- Speculative Decoding ---")
    t0 = time.perf_counter()
    spec_text = sw.transcribe(audio_path, use_speculative=True, max_tokens=args.max_tokens)
    spec_time = time.perf_counter() - t0
    print(f"Text:    {spec_text!r}")
    print(f"Latency: {spec_time:.3f}s")

    # --- Baseline decoding ---
    print("\n--- Baseline Decoding (whisper.decode) ---")
    t0 = time.perf_counter()
    base_text = sw.transcribe(audio_path, use_speculative=False, max_tokens=args.max_tokens)
    base_time = time.perf_counter() - t0
    print(f"Text:    {base_text!r}")
    print(f"Latency: {base_time:.3f}s")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"Speculative: {spec_time:.3f}s")
    print(f"Baseline:    {base_time:.3f}s")
    if spec_time > 0:
        speedup = base_time / spec_time
        print(f"Speedup:     {speedup:.2f}x")
    print(f"Texts match: {spec_text == base_text}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
