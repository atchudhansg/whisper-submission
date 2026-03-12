"""Benchmark runner — compare speculative vs. baseline Whisper decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding vs. standard Whisper Large V3.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing audio files to evaluate.",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help="Path to a text file with ground-truth transcriptions (one per line).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device (default: auto).",
    )
    parser.add_argument(
        "--draft-k",
        type=int,
        default=5,
        help="Tokens to draft per iteration (default: 5).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load models, run benchmarks, print results, and optionally save JSON."""
    raise NotImplementedError("TODO: implement benchmark main")


if __name__ == "__main__":
    main()
