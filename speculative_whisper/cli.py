"""CLI entry point for speculative-whisper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="speculative-whisper",
        description="Transcribe audio with speculative decoding (Whisper Tiny → Large V3).",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        help="One or more audio file paths to transcribe.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory of audio files to transcribe (alternative to --audio).",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="tiny",
        help="Whisper model for drafting (default: tiny).",
    )
    parser.add_argument(
        "--final-model",
        type=str,
        default="large-v3",
        help="Whisper model for verification (default: large-v3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per segment (default: 200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for multi-file transcription (default: 1).",
    )
    parser.add_argument(
        "--no-speculative",
        action="store_true",
        help="Disable speculative decoding; use standard Large V3 greedy.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON output (default: print to stdout).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — parse args and dispatch to SpeculativeWhisper."""
    parser = build_parser()
    args = parser.parse_args(argv)
    raise NotImplementedError("TODO: implement CLI dispatch logic")


if __name__ == "__main__":
    main()
