#!/usr/bin/env python3
"""Quick smoke test: loads both Whisper models and prints diagnostics.

Usage:
    python smoke_test.py                    # auto device, tiny + large-v3
    python smoke_test.py --device cpu       # force CPU
    python smoke_test.py --draft tiny.en --final base  # custom models
"""

import argparse
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Verify Whisper model loading.")
    parser.add_argument("--draft", default="tiny", help="Draft model name (default: tiny)")
    parser.add_argument("--final", default="large-v3", help="Final model name (default: large-v3)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    from speculative_whisper.config import DecodingConfig
    from speculative_whisper.models import load_models

    config = DecodingConfig(draft_model=args.draft, final_model=args.final, device=args.device)
    device = config.device_resolved

    print(f"Device:       {device}")
    print(f"Draft model:  {args.draft}")
    print(f"Final model:  {args.final}")
    print(f"PyTorch:      {torch.__version__}")
    if device == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"VRAM:         {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    t0 = time.perf_counter()
    pair = load_models(config)
    elapsed = time.perf_counter() - t0

    draft_params = sum(p.numel() for p in pair.draft.parameters()) / 1e6
    final_params = sum(p.numel() for p in pair.final.parameters()) / 1e6

    print(f"Draft loaded:  {draft_params:.1f}M params, training={pair.draft.training}")
    print(f"Final loaded:  {final_params:.1f}M params, training={pair.final.training}")
    print(f"Dtype:         {pair.dtype}")
    print(f"Tokenizer EOT: {pair.tokenizer.eot}")
    print(f"Load time:     {elapsed:.1f}s")
    print()
    print("OK — models loaded successfully.")


if __name__ == "__main__":
    main()
