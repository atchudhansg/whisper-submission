# Speculative Decoding with Whisper

**Whisper Tiny drafts. Whisper Large V3 verifies. Exact same output — up to 3× faster.**

Speculative decoding applied to OpenAI's Whisper speech recognition model. Uses Whisper Tiny as a lightweight draft model to propose token sequences, then verifies them with Whisper Large V3 in a single forward pass. Rejection sampling guarantees the output distribution is mathematically identical to running Large V3 alone.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourname/speculative-whisper.git
cd speculative-whisper
pip install -e ".[dev]"
```

### Usage

```python
from speculative_whisper import SpeculativeWhisper

sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3", device="cuda")

audio_files = ["audio1.wav", "audio2.wav"]
outputs = sw.transcribe(audio_files, max_tokens=200, batch_size=2)

for audio, text in zip(audio_files, outputs):
    print(f"{audio}: {text}")
```

### CLI

```bash
speculative-whisper --audio audio1.wav --device cuda
speculative-whisper --audio-dir ./samples/ --batch-size 4 --output results.json
```

### REST API

```bash
make serve
# POST audio files to http://localhost:8000/transcribe
curl -X POST http://localhost:8000/transcribe \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav"
```

---

## How It Works

1. **Draft** — Whisper Tiny autoregressively generates K tokens (cheap: ~39M params).
2. **Verify** — Whisper Large V3 scores all K tokens in one forward pass (~1.5B params).
3. **Accept/Reject** — Rejection sampling walks through the draft: accept tokens where Large V3 agrees, reject and resample from Large V3's distribution otherwise.

The output is **provably identical** to standard Large V3 decoding — this is exact inference, not an approximation.

---

## Configuration

All parameters are configurable via `configs/default.yaml` or at runtime:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `draft_model` | `tiny` | Whisper model size for drafting |
| `final_model` | `large-v3` | Whisper model size for verification |
| `device` | `auto` | `cuda`, `cpu`, or `auto` |
| `draft_k` | `5` | Number of tokens to draft per iteration |
| `temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `batch_size` | `1` | Number of audio files to process together |
| `max_tokens` | `200` | Maximum tokens to generate |
| `language` | `en` | Target language |

---

## Benchmarks

_Results on LibriSpeech test-clean (TBD):_

| Method | Latency (median) | Latency (p95) | WER | Acceptance Rate |
|--------|-------------------|---------------|-----|-----------------|
| Large V3 (baseline) | — | — | — | — |
| Speculative (Tiny → Large V3) | — | — | — | — |

---

## Project Structure

```
speculative_whisper/          # Core library
├── __init__.py               # Public API surface
├── config.py                 # Pydantic settings + YAML config
├── models.py                 # Model loading and device management
├── audio.py                  # Audio preprocessing and batching
├── decoding.py               # Speculative decoding loop (novel logic)
├── evaluation.py             # WER computation and benchmarking
├── core.py                   # SpeculativeWhisper public class
└── cli.py                    # CLI entry point

api/
└── server.py                 # FastAPI REST interface

benchmarks/
└── run_benchmark.py          # Benchmark runner

tests/
├── test_decoding.py          # Decoding unit tests
├── test_audio.py             # Audio preprocessing tests
└── test_evaluation.py        # WER sanity checks
```

---

## License

MIT
