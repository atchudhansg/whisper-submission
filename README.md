# Speculative Decoding with Whisper

Speculative decoding applied to OpenAI Whisper. Whisper Tiny (39M params) drafts token sequences; Whisper Large V3 (1.5B params) verifies them in a single parallel forward pass. Rejection sampling guarantees the output distribution is **mathematically identical** to running Large V3 alone — exact inference, not an approximation.

**Benchmarks (P100 GPU, 30 CREMA clips):** ~50% draft acceptance rate, 1.02x speedup greedy. The `samples/` directory ships with 30 audio files so no dataset download is required.

---

## Quick Start

```bash
git clone https://github.com/atchudhansg/whisper-submission.git
cd whisper-submission
pip install -e ".[dev]"
```

---

## REST API

### Starting the server

**Local testing** (CPU, loads in seconds):
```bash
WHISPER_DRAFT_MODEL=tiny WHISPER_FINAL_MODEL=tiny uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Production** (GPU with Large V3, ~30s startup):
```bash
WHISPER_DRAFT_MODEL=tiny WHISPER_FINAL_MODEL=large-v3 WHISPER_DEVICE=cuda \
  uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Wait for `INFO: Application startup complete.` before sending requests.

### Endpoints

**1. Health check**
```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","model_loaded":true,"draft_model":"tiny","final_model":"tiny","device":"cpu"}
```

**2. Single file — speculative greedy**
```bash
curl -X POST "http://localhost:8000/transcribe/single?draft_k=5&temperature=0.0" \
  -F "file=@samples/1001_DFA_ANG_XX.wav"
```
```json
{"file":"1001_DFA_ANG_XX.wav","text":"Don't forget to check it.","latency_s":0.67,"acceptance_rate":0.5,"num_tokens":7}
```

**3. Single file — top-p sampling**
```bash
curl -X POST "http://localhost:8000/transcribe/single?sampling_strategy=top_p&top_p=0.9&temperature=0.6" \
  -F "file=@samples/1001_DFA_ANG_XX.wav"
```

**4. Single file — baseline only (no speculative)**
```bash
curl -X POST "http://localhost:8000/transcribe/single?use_speculative=false" \
  -F "file=@samples/1001_DFA_ANG_XX.wav"
```

**5. Batch transcription**
```bash
curl -X POST "http://localhost:8000/transcribe?batch_size=3&draft_k=5" \
  -F "files=@samples/1001_DFA_ANG_XX.wav" \
  -F "files=@samples/1002_DFA_ANG_XX.wav" \
  -F "files=@samples/1003_DFA_ANG_XX.wav"
```

**Note:** REST API automatically uses `transcribe_verbose()` internally to populate `acceptance_rate` and `num_tokens` fields in JSON responses. The Python API offers both `transcribe()` (text only) and `transcribe_verbose()` (detailed metrics).

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_speculative` | bool | `true` | Use speculative decoding; `false` runs baseline Large only |
| `draft_k` | int | `5` | Draft tokens per iteration |
| `temperature` | float | `0.0` | Sampling temperature; 0 = greedy |
| `top_p` | float | — | Nucleus sampling probability mass |
| `sampling_strategy` | string | `greedy` | `greedy` or `top_p` |
| `max_tokens` | int | `200` | Maximum tokens to generate |
| `language` | string | `en` | Target language (ISO 639-1 code) |
| `batch_size` | int | `1` | Files per batch (`/transcribe` endpoint only) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_DRAFT_MODEL` | `tiny` | Draft model name |
| `WHISPER_FINAL_MODEL` | `large-v3` | Final model name |
| `WHISPER_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |

---

## Python API

### Basic Usage

```python
from speculative_whisper import SpeculativeWhisper, DecodingConfig

# Initialize with model pairs
sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3", device="cuda")

# Single file transcription
text = sw.transcribe("samples/1001_DFA_ANG_XX.wav")

# Batch transcription
texts = sw.transcribe(
    ["samples/1001_DFA_ANG_XX.wav", "samples/1002_DFA_ANG_XX.wav"],
    batch_size=2,
)

# Runtime parameter overrides
text = sw.transcribe("audio.wav", draft_k=10, temperature=0.0)
text = sw.transcribe("audio.wav", sampling_strategy="top_p", top_p=0.9, temperature=0.6)
text = sw.transcribe("audio.wav", use_speculative=False)  # Baseline mode
```

### Advanced Usage

```python
# Detailed output with metrics
output = sw.transcribe_verbose("audio.wav")
print(f"Text: {output.text}")
print(f"Acceptance rate: {output.acceptance_rate:.2%}")
print(f"Tokens generated: {len(output.tokens)}")
print(f"Drafted: {output.num_drafted}, Accepted: {output.num_accepted}")

# YAML configuration
sw = SpeculativeWhisper(config_path="my_config.yaml")

# Language specification (transcribing non-English audio to that language)
text = sw.transcribe("french_audio.wav", language="fr")
text = sw.transcribe("spanish_audio.wav", language="es")
```

### Supported Languages

Whisper supports 99+ languages via ISO 639-1 codes:  
`en` (English), `es` (Spanish), `fr` (French), `de` (German), `it` (Italian), `pt` (Portuguese), `ru` (Russian), `ja` (Japanese), `ko` (Korean), `zh` (Chinese), `ar` (Arabic), `hi` (Hindi), and many more.

**Note:** Translation task (transcribing non-English audio to English) is configurable but currently defaults to transcription. The model loads with `task="transcribe"` hardcoded.

---

## Benchmark

```bash
pip install jiwer
python benchmark.py samples/
```

Runs three passes (baseline, speculative greedy, speculative top-p) on all files and prints per-sample latency, speedup, acceptance rate, WER, and aggregate statistics.

---

## How It Works

1. **Draft** — Whisper Tiny generates K candidate tokens autoregressively using a per-call KV cache.
2. **Verify** — Whisper Large V3 scores all K tokens in a single parallel forward pass, returning logits for each position plus a bonus logit at position K+1.
3. **Accept/Reject** — Rejection sampling accepts tokens where the final model agrees, resamples from the final model's distribution on the first mismatch, and appends a free bonus token when all K drafts are accepted.

The output distribution is provably identical to standard Large V3 decoding.

---

## Configuration

### YAML Configuration Files

Create custom configuration files and load them at initialization:

```yaml
# my_config.yaml
draft_model: "tiny"
final_model: "large-v3"  
device: "cuda"
draft_k: 8
temperature: 0.2
top_p: 0.95
sampling_strategy: "top_p"
max_tokens: 250
language: "es"
beam_size: null  # Beam search not yet implemented
```

```python
sw = SpeculativeWhisper(config_path="my_config.yaml")
```

### Runtime Parameter Overrides

All parameters can be set at initialization or overridden per call.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `draft_model` | `tiny` | Whisper model for drafting |
| `final_model` | `large-v3` | Whisper model for verification |
| `device` | `auto` | `cuda`, `cpu`, or `auto` device selection |
| `draft_k` | `5` | Tokens to draft per iteration |
| `temperature` | `0.0` | Sampling temperature (0.0 = greedy) |
| `top_p` | `None` | Nucleus sampling probability mass |
| `sampling_strategy` | `greedy` | `greedy` or `top_p` |
| `max_tokens` | `200` | Maximum tokens to generate |
| `language` | `en` | Target language (ISO 639-1 code) |
| `beam_size` | `None` | Beam search width (**not implemented**) |

### Missing Features

- **Beam search decoding:** Configuration supports `beam_size` parameter but beam search algorithm is not implemented
- **Translation task:** Models can be configured for translation (non-English → English) but currently hardcoded to transcription

---

## Benchmarks

**Dataset:** [Speech Emotion Recognition EN](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en) (CREMA subset, 30 clips, 4–6 words each)  
**Hardware:** Tesla P100-PCIE-16GB  
**Models:** draft = whisper-tiny (37.2M params), final = whisper-large (1541.6M params)

| Method | Mean Latency | Speedup | Acceptance Rate | Exact Match | Mean WER |
|--------|--------------|---------|-----------------|-------------|----------|
| Baseline (Large greedy) | 0.658s | 1.00x | — | 100% | 0.00% |
| Speculative Greedy | 0.650s | 1.02x | 50.1% | 100% | 0.00% |
| Speculative Top-p (temp=0.6, p=0.9) | 0.690s | 0.95x | 47.2% | 80% | 9.44% |

Greedy mode produces bit-exact outputs. Speedup is marginal on short clips with a fast GPU — the overhead of two forward passes per rejected iteration dominates when sequences are only 4–6 tokens. Speculative decoding yields larger gains on CPU inference or longer audio sequences where the final model is the bottleneck.

---

## Optimizations

- **Per-call KV cache (draft model):** Prefix warmed in one parallel pass; K single-token steps follow. Cache is discarded after each call — no cross-iteration state.
- **One-shot verification (final model):** Single forward pass over `prefix + draft_tokens`. Logits sliced at `prefix_len-1` for correct causal alignment.
- **Bonus token:** When all K drafts are accepted, the logit at position K+1 is already computed — sampled at zero additional cost.
- **CUDA:** fp16 inference, `cudnn.benchmark`, Flash Attention (`enable_flash_sdp`), separate mel spectrograms for draft (n_mels=80) and final (n_mels=128).
- **Logit filters:** `SuppressTokens` (82 non-speech tokens), `SuppressBlank` at position 0, optional top-p nucleus filter.
- **Rejection sampling:** Accepts token `t` with probability `min(1, p_final(t) / p_draft(t))`. Output distribution is identical to running the final model alone.

---

## Known Limitations

### Current Implementation
- **Audio truncation:** Whisper's `pad_or_trim()` caps all input at 30 seconds. Segment longer audio externally before passing to the API.
- **Sequential batching:** `transcribe()` processes files one at a time. True stacked-batch decoding is not yet implemented.
- **Short-clip GPU bottleneck:** On fast GPUs, large models decode short clips quickly enough that the speculative overhead is not fully amortized.
- **Language/task flexibility:** Translation task (non-English → English) is configurable but hardcoded to transcription in model loading.

### Missing Features
- **Beam search:** Configuration supports `beam_size` but beam search decoding algorithm is not implemented  
- **Advanced batching:** Current batch processing is sequential; parallel GPU batch decoding not implemented
- **Long-form audio:** No sliding window or chunking for audio longer than 30 seconds

### Performance Notes
- Speculative decoding shows larger gains on CPU inference or longer sequences where the large model becomes the bottleneck
- Current benchmarks use short 4-6 word clips where overhead can dominate gains
- GPU memory usage scales with model size (Large-v3 ≈3GB VRAM, Tiny ≈150MB VRAM)

---

## Project Structure

```
speculative_whisper/
├── config.py        # Pydantic configuration with YAML support
├── models.py        # Model loading, device management  
├── audio.py         # Audio preprocessing (load_audio, compute_mel)
├── decoding.py      # Speculative decoding algorithm + baseline
├── core.py          # SpeculativeWhisper public API (transcribe, transcribe_verbose)
└── evaluation.py    # WER computation utilities for evaluation

api/
└── server.py        # FastAPI REST server with batch support

configs/
└── default.yaml     # Default configuration template

samples/             # 30 CREMA emotion clips (1001–1030 speaker IDs)
benchmarks/          
├── benchmark.py     # Working benchmark script for local samples/
└── run_benchmark.py # Stub (not implemented)

examples/
└── api_client_example.py

tests/               # Test stubs (most marked as TODO/skip)
```

---

## License

MIT
