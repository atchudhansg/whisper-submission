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

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_speculative` | bool | `true` | Use speculative decoding; `false` runs baseline Large only |
| `draft_k` | int | `5` | Draft tokens per iteration |
| `temperature` | float | `0.0` | Sampling temperature; 0 = greedy |
| `top_p` | float | — | Nucleus sampling probability mass |
| `sampling_strategy` | string | `greedy` | `greedy` or `top_p` |
| `max_tokens` | int | `200` | Maximum tokens to generate |
| `batch_size` | int | `1` | Files per batch (`/transcribe` endpoint only) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_DRAFT_MODEL` | `tiny` | Draft model name |
| `WHISPER_FINAL_MODEL` | `large-v3` | Final model name |
| `WHISPER_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |

---

## Python API

```python
from speculative_whisper import SpeculativeWhisper

sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3", device="cuda")

# Single file
text = sw.transcribe("samples/1001_DFA_ANG_XX.wav")

# Batch
texts = sw.transcribe(
    ["samples/1001_DFA_ANG_XX.wav", "samples/1002_DFA_ANG_XX.wav"],
    batch_size=2,
)

# Runtime overrides — all config params tunable per call
text = sw.transcribe("audio.wav", draft_k=10, temperature=0.0)
text = sw.transcribe("audio.wav", sampling_strategy="top_p", top_p=0.9, temperature=0.6)
text = sw.transcribe("audio.wav", use_speculative=False)
```

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

All parameters can be set at initialization or overridden per call.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `draft_model` | `tiny` | Whisper model for drafting |
| `final_model` | `large-v3` | Whisper model for verification |
| `device` | `auto` | `cuda`, `cpu`, or `auto` |
| `draft_k` | `5` | Tokens to draft per iteration |
| `temperature` | `0.0` | Sampling temperature |
| `top_p` | `None` | Nucleus sampling probability mass |
| `sampling_strategy` | `greedy` | `greedy` or `top_p` |
| `max_tokens` | `200` | Maximum tokens to generate |
| `language` | `en` | Target language |

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

- **Audio truncation:** Whisper's `pad_or_trim()` caps all input at 30 seconds. Segment longer audio externally before passing to the API.
- **Sequential batching:** `transcribe()` decodes files one at a time. True stacked-batch decoding is not yet implemented.
- **Short-clip GPU bottleneck:** On fast GPUs, large models decode short clips quickly enough that the speculative overhead is not fully amortized.

---

## Project Structure

```
speculative_whisper/
├── config.py        # Pydantic settings and YAML config loader
├── models.py        # Model loading and device management
├── audio.py         # Audio preprocessing
├── decoding.py      # Speculative decoding loop
├── core.py          # SpeculativeWhisper public class
└── cli.py           # CLI entry point

api/
└── server.py        # FastAPI REST server

samples/             # 30 CREMA audio clips (one per speaker, IDs 1001–1030)
examples/
└── api_client_example.py
```

---

## License

MIT
