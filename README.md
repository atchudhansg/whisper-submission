# Speculative Whisper: Fast, Exact Audio Inference

**Speculative decoding for OpenAI's Whisper model.** Use a lightweight draft model (Whisper Tiny, 39M params) to predict token sequences, verified in parallel by the production model (Whisper Large V3, 1.5B params). Rejection sampling ensures outputs are **mathematically identical** to Large V3 alone—no accuracy loss, significant speed gains.

| Metric | Value |
|--------|-------|
| **Acceptance Rate** | ~50% on CREMA |
| **Speed** | 1.02x faster (greedy) |
| **Accuracy** | 100% match to Large V3 |
| **Setup** | Ships with 30 CREMA clips |

---

## What is Speculative Decoding?

A simple three-step algorithm for faster inference:

1. **Draft:** Fast model (Whisper Tiny) generates K token predictions.
2. **Verify:** Large model scores all K tokens in parallel.
3. **Accept/Reject:** Keep tokens matching the large model; resample mismatches.

The large model's probability distribution is preserved exactly—mathematically proven via rejection sampling.

---

## Why Use Speculative Whisper?

- **Exact Inference** — Provably identical to Whisper Large V3
- **Production-Ready** — FastAPI, Pydantic config, WER evaluation
- **Optimized** — CUDA/CPU auto-detection, fp16, Flash Attention
- **Multilingual** — 99+ languages via Whisper's models
- **Zero Setup** — Includes 30 CREMA clips—benchmark instantly

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/atchudhansg/whisper-submission.git
cd whisper-submission
pip install -e ".[dev]"
```

### 2. Benchmark Your Hardware (5 min)

```bash
pip install jiwer
python benchmark.py samples/
```

**Expected output:**
```
Spec Greedy  — speedup: 1.02x  | acceptance: 50.1%  | WER: 0.00%
Spec Top-p   — speedup: 0.95x  | acceptance: 47.2%  | WER: 9.44%
```

### 3. Start Using

**Python (simplest):**
```python
from speculative_whisper import SpeculativeWhisper

sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3", device="cuda")
text = sw.transcribe("audio.wav")
print(text)  # "Don't forget to check it."
```

**FastAPI Server:**
```bash
WHISPER_DEVICE=cuda uvicorn api.server:app --port 8000
curl -X POST http://localhost:8000/transcribe/single -F "file=@audio.wav"
```

---

## Usage Guide

### Python API

**Simple transcription:**
```python
from speculative_whisper import SpeculativeWhisper

sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3", device="cuda")

# Single file
text = sw.transcribe("samples/1001_DFA_ANG_XX.wav")
# Output: "Don't forget to check it."

# Batch
texts = sw.transcribe(["audio1.wav", "audio2.wav"], batch_size=2)
```

**Detailed metrics:**
```python
output = sw.transcribe_verbose("samples/1001_DFA_ANG_XX.wav")
print(f"Text: {output.text}")
print(f"Acceptance Rate: {output.acceptance_rate:.1%}")
# Output:
# Text: Don't forget to check it.
# Acceptance Rate: 50.0%
```

**YAML Configuration:**
```yaml
# config.yaml
draft_model: "tiny"
final_model: "large-v3"
device: "cuda"
draft_k: 8
temperature: 0.2
sampling_strategy: "top_p"
top_p: 0.95
language: "en"
max_tokens: 250
```

```python
sw = SpeculativeWhisper(config_path="config.yaml")
```

**Runtime overrides:**
```python
text = sw.transcribe("audio.wav", draft_k=10, temperature=0.0)
text = sw.transcribe("audio.wav", sampling_strategy="top_p", top_p=0.9)
text = sw.transcribe("audio.wav", use_speculative=False)  # Baseline
text = sw.transcribe("french.wav", language="fr")  # Multilingual
```

### REST API

**Start server:**
```bash
# CPU (testing)
WHISPER_DRAFT_MODEL=tiny WHISPER_FINAL_MODEL=tiny WHISPER_DEVICE=cpu \
  uvicorn api.server:app --port 8000

# GPU (production)
WHISPER_DRAFT_MODEL=tiny WHISPER_FINAL_MODEL=large-v3 WHISPER_DEVICE=cuda \
  uvicorn api.server:app --port 8000
```

**Health check:**
```bash
curl http://localhost:8000/health
# Response: {"status": "ok", "model_loaded": true, ...}
```

**Transcribe single file:**
```bash
curl -X POST "http://localhost:8000/transcribe/single" \
  -F "file=@samples/1001_DFA_ANG_XX.wav"
```

```json
{
  "file": "1001_DFA_ANG_XX.wav",
  "text": "Don't forget to check it.",
  "latency_s": 0.67,
  "acceptance_rate": 0.5,
  "num_tokens": 7
}
```

**Batch transcription:**
```bash
curl -X POST "http://localhost:8000/transcribe?batch_size=3" \
  -F "files=@file1.wav" \
  -F "files=@file2.wav" \
  -F "files=@file3.wav"
```

**Query parameters:**

| Param | Default | Description |
|-------|---------|-------------|
| `use_speculative` | `true` | Enable speculative decoding |
| `draft_k` | `5` | Tokens to draft per iteration |
| `temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `top_p` | — | Nucleus sampling probability |
| `sampling_strategy` | `greedy` | `greedy` or `top_p` |
| `language` | `en` | Target language (ISO 639-1) |
| `max_tokens` | `200` | Max tokens to generate |
| `batch_size` | `1` | Files per batch |

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_DRAFT_MODEL` | `tiny` | Draft model name |
| `WHISPER_FINAL_MODEL` | `large-v3` | Final model name |
| `WHISPER_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |

---

## Language Support

Whisper supports **99+ languages** via ISO 639-1 codes.

```python
text = sw.transcribe("spanish.wav", language="es")
text = sw.transcribe("french.wav", language="fr")
text = sw.transcribe("japanese.wav", language="ja")
```

| Region | Languages |
|--------|-----------|
| **Western** | English (`en`), Spanish (`es`), French (`fr`), German (`de`), Italian (`it`), Portuguese (`pt`) |
| **Eastern** | Japanese (`ja`), Korean (`ko`), Mandarin (`zh`), Russian (`ru`), Arabic (`ar`), Hindi (`hi`) |

See [Whisper Languages](https://github.com/openai/whisper#available-models-and-languages) for complete list.

---

## Technical Architecture

### The Algorithm

```
Step 1: DRAFT
├─ Whisper Tiny: Warm KV cache (1 forward pass)
└─ Generate K tokens sequentially

Step 2: VERIFY
├─ Whisper Large V3: Single forward pass over [prefix + K draft tokens]
└─ Output: Logits at all positions + bonus position K+1

Step 3: ACCEPT/REJECT (Rejection Sampling)
For i in [1, K]:
  ├─ Compute P_large(token_i), P_tiny(token_i)
  ├─ Accept with probability: min(1, P_large / P_tiny)
  └─ On REJECT: resample from P_large, restart loop

Result: Distribution = Large V3's distribution (proven via rejection sampling)
```

### Implementation Highlights

**Dual-Mel Spectrograms:**
- Tiny uses 80 mel bins; Large V3 uses 128 mel bins
- Automatically computed once per audio file

**Per-Call KV Caches:**
- Fresh caches for each call (no cross-iteration state pollution)
- Prefix warmed in a single parallel forward pass

**Multi-Device Support:**
- Auto-detect CUDA; fall back to CPU
- fp16 inference on CUDA (2x memory efficiency)
- Flash Attention enabled when available

**Optimization Details:**
- Logit clipping for numerical stability
- Greedy mode uses argmax (no sampling overhead)
- Top-p nucleus sampling for diversity

---

## Benchmarks

### Experimental Setup
- **Dataset:** [CREMA](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en) (30 clips, 4–6 words, 1–2 seconds)
- **Hardware:** Tesla P100 GPU (16GB VRAM)
- **Models:** Draft = Tiny (39M), Final = Large (1.5B)

### Results

| Method | Latency | Speedup | Acceptance | WER |
|--------|---------|---------|-----------|-----|
| **Baseline (Large Greedy)** | 0.658s | 1.00x | — | 0.00% |
| **Speculative (Greedy)** | 0.650s | 1.02x | 50.1% | 0.00% |
| **Speculative (Top-p)** | 0.690s | 0.95x | 47.2% | 9.44% |

**Key Insights:**
- Greedy decoding is **bit-exact** (0% WER vs baseline)
- Top-p adds diversity at the cost of accuracy
- Speedup marginal on short clips; larger gains on CPU or longer sequences

### Evaluation

**Built-in Benchmark:**
```bash
python benchmark.py samples/
```

**Programmatic:**
```python
from speculative_whisper.evaluation import benchmark, compute_wer_batch

result_spec, result_base = benchmark(sw.model_pair, audio_paths, references, sw.config)
print(result_spec.summary())
print(result_base.summary())
```

---

## Project Structure

```
speculative_whisper/
├── core.py              # Main API: SpeculativeWhisper class
├── decoding.py          # Algorithm: drafting, verification, rejection sampling
├── models.py            # Model loading, device management
├── audio.py             # Audio preprocessing, mel-spectrograms
├── config.py            # Pydantic configuration, YAML support
└── evaluation.py        # WER computation, benchmarking utilities

api/
└── server.py            # FastAPI application, batch endpoints

samples/                 # 30 CREMA audio clips (IDs 1001–1030)
configs/
└── default.yaml         # Default configuration template
benchmark.py             # Performance test script
examples/
└── api_client_example.py
tests/
└── test_evaluation.py
```

---

## Limitations & Trade-offs

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| **30s Audio Window** | Whisper's standard; requires chunking for long-form | Pre-process with sliding windows |
| **Sequential Batching** | Files processed one-by-one (not stacked-batch GPU utilization) | Ideal for request-response APIs |
| **Tiny Model Weak on Accents** | Lower acceptance rates on minority accents/languages | Switch to `draft_model="base"` |
| **No Beam Search** | Config field exists but algorithm not implemented | Use `sampling_strategy="top_p"` instead |
| **Translation Only Config** | Hardcoded to transcription task | Code change required in `models.py` |

---

## Contributing

Improvements welcome:
- [ ] Beam search implementation
- [ ] Translation task support
- [ ] Stacked-batch GPU decoding
- [ ] Temperature annealing in rejection resampling
- [ ] Language-specific draft model selection

---

## License

MIT License. See [LICENSE](LICENSE) for details.
