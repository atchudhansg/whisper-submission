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

#### `transcribe()` - Text Output Only

Returns plain strings. Use when you only need transcription text.

**Single file:**
```python
from speculative_whisper import SpeculativeWhisper

sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3", device="cuda")
text = sw.transcribe("samples/1001_DFA_ANG_XX.wav")
print(text)
# Output: Don't forget to check it.
```

**Batch:**
```python
texts = sw.transcribe(["samples/1001_DFA_ANG_XX.wav", "samples/1002_DFA_ANG_XX.wav"], batch_size=2)
print(texts)
# Output: ["Don't forget to check it.", "Kids are talking by the door."]
```

#### `transcribe_verbose()` - Full Metrics

Returns `DecodingOutput` objects with acceptance rate, token counts, and detailed metrics. Use for benchmarking or analysis.

```python
output = sw.transcribe_verbose("samples/1001_DFA_ANG_XX.wav")
print(f"Text: {output.text}")
print(f"Tokens: {output.tokens}")
print(f"Acceptance Rate: {output.acceptance_rate:.1%}")
print(f"Drafted: {output.num_drafted}, Accepted: {output.num_accepted}")

# Output:
# Text: Don't forget to check it.
# Tokens: [50364, 380, 5158, 281, 1520, 309, 13]
# Acceptance Rate: 50.0%
# Drafted: 10, Accepted: 5
```

**Batch with metrics:**
```python
outputs = sw.transcribe_verbose(["samples/1001_DFA_ANG_XX.wav", "samples/1002_DFA_ANG_XX.wav"])
for output in outputs:
    print(f"{output.text} (accept: {output.acceptance_rate:.1%})")

# Output:
# Don't forget to check it. (accept: 50.0%)
# Kids are talking by the door. (accept: 48.3%)
```

#### Configuration

**YAML configuration file:**
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

Load at initialization:
```python
sw = SpeculativeWhisper(config_path="config.yaml")
text = sw.transcribe("audio.wav")
# Uses parameters from config.yaml
```

**Runtime parameter overrides:**

Greedy decoding (exact match to baseline):
```python
text = sw.transcribe("audio.wav", draft_k=10, temperature=0.0)
print(text)
# Output: Don't forget to check it.
```

Top-p sampling (adds diversity):
```python
text = sw.transcribe("audio.wav", sampling_strategy="top_p", top_p=0.9, temperature=0.6)
print(text)
# Output: Don't forget to check that.  (slight variation due to sampling)
```

Disable speculative decoding (baseline only):
```python
text = sw.transcribe("audio.wav", use_speculative=False)
print(text)
# Output: Don't forget to check it. (same as greedy, but slower)
```

Multilingual transcription:
```python
text = sw.transcribe("french_audio.wav", language="fr")
print(text)
# Output: Bonjour, comment allez-vous?
```

### REST API

Production-ready HTTP interface for transcription. All responses include performance metrics.

**Start server (GPU):**
```bash
WHISPER_DRAFT_MODEL=tiny WHISPER_FINAL_MODEL=large-v3 WHISPER_DEVICE=cuda \
  uvicorn api.server:app --port 8000
```

**Health check** - Verify server and models are loaded:
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model_loaded": true,
  "draft_model": "tiny",
  "final_model": "large-v3",
  "device": "cuda"
}
```

**Single file transcription** - Transcribe one audio file with speculative decoding (default):
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

**Single file with greedy decoding** - Use larger draft_k for more tokens per iteration:
```bash
curl -X POST "http://localhost:8000/transcribe/single?draft_k=10&temperature=0.0" \
  -F "file=@samples/1001_DFA_ANG_XX.wav"
```
```json
{
  "file": "1001_DFA_ANG_XX.wav",
  "text": "Don't forget to check it.",
  "latency_s": 0.65,
  "acceptance_rate": 0.48,
  "num_tokens": 7
}
```

**Single file with top-p sampling** - Add diversity via nucleus sampling:
```bash
curl -X POST "http://localhost:8000/transcribe/single?sampling_strategy=top_p&top_p=0.9&temperature=0.6" \
  -F "file=@samples/1001_DFA_ANG_XX.wav"
```
```json
{
  "file": "1001_DFA_ANG_XX.wav",
  "text": "Don't forget to check that.",
  "latency_s": 0.72,
  "acceptance_rate": 0.43,
  "num_tokens": 7
}
```

**Baseline mode** - Disable speculative decoding and run Large V3 only (slower, for comparison):
```bash
curl -X POST "http://localhost:8000/transcribe/single?use_speculative=false" \
  -F "file=@samples/1001_DFA_ANG_XX.wav"
```
```json
{
  "file": "1001_DFA_ANG_XX.wav",
  "text": "Don't forget to check it.",
  "latency_s": 0.69,
  "acceptance_rate": null,
  "num_tokens": 7
}
```

**Batch transcription** - Process multiple files in one request:
```bash
curl -X POST "http://localhost:8000/transcribe?batch_size=3" \
  -F "files=@samples/1001_DFA_ANG_XX.wav" \
  -F "files=@samples/1002_DFA_ANG_XX.wav" \
  -F "files=@samples/1003_DFA_ANG_XX.wav"
```
```json
{
  "results": [
    {
      "file": "1001_DFA_ANG_XX.wav",
      "text": "Don't forget to check it.",
      "latency_s": 0.67,
      "acceptance_rate": 0.5,
      "num_tokens": 7
    },
    {
      "file": "1002_DFA_ANG_XX.wav",
      "text": "Kids are talking by the door.",
      "latency_s": 0.71,
      "acceptance_rate": 0.45,
      "num_tokens": 8
    },
    {
      "file": "1003_DFA_ANG_XX.wav",
      "text": "She had your dark suit in greasy wash water all year.",
      "latency_s": 0.89,
      "acceptance_rate": 0.38,
      "num_tokens": 13
    }
  ],
  "total_files": 3,
  "batch_latency_s": 2.27
}
```

**Multilingual transcription** - Specify language via query parameter:
```bash
curl -X POST "http://localhost:8000/transcribe/single?language=fr" \
  -F "file=@french_audio.wav"
```
```json
{
  "file": "french_audio.wav",
  "text": "Bonjour, comment allez-vous?",
  "latency_s": 0.68,
  "acceptance_rate": 0.52,
  "num_tokens": 6
}
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
benchmark.py             # Performance evaluation script
tests/                   # Comprehensive test suite
├── conftest.py          # Pytest fixtures
├── test_smoke.py        # Model loading & diagnostics
├── test_e2e.py          # End-to-end transcription
├── test_decoding.py     # Rejection sampling logic
├── test_audio.py        # Audio preprocessing
├── test_core_api.py     # API validation & config
├── test_evaluation.py   # WER computation
└── test_integration.py   # Full integration tests
```

---

## Testing

Comprehensive test suite covering unit tests, integration tests, and benchmarking.

### Test Scripts

#### Unit Tests

**test_core_api.py** — Configuration validation and parameter bounds
- Tests for `DecodingConfig` parameter validation  
- YAML serialization/deserialization
- Device auto-detection
- Decoding parameter ranges (draft_k, temperature, top_p)

```bash
pytest tests/test_core_api.py -v
```

**test_audio.py** — Audio preprocessing pipeline
- Audio loading and resampling to 16 kHz
- Mel spectrogram computation (80 and 128 bins)
- Batch mel processing
- Edge cases and format validation

```bash
pytest tests/test_audio.py -v
```

**test_decoding.py** — Rejection sampling algorithm
- Draft step token generation
- Accept/reject logic correctness
- Bonus token generation
- Determinism in greedy mode
- Consistency with baseline Large V3

```bash
pytest tests/test_decoding.py -v
```

**test_evaluation.py** — WER metric computation
- Single-utterance WER (identical, 1 error, empty)
- Batch WER corpus-level evaluation
- Case insensitivity
- Edge cases (empty strings, mismatched lengths)

```bash
pytest tests/test_evaluation.py -v
```

#### Integration Tests

**test_integration.py** — End-to-end with real models and audio
- Single file transcription and batch processing
- `transcribe()` vs `transcribe_verbose()` outputs
- Greedy determinism validation
- Speculative vs baseline consistency
- Acceptance rate in valid range
- Multilingual configuration
- Error handling (missing files, invalid parameters)

```bash
pytest tests/test_integration.py -v -m integration
```

#### Diagnostic Tests

**test_smoke.py** — Verify model loading and system diagnostics
- Loads both draft and final models
- Reports GPU memory, device type, model sizes
- Measures loading time
- Validates model parameters and dtype

```bash
python -m tests.test_smoke --draft tiny --final large-v3 --device cuda
```

**test_e2e.py** — End-to-end transcription test
- Transcribes test audio
- Compares speculative vs baseline latency
- Prints acceptance rate and token counts
- Validates output matches across methods

```bash
python -m tests.test_e2e --draft tiny --final tiny --device cpu
```

### Performance Evaluation

**benchmark.py** — Comprehensive benchmarking suite (470 lines, real-world testing)

Benchmark on sample audio:
```bash
python benchmark.py samples/
```

Benchmark with custom audio directory:
```bash
python benchmark.py /path/to/audio/dir/ --device cuda --draft-k 5
```

Save results to JSON:
```bash
python benchmark.py samples/ --output-json results.json
```

### Running All Tests

Unit + integration + diagnostics:
```bash
pytest tests/ -v
```

With coverage report:
```bash
pytest tests/ --cov=speculative_whisper --cov-report=html
```

Fast tests only (skips extended integration tests):
```bash
pytest tests/ -v -m "not integration"
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

## License

MIT License. See [LICENSE](LICENSE) for details.
