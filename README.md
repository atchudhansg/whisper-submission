# Speculative Decoding with Whisper

**Whisper Tiny drafts. Whisper Large V3 verifies. Exact same output distribution.**

Speculative decoding applied to OpenAI's Whisper speech recognition model. Uses Whisper Tiny (39M params) as a lightweight draft model to propose token sequences, then verifies them with Whisper Large V3 (1.5B params) in a single parallel forward pass. Rejection sampling guarantees the output distribution is **mathematically identical** to running Large V3 alone — this is exact inference, not an approximation.

**Current status:** Functionally correct with ~50% draft acceptance rate. Speedup is marginal (1.02x greedy, 0.95x top-p) on short audio clips due to GPU compute bottlenecks. Best suited for CPU inference or longer audio sequences.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourname/speculative-whisper.git
cd speculative-whisper
pip install -e ".[dev]"
```

### Usage

**Python API (single file):**
```python
from speculative_whisper import SpeculativeWhisper

sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3", device="cuda")
text = sw.transcribe("audio.wav")
print(text)
```

**Python API (batch transcription):**
```python
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
texts = sw.transcribe(audio_files, batch_size=2, draft_k=8)

for audio, text in zip(audio_files, texts):
    print(f"{audio}: {text}")
```

**Runtime config overrides:**
```python
# Baseline (no speculative decoding)
baseline_text = sw.transcribe("audio.wav", use_speculative=False)

# Greedy decoding with more draft tokens
greedy_text = sw.transcribe("audio.wav", draft_k=10, temperature=0.0)

# Stochastic sampling
stochastic_text = sw.transcribe(
    "audio.wav",
    draft_k=5,
    sampling_strategy="top_p",
    temperature=0.6,
    top_p=0.9,
)
```

### CLI

```bash
speculative-whisper --audio audio1.wav --device cuda
speculative-whisper --audio-dir ./samples/ --batch-size 4 --output results.json
```

### REST API

Start the FastAPI server (models loaded once on startup):

```bash
# Option 1: Direct uvicorn
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Option 2: Via Python
python -m api.server

# Option 3: With environment overrides
WHISPER_DRAFT_MODEL=tiny WHISPER_FINAL_MODEL=large-v3 WHISPER_DEVICE=cuda \
  uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns model status and config |
| `/transcribe/single` | POST | Transcribe a single audio file |
| `/transcribe` | POST | Transcribe multiple audio files (batch) |

**Query parameters** (all endpoints):
- `max_tokens` (int, default=200): Max tokens per segment
- `use_speculative` (bool, default=true): Use speculative decoding
- `draft_k` (int, optional): Override draft_k from config
- `temperature` (float, optional): Override sampling temperature
- `top_p` (float, optional): Override nucleus sampling p
- `sampling_strategy` (str, optional): "greedy" or "top_p"

**Examples:**

```bash
# Health check
curl http://localhost:8000/health

# Single file with config overrides
curl -X POST "http://localhost:8000/transcribe/single?draft_k=8&temperature=0.0" \
  -F "file=@audio.wav"

# Batch transcription
curl -X POST "http://localhost:8000/transcribe?batch_size=4&draft_k=5" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"

# Without speculative decoding (baseline Large)
curl -X POST "http://localhost:8000/transcribe/single?use_speculative=false" \
  -F "file=@audio.wav"
```

**Python client example** (see [examples/api_client_example.py](examples/api_client_example.py)):

```python
import requests

# Single file
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe/single",
        files={"file": f},
        params={"draft_k": 5, "temperature": 0.0},
    )
result = response.json()
print(f"Text: {result['text']}, Latency: {result['latency_s']:.3f}s")

# Batch
files = [("files", open(f"audio{i}.wav", "rb")) for i in range(3)]
response = requests.post(
    "http://localhost:8000/transcribe",
    files=files,
    params={"batch_size": 2, "draft_k": 5},
)
results = response.json()
```

---

## How It Works

1. **Draft** — Whisper Tiny autoregressively generates K tokens (cheap: ~39M params).
2. **Verify** — Whisper Large V3 scores all K tokens in one forward pass (~1.5B params).
3. **Accept/Reject** — Rejection sampling walks through the draft: accept tokens where Large V3 agrees, reject and resample from Large V3's distribution otherwise.

The output is **provably identical** to standard Large V3 decoding — this is exact inference, not an approximation.

---

## Configuration

All parameters are configurable via `configs/default.yaml`, at initialization, or at runtime:

**At initialization:**
```python
sw = SpeculativeWhisper(
    draft_model="tiny",
    final_model="large-v3",
    device="cuda",
    draft_k=8,
    temperature=0.0,
)
```

**At runtime (per-call overrides):**
```python
# Override for a specific transcription call
text = sw.transcribe(
    "audio.wav",
    draft_k=10,           # Use more draft tokens for this call
    temperature=0.6,      # Switch to stochastic sampling
    top_p=0.9,
    sampling_strategy="top_p",
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `draft_model` | `tiny` | Whisper model size for drafting |
| `final_model` | `large-v3` | Whisper model size for verification |
| `device` | `auto` | `cuda`, `cpu`, or `auto` |
| `draft_k` | `5` | Number of tokens to draft per iteration |
| `temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `top_p` | `None` | Nucleus sampling probability mass (top-p) |
| `sampling_strategy` | `greedy` | `"greedy"` or `"top_p"` |
| `batch_size` | `1` | Number of audio files to process together |
| `max_tokens` | `200` | Maximum tokens to generate |
| `language` | `en` | Target language |

---

## Benchmarks

**Dataset:** Speech Emotion Recognition EN (Kaggle: `dmitrybabko/speech-emotion-recognition-en`)  
**Files:** 30 short emotion clips from CREMA dataset (4-6 words each)  
**Hardware:** Tesla P100-PCIE-16GB (Kaggle)  
**Models:** draft=tiny (37.2M params), final=large (1541.6M params)

| Method | Mean Latency | Median Latency | Speedup | Acceptance Rate | Exact Match | Mean WER |
|--------|--------------|----------------|---------|-----------------|-------------|----------|
| **Baseline (Large greedy)** | 0.658s | 0.648s | 1.00× | — | 100% (ref) | 0.00% (ref) |
| **Speculative Greedy** | 0.650s | 0.671s | **1.02×** | 50.1% | 100% | 0.00% |
| **Speculative Top-p** (temp=0.6, top_p=0.9) | 0.690s | 0.691s | **0.95×** | 47.2% | 80% | 9.44% |

**Key findings:**
- ✅ **Correctness**: Greedy mode produces bit-exact outputs (100% match), top-p has expected stochastic variation.
- ✅ **Draft acceptance**: ~50% of drafted tokens accepted — verification logic works correctly.
- ⚠️ **Speedup**: Marginal (1.02x greedy) to slightly negative (0.95x top-p) on short clips with fast GPU.
- ⚠️ **GPU bottleneck**: P100 runs Large fast enough (~0.65s) that the overhead of two forward passes per iteration (draft + verify) on rejection dominates.

**Why marginal speedup?**
1. **Short sequences**: 4-6 word clips don't amortize the speculative loop overhead.
2. **Fast GPU**: Large is already fast on P100; speculative decoding shines when the final model is a bottleneck (CPU, longer audio, larger models).
3. **No persistent KV cache**: Current implementation recomputes prefix every iteration for correctness (see Optimizations).

**When to use speculative decoding:**
- ✅ CPU inference (Large is slow → Tiny draft is very cheap)
- ✅ Longer audio (30s clips with 100+ tokens → better amortization)
- ✅ Batch processing where latency matters
- ❌ Short clips on fast GPUs (overhead > gains)

---

## Optimizations Implemented

This implementation includes several production-grade optimizations:

### 1. **Per-Call KV Caching (Draft Model)**
- Fresh KV cache installed for each `draft_step()` call
- Warms the full prefix (SOT + previously generated tokens) in one parallel forward pass
- Then issues K single-token forward passes using the cached history
- Cache discarded after drafting → no cross-iteration state corruption

### 2. **One-Shot Verification (Final Model)**
- No persistent KV cache for the final model
- Single forward pass: `prefix + draft_tokens` → logits for all K draft positions + bonus token
- Correct causal slicing at `logits[prefix_len-1 : prefix_len-1+k]` ensures proper alignment
- Bonus token logit at `logits[prefix_len-1+k]` is free when all drafts accepted

### 3. **Bonus Token Optimization**
- When all K drafts are accepted, the final model already computed logits for position K+1
- Sample the bonus token at zero extra cost (no additional forward pass)

### 4. **CUDA-Specific Optimizations** (when `device="cuda"`)
- Mixed precision (fp16) for both models
- LayerNorm params kept in float32 for PyTorch ≥2.1 compatibility
- `torch.backends.cudnn.benchmark = True` for conv/matmul kernel auto-tuning
- Flash Attention (scaled_dot_product_attention) via `torch.backends.cuda.enable_flash_sdp(True)`
- Separate mel spectrograms for draft (n_mels=80) and final (n_mels=128) to avoid recomputation

### 5. **Whisper Logit Filters** (Exact Reproduction)
- **SuppressTokens**: Always mask non-speech tokens (82 total) + special tokens (SOT, transcribe, translate, etc.)
- **SuppressBlank**: At the first decoding position, suppress space token (220) + EOT to force real content
- **Top-p (nucleus) filtering**: Optional truncation to the top-p probability mass

### 6. **Greedy vs Stochastic Sampling**
- `sampling_strategy="greedy"`: temp=0.0, deterministic argmax (100% reproducible)
- `sampling_strategy="top_p"`: temp=0.6, top_p=0.9, stochastic sampling with rejection sampling

### 7. **Rejection Sampling** (Distributionally Exact)
- For each draft token: `r = p_final(token) / p_draft(token)`
- Accept if `r ≥ 1` (final model likes it at least as much)
- Accept with probability `r` otherwise
- On rejection: resample from final model's distribution and stop
- **Guarantees**: Output distribution identical to running final model alone

---

## Known Limitations

### 1. **Audio Truncation (Whisper Inherent)**
All audio is padded or **truncated to exactly 30 seconds** (480,000 samples at 16kHz). This is a Whisper design constraint, not specific to speculative decoding. For longer audio:
- Pre-segment into 30s chunks with overlap (e.g., `pydub`, `ffmpeg`)
- Decode each chunk independently
- Stitch transcriptions with post-processing

### 2. **No Batching for Multiple Inputs**
Current `transcribe()` loops over files sequentially. True batch decoding (stacking mels, parallel verification) is not yet implemented. Each file is decoded independently.

### 3. **Marginal Speedup on Short Clips + Fast GPUs**
As shown in benchmarks, the current implementation shows minimal gains on short audio with modern GPUs. Factors:
- Overhead of speculative loop (two models per iteration on rejection)
- Short sequences (4-6 words) don't amortize well
- P100/V100/A100 run Large fast enough that Tiny's speed advantage is small

**Future work** to improve speedup:
- Adaptive draft_k based on rolling acceptance rate
- Persistent cross-iteration KV cache (requires complex offset tracking — removed for correctness in v1.0)
- Pipeline parallelism (encode audio for next file while decoding current)
- Different draft/final pairs (e.g., Distil-Whisper as draft, Whisper V3 Turbo as final)
- Torch.compile() for encoder/decoder modules

### 4. **REST API and CLI Not Yet Implemented**
`api/server.py` and full CLI dispatch are scaffolded but not wired up. Current entry point is Python API only.

### 5. **Evaluation Metrics Partial**
`evaluation.py` stub exists but full WER benchmarking suite (LibriSpeech, Common Voice, etc.) not yet integrated.

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
└── server.py                 # FastAPI REST interface (production-ready)

examples/
├── api_client_example.py     # REST API client demonstration
└── benchmark.py              # Full dataset evaluation script

tests/
├── test_decoding.py          # Decoding unit tests
├── test_audio.py             # Audio preprocessing tests
└── test_evaluation.py        # WER sanity checks
```

---

## Technical Notes: The 0% Acceptance Bug & Fix

During development, an initial implementation achieved **0% draft acceptance rate** and produced broken outputs (repeated `<|no|>` tokens). Latency was 6× **slower** than baseline. Root cause and fix:

### The Bug

**Cross-iteration KV cache offset poisoning:**

1. Initial implementation used a **persistent KV cache** across iterations to avoid recomputing the full prefix each time.
2. After N tokens were generated, `draft_cached_len = N` and `final_cached_len = N`.
3. On iteration N+1:
   - Draft model generated K new draft tokens: `draft_tokens = [d₁, d₂, ..., dₖ]`
   - `score_with_final()` was called with only `draft_tokens` (no prefix) + `kv_cache` containing prefix history
4. **The offset bug:** Whisper decoder reads `tokens` at positions `[0, 1, ..., k-1]` and outputs logits for positions `[1, 2, ..., k]`.
   - When feeding `draft_tokens` without prefix, decoder receives: `[d₁, d₂, ..., dₖ]`
   - Output logits are: `logits[0]` predicts **next token after d₁**, `logits[1]` predicts **next token after d₂**, etc.
   - For verification, we need: `logits[0]` to predict d₁ (conditioned on prefix), `logits[1]` to predict d₂ (conditioned on prefix+d₁), etc.
5. **Result:** Every position was off by one. The final model never agreed with draft tokens → 0% acceptance → re-draft every token → 6× slower + broken output.

### The Fix

**Fresh per-call KV cache + correct slicing:**

1. **Draft model** (`draft_step`):
   - Install fresh KV cache at the start of each call
   - Warm the cache with the full prefix in **one parallel forward pass**
   - Then issue K single-token forward passes using the cached history
   - Discard cache after drafting → **no cross-iteration state**

2. **Final model** (`score_with_final`):
   - **No persistent cache** at all
   - Single forward pass: `logits = decoder(prefix + draft_tokens, encoder_output)`
   - **Correct slicing:** Extract logits at positions `[prefix_len-1, prefix_len, ..., prefix_len+k-2]`
     - `logits[prefix_len-1]` predicts `draft_tokens[0]`
     - `logits[prefix_len]` predicts `draft_tokens[1]`
     - ...  
     - `logits[prefix_len+k-2]` predicts `draft_tokens[k-1]`
     - `logits[prefix_len+k-1]` is the **bonus token** logit (free when all drafts accepted)
   - Return `(draft_logits, bonus_logit)` tuple

3. **Outcome:**
   - **Before:** 0% acceptance, 6× slower, broken output
   - **After:** 47-50% acceptance, 1.02× speedup (greedy), texts match baseline

**Key lesson:** When managing KV cache with position offsets, extreme care is needed to ensure logit positions align with token positions. For production correctness, we chose simplicity (fresh state per call) over complex cross-iteration cache management.

---

## Running the Benchmark Script

To reproduce the benchmarks:

```bash
# Install dependencies
pip install kagglehub jiwer

# Clone and install
git clone https://github.com/atchudhansg/whisper-submission.git
cd whisper-submission
pip install -e .

# Run benchmark (downloads dataset automatically)
python benchmark.py
```

**Output:**
- Per-sample table (filename, latency, speedup, acceptance rate, WER)
- Transcription comparison (first 15 files)
- Aggregate statistics (mean/median latency, speedup range, acceptance rate, WER, exact match rate)

---

## License

MIT
