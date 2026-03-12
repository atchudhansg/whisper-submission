"""Example REST API client for Speculative Whisper.

Demonstrates how to use the /transcribe and /transcribe/single endpoints.
"""

import requests
from pathlib import Path


def transcribe_single(
    audio_path: str,
    base_url: str = "http://localhost:8000",
    draft_k: int = 5,
    temperature: float = 0.0,
) -> dict:
    """Transcribe a single audio file."""
    url = f"{base_url}/transcribe/single"
    params = {
        "draft_k": draft_k,
        "temperature": temperature,
        "use_speculative": True,
    }

    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "audio/wav")}
        response = requests.post(url, files=files, params=params)

    response.raise_for_status()
    return response.json()


def transcribe_batch(
    audio_paths: list[str],
    base_url: str = "http://localhost:8000",
    draft_k: int = 5,
    batch_size: int = 2,
) -> dict:
    """Transcribe multiple audio files in batch."""
    url = f"{base_url}/transcribe"
    params = {
        "draft_k": draft_k,
        "batch_size": batch_size,
        "use_speculative": True,
    }

    files = []
    for path in audio_paths:
        files.append(("files", (Path(path).name, open(path, "rb"), "audio/wav")))

    response = requests.post(url, files=files, params=params)

    # Close file handles
    for _, (_, fh, _) in files:
        fh.close()

    response.raise_for_status()
    return response.json()


def check_health(base_url: str = "http://localhost:8000") -> dict:
    """Check API health status."""
    response = requests.get(f"{base_url}/health")
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python api_client_example.py <audio_file.wav> [audio_file2.wav ...]")
        sys.exit(1)

    # Check health
    print("Checking API health...")
    health = check_health()
    print(f"Health: {health}")
    print()

    audio_files = sys.argv[1:]

    if len(audio_files) == 1:
        # Single file transcription
        print(f"Transcribing single file: {audio_files[0]}")
        result = transcribe_single(audio_files[0])
        print(f"Text: {result['text']}")
        print(f"Latency: {result['latency_s']:.3f}s")
    else:
        # Batch transcription
        print(f"Transcribing {len(audio_files)} files in batch...")
        result = transcribe_batch(audio_files)
        print(f"Total latency: {result['total_latency_s']:.3f}s")
        print()
        for item in result["results"]:
            print(f"{item['file']}: {item['text']}")
