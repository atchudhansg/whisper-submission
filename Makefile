.PHONY: install install-dev test lint typecheck serve benchmark clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check speculative_whisper/ api/ tests/ benchmarks/
	ruff format --check speculative_whisper/ api/ tests/ benchmarks/

typecheck:
	mypy --ignore-missing-imports speculative_whisper/

serve:
	uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

benchmark:
	python benchmarks/run_benchmark.py \
		--audio-dir data/test-clean/ \
		--reference-file data/test-clean/references.txt \
		--output-json benchmarks/results/latest.json

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist build .mypy_cache .ruff_cache .pytest_cache htmlcov
