# pdefa - AGPL-3.0-or-later - see LICENSE
.PHONY: help install test lint format build clean examples benches

help:
	@echo "Targets:"
	@echo "  install   pip install -e .[dev]"
	@echo "  test      pytest -v"
	@echo "  lint      ruff check"
	@echo "  format    ruff format"
	@echo "  build     python -m build --wheel"
	@echo "  clean     remove build artifacts"

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest -v

lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

build:
	python -m build --wheel

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

examples:
	@cat examples/README.md

benches:
	@cat benchmarks/README.md