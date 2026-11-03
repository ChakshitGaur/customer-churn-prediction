# Each target is a plain Python command, so it also works without `make`
# (copy the command after the tab). The venv interpreter is found on Windows
# (.venv/Scripts) and Linux/macOS (.venv/bin); override with `make PY=...`.
PY ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)

.PHONY: help setup lint format test pipeline app clean

help:
	@echo "setup    - create venv and install pinned dependencies"
	@echo "lint     - ruff + black --check"
	@echo "format   - auto-format with ruff --fix and black"
	@echo "test     - run pytest"
	@echo "pipeline - run the full pipeline end to end (Phase 2+)"
	@echo "app      - launch the Streamlit dashboard (Phase 7)"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

lint:
	$(PY) -m ruff check app src tests
	$(PY) -m black --check app src tests

format:
	$(PY) -m ruff check --fix app src tests
	$(PY) -m black app src tests

test:
	$(PY) -m pytest

pipeline:
	$(PY) -m src.pipeline

app:
	$(PY) -m streamlit run app/app.py

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
