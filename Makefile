# Each target is a plain Python command, so it also works without `make`
# (copy the command after the tab). Windows venv paths are used below.
PY := .venv/Scripts/python.exe

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
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m black src tests

test:
	$(PY) -m pytest

pipeline:
	$(PY) -m src.pipeline

app:
	$(PY) -m streamlit run app/app.py

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
