.PHONY: setup fixture test serve scheduler process clean-data

PYTHON := .venv/bin/python
PYTHON_BOOTSTRAP ?= python3.12
PORT ?= 8077

setup:
	$(PYTHON_BOOTSTRAP) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

fixture:
	$(PYTHON) scripts/generate_fixture.py

test:
	$(PYTHON) -m pytest -q

serve:
	$(PYTHON) -m uvicorn backend.app.main:app --reload --port $(PORT)

scheduler:
	$(PYTHON) -m backend.app.workers.scheduler

process:
	$(PYTHON) -m backend.app.workers.processor --file data/videos/sample_hardpoint.mp4 --hud-profile CDL_2026_1080p

clean-data:
	rm -f cod_spectrum.db
	rm -rf data/frames/* data/crops/* data/clips/* data/reports/*
