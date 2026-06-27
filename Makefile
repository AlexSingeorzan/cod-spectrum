.PHONY: setup fixture scorebar-ocr-dataset scorebar-ocr-eval killfeed-dataset killfeed-eval killfeed-segments killfeed-segment-eval killfeed-segment-sample killfeed-content-eval killfeed-sample killfeed-content-sample kill-type-dataset kill-type-eval kill-type-review kill-type-contact-sheet kill-type-sample weapon-dataset weapon-eval weapon-sample panel-counter panel-eval test serve scheduler process clean-data

PYTHON := .venv/bin/python
PYTHON_BOOTSTRAP ?= python3.12
PORT ?= 8077

setup:
	$(PYTHON_BOOTSTRAP) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

fixture:
	$(PYTHON) scripts/generate_fixture.py

scorebar-ocr-dataset:
	$(PYTHON) scripts/build_scorebar_ocr_dataset.py

scorebar-ocr-eval:
	$(PYTHON) scripts/eval_scorebar_ocr.py --write-json data/fixtures/scorebar_ocr/lat_van_hp/eval_results.json

killfeed-dataset:
	$(PYTHON) scripts/build_killfeed_dataset.py --vod data/videos/lat_van.mp4

killfeed-eval:
	$(PYTHON) scripts/eval_killfeed.py --dataset data/killfeed_dataset

killfeed-segments:
	$(PYTHON) scripts/build_killfeed_segments.py --dataset data/killfeed_dataset

killfeed-segment-eval:
	$(PYTHON) scripts/eval_killfeed_segments.py --dataset data/killfeed_dataset --write-json data/killfeed_dataset/segmentation_eval_results.json

killfeed-segment-sample:
	$(PYTHON) scripts/sample_killfeed_segments.py

killfeed-content-eval:
	$(PYTHON) scripts/eval_killfeed_content.py --dataset data/killfeed_dataset --write-json data/killfeed_dataset/content_eval_results.json

killfeed-sample:
	$(PYTHON) scripts/sample_killfeed.py

killfeed-content-sample:
	$(PYTHON) scripts/sample_killfeed_content.py

kill-type-dataset:
	$(PYTHON) scripts/build_kill_type_dataset.py --killfeed-dataset data/killfeed_dataset --out data/kill_type_dataset

kill-type-eval:
	$(PYTHON) scripts/eval_kill_type_recognition.py --dataset data/kill_type_dataset --write-json data/kill_type_dataset/eval_results.json

kill-type-review:
	$(PYTHON) scripts/review_kill_type_dataset.py --dataset data/kill_type_dataset summary --write-json data/kill_type_dataset/review_summary.json

kill-type-contact-sheet:
	$(PYTHON) scripts/review_kill_type_dataset.py --dataset data/kill_type_dataset contact-sheet --out data/kill_type_dataset/review_contact_sheet.png

kill-type-sample:
	$(PYTHON) scripts/sample_kill_type_recognition.py

weapon-dataset: kill-type-dataset

weapon-eval: kill-type-eval

weapon-sample: kill-type-sample

panel-counter:
	$(PYTHON) scripts/run_panel_counter.py --vod data/videos/lat_van.mp4

panel-eval:
	$(PYTHON) scripts/eval_panel_counter.py --dataset data/panel_counter --write-json data/panel_counter/eval_results.json

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
