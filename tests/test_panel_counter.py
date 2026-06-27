from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.app.events import EventKind, SourceKind
from backend.app.services import panel_counter as pc
from backend.app.services.panel_counter import (
    PanelKillCounter,
    read_panels,
    reconcile_with_killfeed,
)
from backend.app.events import Evidence

ROOT = Path(__file__).resolve().parents[1]


def _ev(t=1.0):
    return Evidence(video_timestamp_seconds=t, crop_path="data/killfeed/x.png")


def _rd(a, b):
    return {"a": a, "b": b}


# --- counting logic --------------------------------------------------------------

def test_first_reading_seeds_history_without_emitting():
    c = PanelKillCounter()
    out = c.update(0.0, _rd([(7, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    assert out == []                       # we never invent the kills that happened before we watched


def test_increment_emits_kill_event():
    c = PanelKillCounter(team_a="LAT")
    c.update(0.0, _rd([(7, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    out = c.update(1.0, _rd([(8, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    kills = [e for e in out if e.event_type == "kill"]
    assert len(kills) == 1
    assert kills[0].payload.attacker_side == "a" and kills[0].payload.attacker_team == "LAT"


def test_simultaneous_kill_and_death_are_paired():
    c = PanelKillCounter(team_a="LAT", team_b="VAN", names={"a1": "SCRAP", "b3": "LUNARZ"})
    c.update(0.0, _rd([(7, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    out = c.update(1.0, _rd([(8, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 8), (8, 7)]), _ev())
    kill = next(e for e in out if e.event_type == "kill")
    death = next(e for e in out if e.event_type == "death")
    assert kill.payload.attacker == "SCRAP" and kill.payload.victim == "LUNARZ"
    assert kill.payload.attributes["paired"] is True
    assert death.payload.player == "LUNARZ" and death.payload.killer == "SCRAP"


def test_monotonic_guard_ignores_a_decrease():
    c = PanelKillCounter()
    c.update(0.0, _rd([(8, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    out = c.update(1.0, _rd([(7, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    assert out == []                       # kills can't decrease -> an OCR glitch is dropped


def test_large_jump_requires_a_confirming_frame():
    c = PanelKillCounter()
    c.update(0.0, _rd([(6, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    first = c.update(1.0, _rd([(20, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    assert first == []                     # implausible single-frame jump is withheld
    second = c.update(2.0, _rd([(20, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    assert len([e for e in second if e.event_type == "kill"]) == 14   # confirmed -> emitted


def test_unreadable_cell_is_skipped():
    c = PanelKillCounter()
    c.update(0.0, _rd([(7, 5), None, (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    out = c.update(1.0, _rd([(7, 5), None, (9, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    assert len([e for e in out if e.event_type == "kill"]) == 1       # a3 7->... no crash on None


def test_kill_event_satisfies_fact_invariants():
    c = PanelKillCounter(team_a="LAT")
    c.update(0.0, _rd([(7, 5), (6, 7), (8, 6), (6, 9)], [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
    kill = next(e for e in c.update(1.0, _rd([(8, 5), (6, 7), (8, 6), (6, 9)],
                                             [(4, 7), (7, 6), (8, 7), (8, 7)]), _ev())
               if e.event_type == "kill")
    assert kill.kind == EventKind.FACT and kill.evidence.has_visual() and kill.derived_from == []
    assert kill.provenance.source == SourceKind.MODEL
    assert kill.provenance.model_name == "panel_kill_counter" and kill.provenance.model_version


# --- panel reading ---------------------------------------------------------------

class _FakeReader:
    def __init__(self, values):
        self._values = iter(values)

    def read_cell(self, image):
        return next(self._values, None)


def test_read_panels_routes_cells_through_reader():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    reader = _FakeReader([(7, 5), (6, 7), (8, 6), (6, 9), (4, 7), (7, 6), (8, 7), (8, 7)])
    out = read_panels(frame, reader=reader)
    assert out["a"] == [(7, 5), (6, 7), (8, 6), (6, 9)]
    assert out["b"] == [(4, 7), (7, 6), (8, 7), (8, 7)]


def test_panel_kd_regions_match_hud_profile():
    profile = json.loads((ROOT / "data/configs/hud_profiles/CDL_2026_1080p.json").read_text())
    assert profile["regions"]["kd_panel_a"] == pc.KD_PANEL_A
    assert profile["regions"]["kd_panel_b"] == pc.KD_PANEL_B


# --- reconciliation --------------------------------------------------------------

def test_eval_against_verified_post_game_card_is_exact():
    # Regression lock: the counter reproduces the human-verified post-game card exactly
    # from the committed readings cache (offline, no OCR/VOD).
    import pytest
    from scripts.eval_panel_counter import evaluate
    dataset = ROOT / "data" / "panel_counter"
    if not (dataset / "readings.jsonl").exists():
        pytest.skip("run `make panel-counter` to build the readings cache")
    m = evaluate(dataset)["metrics"]
    assert m["mean_abs_error_kills"] == 0.0
    assert m["exact_player_matches"] == "8/8"
    assert m["team_totals"]["error"] == {"LAT": 0, "VAN": 0}
    assert m["checkpoint_505s"]["predicted"] == m["checkpoint_505s"]["verified"]


def test_reconcile_classifies_confirmed_panel_only_and_killfeed_only():
    panel = [10.0, 20.0, 30.0]            # ground-truth kills
    killfeed = [10.4, 20.2, 55.0]         # 2 match, 55 is a killfeed false positive
    rec = reconcile_with_killfeed(panel, killfeed, window_seconds=2.0)
    assert rec.confirmed == 2             # 10 & 20 matched
    assert rec.panel_only == 1            # 30 had no killfeed row (missed read)
    assert rec.killfeed_only == 1         # 55 had no kill (false positive)
    assert rec.as_dict()["killfeed_precision_estimate"] == round(2 / 3, 4)
