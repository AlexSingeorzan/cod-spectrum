from __future__ import annotations

from backend.app.services import coach_view as cv
from backend.app.services import real_match


def test_momentum_and_rotation_shapes():
    a = cv.analysis()
    assert len(a["momentum"]) == len(real_match.VERIFIED)
    assert len(a["rotation"]) == 10
    assert all(0.0 <= m["p"] <= 1.0 for m in a["momentum"])


def test_turning_points_ranked_by_winprob_delta():
    tps = cv.analysis()["turning_points"]
    deltas = [abs(t["dwin"]) for t in tps]
    assert deltas == sorted(deltas, reverse=True)
    # the two LAT hill-lockouts are the biggest movers
    assert {tps[0]["hill"], tps[1]["hill"]} == {3, 7}


def test_early_van_lead_barely_moves_winprob():
    # the coach insight: VAN taking the lead early (P1/P2) is a small win% mover
    tps = {t["hill"]: t for t in cv.analysis()["turning_points"]}
    assert abs(tps[1]["dwin"]) < 4 and abs(tps[2]["dwin"]) < 4
    # while P7 dwarfs them
    assert tps[7]["dwin"] > 12


def test_van_counter_is_the_main_negative_swing():
    negs = [t for t in cv.analysis()["turning_points"] if t["dwin"] < 0]
    assert negs[0]["hill"] == 5 and negs[0]["winner"] == "VAN"


def test_events_carry_breaks_retakes_and_spawns():
    types = {e["type"] for e in cv.analysis()["events"]}
    assert {"break", "retake", "spawn"} <= types
