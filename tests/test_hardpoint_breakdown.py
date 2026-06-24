from __future__ import annotations

from backend.app.services import hardpoint_breakdown as hb


def test_scores_are_monotonic_and_end_correct():
    scores = hb.SCORE_AT_HILL_BOUNDARY
    la = [a for _, a, _ in scores]
    vn = [b for _, _, b in scores]
    assert la == sorted(la) and vn == sorted(vn)          # never decrease
    assert (la[-1], vn[-1]) == (250, 156)                  # verified final


def test_player_stats_are_internally_consistent():
    lat_k = sum(k for k, _ in hb.PLAYER_MAP_STATS["LAT"].values())
    lat_d = sum(d for _, d in hb.PLAYER_MAP_STATS["LAT"].values())
    van_k = sum(k for k, _ in hb.PLAYER_MAP_STATS["VAN"].values())
    van_d = sum(d for _, d in hb.PLAYER_MAP_STATS["VAN"].values())
    assert lat_k == van_d                                   # every LAT kill is a VAN death
    assert van_k == lat_d
    assert (lat_k, van_k) == (106, 79)


def test_hills_are_60s_and_segment_cleanly():
    hills = hb.HillSegmenter(names=hb.HILL_NAMES).segment(hb.SCORE_AT_HILL_BOUNDARY)
    assert len(hills) == len(hb.SCORE_AT_HILL_BOUNDARY) - 1   # 10 hills
    # boundaries are contiguous and ~60s apart
    for prev, nxt in zip(hills, hills[1:]):
        assert prev.t_end == nxt.t_start
    assert all(h.da >= 0 and h.db >= 0 for h in hills)


def test_spawn_inference_flags_the_lockout_hills():
    hills = hb.HillSegmenter(names=hb.HILL_NAMES).segment(hb.SCORE_AT_HILL_BOUNDARY)
    flips = {f.hill_index: f for f in hb.SpawnInference().infer(hills)}
    assert flips[3].team_locked == "VAN" and flips[3].severity == "lock"   # +45-2
    assert flips[7].team_locked == "VAN" and flips[7].severity == "lock"   # +40-6
    assert flips[5].team_locked == "LAT"                                    # VAN counter
    assert all(0.0 < f.confidence <= 0.5 for f in flips.values())          # never overconfident


def test_analysis_payload_shape():
    a = hb.analysis()
    assert len(a["hills"]) == 10
    assert set(a["key_hills"]) == {3, 5, 7}
    assert a["gunfights"]["team_kills"] == {"LAT": 106, "VAN": 79}
    assert a["hills"][2]["winner"] == "LAT" and a["hills"][2]["margin"] == 43
