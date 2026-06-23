from __future__ import annotations

from backend.app.services import simulation as sim


def test_demo_is_synthetic_and_labelled():
    match = sim.build_demo_match()
    assert match.synthetic is True
    assert "SYNTHETIC" in match.source_note.upper()
    assert all(event.synthetic for event in match.events)


def test_simulation_is_deterministic():
    match = sim.build_demo_match()
    first = sim.simulate(match)
    second = sim.simulate(match)
    assert (first.final_a, first.final_b) == (second.final_a, second.final_b)
    assert first.timeline == second.timeline


def test_demo_match_is_won_and_bounded():
    match = sim.build_demo_match()
    result = sim.simulate(match)
    assert max(result.final_a, result.final_b) == match.target
    assert result.winner == "LAT"
    assert 0.0 <= result.win_prob_a <= 1.0
    for point in result.timeline:
        assert 0.0 <= point["prob_a"] <= 1.0
        assert point["a"] <= match.target and point["b"] <= match.target


def test_removing_hydra_triple_cannot_help_lat():
    match = sim.build_demo_match()
    base = sim.simulate(match)
    triple = tuple(
        sim.Intervention(kind="remove_event", event_id=eid) for eid in ("H1", "H2", "H3")
    )
    without = sim.simulate(match, triple)
    # Taking away LAT's hero window must not improve LAT's margin or VAN's score.
    assert without.final_b >= base.final_b
    assert (without.final_a - without.final_b) <= (base.final_a - base.final_b)


def test_spawn_flip_is_a_top_lever():
    match = sim.build_demo_match()
    flip = next(item for item in sim.event_impacts(match) if item["id"] == "f1")
    # The systemic spawn error swings far more than any single kill in a dense log.
    assert flip["swing_points"] >= 15


def test_preventing_van_spawn_flip_helps_van():
    match = sim.build_demo_match()
    base = sim.simulate(match)
    held = sim.simulate(match, (sim.Intervention(kind="prevent_flip", event_id="f1"),))
    # The scripted flip sends VAN to the long spawn, so undoing it should not cost
    # them points and should not reduce their final score below the baseline.
    assert held.final_b >= base.final_b


def test_event_impacts_sorted_and_bounded():
    match = sim.build_demo_match()
    impacts = sim.event_impacts(match)
    assert len(impacts) == len(match.events)
    swings = [abs(item["swing_points"]) for item in impacts]
    assert swings == sorted(swings, reverse=True)
    assert all(-1.0 <= item["win_prob_shift"] <= 1.0 for item in impacts)


def test_swap_kill_inverts_duel():
    match = sim.build_demo_match()
    first = next(event for event in match.events if event.kind == "kill")
    swapped = sim.apply_interventions(
        match.events, (sim.Intervention(kind="swap_kill", event_id=first.id),)
    )
    event = next(item for item in swapped if item.id == first.id)
    assert event.killer == first.victim and event.victim == first.killer
    assert event.team == first.victim_team and event.victim_team == first.team


def test_payload_round_trips_through_dict():
    match = sim.build_demo_match()
    payload = sim.to_payload(match)
    assert payload["match"]["synthetic"] is True
    assert "impacts" in payload
    rebuilt = sim.match_from_dict(
        {
            **payload["match"],
            "events": payload["events"],
        }
    )
    assert sim.simulate(rebuilt).final_a == sim.simulate(match).final_a
