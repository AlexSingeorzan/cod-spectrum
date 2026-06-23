"""Counterfactual Hardpoint simulation engine — the "What-If Lab".

This is a TRANSPARENT, UNCALIBRATED model, not ground truth. It maps a coded
event log (kills + spawn flips) onto a Hardpoint score timeline using a single
documented rule (the "momentum-carry control" model below), then runs
leave-one-out and intervention counterfactuals on that log.

Honesty contract:
  * Absolute scores produced here are MODEL ESTIMATES. The meaningful outputs are
    the *deltas* between baseline and counterfactual, and the *mechanism* each
    event drives (a freed-up player, a shorter spawn travel, a flipped duel).
  * Every event carries a ``synthetic`` flag. The bundled demo match is fully
    synthetic and labelled as such end to end. Replace it with a coded real
    match (``load_match_file``) once a VOD's events are tagged — the model and
    UI are identical, only the ``synthetic`` flag and source note change.

The control model (stated so a coach can argue with it):
  At each 1-second tick the active hill is controlled by whichever team has more
  *available* players (alive and returned from spawn). A kill removes the victim
  for ``respawn_delay`` seconds plus the spawn-to-hill travel time for that
  team's CURRENT spawn — which is why both "remove this kill" and "if spawns
  hadn't flipped" have a concrete, recomputable effect. When both teams have an
  equal number available, the last team to hold an advantage keeps scoring
  (momentum carry); before any advantage exists the hill is contested and no
  one scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ..config import get_settings
from .analytics import HeuristicV0

DEFAULT_TARGET = 250
DEFAULT_HILL_SECONDS = 60.0
DEFAULT_RESPAWN_DELAY = 5.0
TICK_CAP_SECONDS = 1200


# --------------------------------------------------------------------------- #
# Domain model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Hill:
    """One hill in the rotation. ``travel`` maps a spawn name -> seconds for a
    freshly respawned player to reach this hill from that spawn."""

    id: str
    name: str
    order: int
    travel: dict[str, float]


@dataclass(frozen=True)
class Player:
    name: str
    team: str


@dataclass(frozen=True)
class SimEvent:
    id: str
    t: float
    kind: Literal["kill", "spawn_flip"]
    team: str  # acting team: the killer's team, or the team whose spawn flips
    killer: str | None = None
    victim: str | None = None
    victim_team: str | None = None
    zone: str | None = None
    spawn: str | None = None  # for spawn_flip: the new spawn name for ``team``
    note: str | None = None
    synthetic: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "t": self.t,
            "kind": self.kind,
            "team": self.team,
            "killer": self.killer,
            "victim": self.victim,
            "victim_team": self.victim_team,
            "zone": self.zone,
            "spawn": self.spawn,
            "note": self.note,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class Match:
    map_name: str
    mode: str
    team_a: str
    team_b: str
    players: tuple[Player, ...]
    hills: tuple[Hill, ...]
    events: tuple[SimEvent, ...]
    default_spawn: dict[str, str]  # team -> spawn name held at map start
    synthetic: bool
    source_note: str
    target: int = DEFAULT_TARGET
    hill_seconds: float = DEFAULT_HILL_SECONDS
    respawn_delay: float = DEFAULT_RESPAWN_DELAY


@dataclass(frozen=True)
class Intervention:
    """A single what-if edit applied to the event log before re-simulation."""

    kind: Literal["remove_event", "swap_kill", "shift_time", "prevent_flip"]
    event_id: str
    delta: float = 0.0  # seconds, for shift_time

    @classmethod
    def from_dict(cls, raw: dict) -> "Intervention":
        return cls(
            kind=raw["kind"],
            event_id=str(raw["event_id"]),
            delta=float(raw.get("delta", 0.0)),
        )


@dataclass(frozen=True)
class SimResult:
    final_a: int
    final_b: int
    winner: str
    win_prob_a: float
    timeline: list[dict]  # score-change points: {t, a, b, prob_a}
    control_segments: list[dict]  # run-length: {start, end, holder}
    hill_breakdown: list[dict]  # per hill window: {window, hill, a, b}
    duration: float


# --------------------------------------------------------------------------- #
# Event-log transforms (pure)
# --------------------------------------------------------------------------- #
def apply_interventions(
    events: tuple[SimEvent, ...], interventions: tuple[Intervention, ...]
) -> tuple[SimEvent, ...]:
    by_id = {event.id: event for event in events}
    drop: set[str] = set()
    for intervention in interventions:
        event = by_id.get(intervention.event_id)
        if event is None:
            continue
        if intervention.kind in ("remove_event", "prevent_flip"):
            drop.add(event.id)
        elif intervention.kind == "swap_kill" and event.kind == "kill":
            by_id[event.id] = SimEvent(
                id=event.id,
                t=event.t,
                kind="kill",
                team=event.victim_team or event.team,
                killer=event.victim,
                victim=event.killer,
                victim_team=event.team,
                zone=event.zone,
                note=f"swapped: {event.victim} over {event.killer}",
                synthetic=event.synthetic,
            )
        elif intervention.kind == "shift_time":
            by_id[event.id] = SimEvent(
                id=event.id,
                t=max(0.0, event.t + intervention.delta),
                kind=event.kind,
                team=event.team,
                killer=event.killer,
                victim=event.victim,
                victim_team=event.victim_team,
                zone=event.zone,
                spawn=event.spawn,
                note=event.note,
                synthetic=event.synthetic,
            )
    kept = [by_id[event.id] for event in events if event.id not in drop]
    return tuple(sorted(kept, key=lambda item: item.t))


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #
def _spawn_at(team: str, t: float, flips: list[SimEvent], default: str) -> str:
    spawn = default
    for flip in flips:
        if flip.team == team and flip.t <= t and flip.spawn:
            spawn = flip.spawn
    return spawn


def _active_hill(match: Match, t: float) -> Hill:
    index = int(t // match.hill_seconds) % len(match.hills)
    return match.hills[index]


def simulate(match: Match, interventions: tuple[Intervention, ...] = ()) -> SimResult:
    events = apply_interventions(match.events, interventions)
    flips = [event for event in events if event.kind == "spawn_flip"]
    kills = [event for event in events if event.kind == "kill"]

    # Last death time per player (the most recent death <= t governs availability).
    deaths: dict[str, list[float]] = {player.name: [] for player in match.players}
    for kill in kills:
        if kill.victim in deaths:
            deaths[kill.victim].append(kill.t)
    for name in deaths:
        deaths[name].sort()

    team_of = {player.name: player.team for player in match.players}
    roster = {match.team_a: [], match.team_b: []}
    for player in match.players:
        roster[player.team].append(player.name)

    model = HeuristicV0(target=match.target)

    def available(name: str, t: float, hill: Hill) -> bool:
        prior = [death for death in deaths[name] if death <= t]
        if not prior:
            return True
        last_death = prior[-1]
        team = team_of[name]
        spawn = _spawn_at(team, last_death, flips, match.default_spawn[team])
        travel = hill.travel.get(spawn, 0.0)
        return t >= last_death + match.respawn_delay + travel

    score_a = score_b = 0
    timeline = [{"t": 0.0, "a": 0, "b": 0, "prob_a": round(model.predict(0, 0), 4)}]
    control_segments: list[dict] = []
    current_segment: dict | None = None
    window_points: dict[int, dict] = {}
    tick = 0

    while tick < TICK_CAP_SECONDS and score_a < match.target and score_b < match.target:
        t = float(tick)
        hill = _active_hill(match, t)
        avail_a = sum(available(name, t, hill) for name in roster[match.team_a])
        avail_b = sum(available(name, t, hill) for name in roster[match.team_b])

        # Strict man-advantage: the hill only ticks for a team that out-numbers the
        # enemy on it. Even counts (incl. a full 4v4) are a contested standoff and
        # score for no one — so every kill, and the spawn travel that follows it,
        # is what actually generates points.
        if avail_a > avail_b:
            scorer = match.team_a
        elif avail_b > avail_a:
            scorer = match.team_b
        else:
            scorer = None

        window = int(t // match.hill_seconds)
        bucket = window_points.setdefault(
            window, {"window": window + 1, "hill": hill.name, "a": 0, "b": 0}
        )
        if scorer == match.team_a:
            score_a += 1
            bucket["a"] += 1
        elif scorer == match.team_b:
            score_b += 1
            bucket["b"] += 1

        seg_holder = scorer or "contested"
        if current_segment is None or current_segment["holder"] != seg_holder:
            if current_segment is not None:
                current_segment["end"] = t
                control_segments.append(current_segment)
            current_segment = {"start": t, "end": t + 1, "holder": seg_holder}
        else:
            current_segment["end"] = t + 1

        if scorer is not None:
            prob_a = round(model.predict(score_a, score_b), 4)
            last = timeline[-1]
            if (score_a, score_b) != (last["a"], last["b"]):
                timeline.append({"t": t, "a": score_a, "b": score_b, "prob_a": prob_a})
        tick += 1

    if current_segment is not None:
        control_segments.append(current_segment)

    winner = match.team_a if score_a >= match.target else (
        match.team_b if score_b >= match.target else (match.team_a if score_a >= score_b else match.team_b)
    )
    hill_breakdown = [window_points[key] for key in sorted(window_points)]
    return SimResult(
        final_a=score_a,
        final_b=score_b,
        winner=winner,
        win_prob_a=round(model.predict(score_a, score_b), 4),
        timeline=timeline,
        control_segments=control_segments,
        hill_breakdown=hill_breakdown,
        duration=float(tick),
    )


def event_impacts(match: Match) -> list[dict]:
    """Leave-one-out win-probability impact per event.

    ``impact_a`` = baseline P(team_a wins) - P(team_a wins | event removed).
    Positive means the event, as it happened, helped team A. This is a defined,
    recomputable swing metric — not the broadcaster's in-game number.
    """
    base = simulate(match)
    base_margin = base.final_a - base.final_b
    impacts: list[dict] = []
    for event in match.events:
        without = simulate(match, (Intervention(kind="remove_event", event_id=event.id),))
        # Net point swing this event added to LAT's margin (positive = helped LAT).
        swing_points = base_margin - (without.final_a - without.final_b)
        impacts.append(
            {
                **event.to_dict(),
                "swing_points": swing_points,
                "win_prob_shift": round(base.win_prob_a - without.win_prob_a, 4),
                "flips_winner": without.winner != base.winner,
                "counterfactual_final": {"a": without.final_a, "b": without.final_b},
            }
        )
    impacts.sort(key=lambda item: abs(item["swing_points"]), reverse=True)
    return impacts


def to_payload(match: Match, interventions: tuple[Intervention, ...] = ()) -> dict:
    result = simulate(match, interventions)
    payload = {
        "match": {
            "map_name": match.map_name,
            "mode": match.mode,
            "team_a": match.team_a,
            "team_b": match.team_b,
            "target": match.target,
            "synthetic": match.synthetic,
            "source_note": match.source_note,
            "players": [p.__dict__ for p in match.players],
            "hills": [h.__dict__ for h in match.hills],
            "default_spawn": match.default_spawn,
        },
        "result": result.__dict__,
        "events": [event.to_dict() for event in match.events],
    }
    if not interventions:
        payload["impacts"] = event_impacts(match)
    return payload


# --------------------------------------------------------------------------- #
# Match (de)serialisation + the bundled synthetic demo
# --------------------------------------------------------------------------- #
def match_from_dict(raw: dict) -> Match:
    return Match(
        map_name=raw["map_name"],
        mode=raw.get("mode", "hardpoint"),
        team_a=raw["team_a"],
        team_b=raw["team_b"],
        players=tuple(Player(**player) for player in raw["players"]),
        hills=tuple(Hill(**hill) for hill in raw["hills"]),
        events=tuple(SimEvent(**event) for event in raw["events"]),
        default_spawn=raw["default_spawn"],
        synthetic=raw.get("synthetic", True),
        source_note=raw.get("source_note", "unspecified"),
        target=raw.get("target", DEFAULT_TARGET),
        hill_seconds=raw.get("hill_seconds", DEFAULT_HILL_SECONDS),
        respawn_delay=raw.get("respawn_delay", DEFAULT_RESPAWN_DELAY),
    )


def load_match_file(path: Path) -> Match:
    return match_from_dict(json.loads(Path(path).read_text()))


def demo_match_path() -> Path:
    return get_settings().data_dir / "fixtures" / "demo_synthetic_match.json"


def _generate_demo_events() -> tuple[SimEvent, ...]:
    """Deterministic (seeded) synthetic kill log with a slight, swinging LAT edge,
    plus two scripted set pieces: a VAN spawn flip and a HyDra triple at Garden."""
    import random

    rng = random.Random(11)
    lat = ["aBeZy", "Nium", "Scrap", "HyDra"]
    van = ["Craze", "Lunarz", "Nero", "Mamba"]
    mates = {"LAT": lat, "VAN": van}
    avail = {name: 0.0 for name in lat + van}
    respawn, nominal_travel = 5.0, 6.0
    # Per 60s window, P(the next kill is a LAT kill). The dip in window 2 hands
    # VAN the early lead; the Garden spike (window 3) is the LAT surge.
    forms = [0.53, 0.43, 0.61, 0.46, 0.52, 0.49, 0.52]
    zones = ["Mansion", "Stables", "Garden", "Comms"]
    events: list[SimEvent] = []
    count = 0
    t = 4.0
    while t < 900.0:
        if 137.0 <= t < 151.0:  # reserved for the scripted triple
            t += 3.0
            continue
        window = int(t // 60)
        p_lat = forms[min(window, len(forms) - 1)]
        killer_team = "LAT" if rng.random() < p_lat else "VAN"
        victim_team = "VAN" if killer_team == "LAT" else "LAT"
        killers = [name for name in mates[killer_team] if avail[name] <= t] or mates[killer_team]
        victims = [name for name in mates[victim_team] if avail[name] <= t]
        if not victims:
            t += rng.uniform(1.6, 3.4)
            continue
        killer = rng.choice(killers)
        victim = rng.choice(victims)
        avail[victim] = t + respawn + nominal_travel
        count += 1
        events.append(SimEvent(
            id=f"k{count:03d}", t=round(t, 1), kind="kill", team=killer_team,
            killer=killer, victim=victim, victim_team=victim_team,
            zone=zones[window % 4], synthetic=True,
        ))
        t += rng.uniform(1.6, 3.6)

    events.append(SimEvent(
        id="f1", t=132.0, kind="spawn_flip", team="VAN", spawn="V-South",
        note="VAN flips to the South spawn — long travel back to Garden", synthetic=True,
    ))
    for index, (when, victim) in enumerate(zip([140.0, 144.0, 148.0], ["Craze", "Lunarz", "Nero"]), start=1):
        avail[victim] = when + respawn + nominal_travel
        events.append(SimEvent(
            id=f"H{index}", t=when, kind="kill", team="LAT", killer="HyDra",
            victim=victim, victim_team="VAN", zone="Garden",
            note=f"HyDra {index}/3 — Garden hero window", synthetic=True,
        ))
    events.sort(key=lambda event: event.t)
    return tuple(events)


def build_demo_match() -> Match:
    """A fully SYNTHETIC Hacienda-style Hardpoint used to exercise the engine and
    UI. Team and player names mirror the real matchup so the layout is realistic,
    but every kill, spawn flip and timestamp below is fabricated for demonstration
    and is NOT extracted from any VOD. Swapped for real coded events via
    ``load_match_file`` once a match is tagged."""
    lat = ["aBeZy", "Nium", "Scrap", "HyDra"]
    van = ["Craze", "Lunarz", "Nero", "Mamba"]
    players = tuple(
        [Player(name=name, team="LAT") for name in lat]
        + [Player(name=name, team="VAN") for name in van]
    )
    # Each team owns its own spawn label. The scripted VAN flip to "V-South" is
    # deliberately self-harming at Garden (travel 13 vs 4) so "prevent the flip"
    # has a visible payoff, while paying off mildly at Comms (5 vs 8).
    hills = (
        Hill(id="h1", name="Mansion", order=1, travel={"L-Base": 6, "V-Base": 6, "V-South": 7}),
        Hill(id="h2", name="Stables", order=2, travel={"L-Base": 6, "V-Base": 5, "V-South": 8}),
        Hill(id="h3", name="Garden", order=3, travel={"L-Base": 5, "V-Base": 4, "V-South": 13}),
        Hill(id="h4", name="Comms", order=4, travel={"L-Base": 7, "V-Base": 8, "V-South": 5}),
    )
    match = Match(
        map_name="Hacienda (synthetic)",
        mode="hardpoint",
        team_a="LAT",
        team_b="VAN",
        players=players,
        hills=hills,
        events=_generate_demo_events(),
        default_spawn={"LAT": "L-Base", "VAN": "V-Base"},
        synthetic=True,
        source_note="SYNTHETIC DEMO — fabricated event log, not extracted from any VOD.",
    )
    # Drop generated events that fall after the map naturally ends at 250.
    end = simulate(match).duration
    return replace(match, events=tuple(event for event in match.events if event.t <= end + 1.0))


def get_active_match() -> Match:
    """Real coded match if one has been tagged, else the synthetic demo."""
    path = demo_match_path()
    if path.exists():
        match = load_match_file(path)
        if not match.synthetic:
            return match
    return build_demo_match()
