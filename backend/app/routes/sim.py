from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services import simulation as sim

router = APIRouter(prefix="/api/sim", tags=["simulation"])


class InterventionIn(BaseModel):
    kind: Literal["remove_event", "swap_kill", "shift_time", "prevent_flip"]
    event_id: str
    delta: float = 0.0


class CounterfactualIn(BaseModel):
    interventions: list[InterventionIn] = Field(default_factory=list, max_length=200)


@lru_cache(maxsize=1)
def _baseline_payload() -> dict:
    """Baseline match + per-event leave-one-out impacts (deterministic, so cached)."""
    return sim.to_payload(sim.get_active_match())


def _result_dict(result: sim.SimResult) -> dict:
    return {
        "final_a": result.final_a,
        "final_b": result.final_b,
        "winner": result.winner,
        "win_prob_a": result.win_prob_a,
        "timeline": result.timeline,
        "control_segments": result.control_segments,
        "hill_breakdown": result.hill_breakdown,
        "duration": result.duration,
    }


@router.get("/match")
def get_match() -> dict:
    """Baseline simulation of the active match (synthetic demo until a real coded
    match is dropped in), including the win-probability timeline, control
    segments, hill breakdown and per-event win-probability impacts."""
    return _baseline_payload()


@router.post("/counterfactual")
def counterfactual(body: CounterfactualIn) -> dict:
    """Re-simulate the active match with a set of what-if edits applied, and
    return the baseline vs counterfactual deltas. Results are MODEL ESTIMATES."""
    match = sim.get_active_match()
    interventions = tuple(
        sim.Intervention(kind=item.kind, event_id=item.event_id, delta=item.delta)
        for item in body.interventions
    )
    baseline = sim.simulate(match)
    result = sim.simulate(match, interventions)
    return {
        "synthetic": match.synthetic,
        "interventions": [item.model_dump() for item in body.interventions],
        "baseline": _result_dict(baseline),
        "counterfactual": _result_dict(result),
        "delta": {
            "margin_baseline": baseline.final_a - baseline.final_b,
            "margin_counterfactual": result.final_a - result.final_b,
            "team_a_change": result.final_a - baseline.final_a,
            "team_b_change": result.final_b - baseline.final_b,
            "winner_flipped": result.winner != baseline.winner,
            "winner": result.winner,
        },
    }
