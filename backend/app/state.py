from __future__ import annotations

from .models import Broadcast, BroadcastStatus, utcnow


LEGAL_TRANSITIONS: dict[BroadcastStatus, set[BroadcastStatus]] = {
    BroadcastStatus.discovered: {BroadcastStatus.downloading, BroadcastStatus.downloaded, BroadcastStatus.processing, BroadcastStatus.skipped, BroadcastStatus.failed},
    BroadcastStatus.downloading: {BroadcastStatus.downloaded, BroadcastStatus.failed},
    BroadcastStatus.downloaded: {BroadcastStatus.processing, BroadcastStatus.failed},
    BroadcastStatus.processing: {BroadcastStatus.processed, BroadcastStatus.failed},
    BroadcastStatus.failed: {BroadcastStatus.downloading, BroadcastStatus.processing, BroadcastStatus.skipped},
    BroadcastStatus.processed: set(),
    BroadcastStatus.skipped: set(),
}


def transition_broadcast(broadcast: Broadcast, target: BroadcastStatus, error: str | None = None) -> None:
    if target == broadcast.status:
        return
    if target not in LEGAL_TRANSITIONS[broadcast.status]:
        raise ValueError(f"illegal broadcast status transition: {broadcast.status.value} -> {target.value}")
    now = utcnow()
    history = list(broadcast.status_history or [])
    history.append({"from": broadcast.status.value, "to": target.value, "at": now.isoformat(), "error": error})
    broadcast.status_history = history
    broadcast.status = target
    broadcast.status_changed_at = now
    broadcast.last_error = error

