"""Compatibility wrapper for the Phase 5 kill-type recognizer.

The old task name was "weapon recognition", but the platform now classifies
coarse ``kill_type`` values. New code should import
``backend.app.services.kill_type_recognition`` directly.
"""
from __future__ import annotations

from .kill_type_recognition import *  # noqa: F401,F403
from .kill_type_recognition import KillTypeRecognizer, kill_type_event_from_prediction

WeaponRecognizer = KillTypeRecognizer
weapon_event_from_prediction = kill_type_event_from_prediction
