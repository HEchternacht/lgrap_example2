"""
Active-run manager — tracks in-flight completions and supports cancellation.

Each POST /v1/chat/completions call registers a run_id. Callers can cancel
it via DELETE /v1/chat/completions/{run_id}. The streaming generator polls
`is_cancelled()` and exits early when the run is cancelled.
"""
from __future__ import annotations

import asyncio
import uuid


class RunManager:
    """Thread-safe manager for active LangGraph completions."""

    def __init__(self) -> None:
        self._cancel_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_run(self) -> str:
        """Register a new run and return its unique ID."""
        run_id = str(uuid.uuid4())
        self._cancel_events[run_id] = asyncio.Event()
        return run_id

    def complete_run(self, run_id: str) -> None:
        """Mark a run as finished and release its resources."""
        self._cancel_events.pop(run_id, None)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_run(self, run_id: str) -> bool:
        """
        Request cancellation of an active run.

        Returns True if the run existed, False if not found.
        """
        event = self._cancel_events.get(run_id)
        if event is None:
            return False
        event.set()
        return True

    def is_cancelled(self, run_id: str) -> bool:
        """Return True if the run has been cancelled."""
        event = self._cancel_events.get(run_id)
        return event is not None and event.is_set()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def active_runs(self) -> list[str]:
        """Return IDs of all currently tracked runs."""
        return list(self._cancel_events.keys())


# Module-level singleton
run_manager = RunManager()
