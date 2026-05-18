from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ObservationSummary:
    length: int
    nonzero: int
    total: float
    positive_ratio: float
    max_value: float


def _summarize_obs(obs: Iterable[float]) -> ObservationSummary:
    arr = np.asarray(obs, dtype=float).ravel()
    if arr.size == 0:
        return ObservationSummary(0, 0, 0.0, 0.0, 0.0)

    nonzero = int(np.count_nonzero(arr))
    return ObservationSummary(
        length=int(arr.size),
        nonzero=nonzero,
        total=float(arr.sum()),
        positive_ratio=float(nonzero / arr.size),
        max_value=float(arr.max(initial=0.0)),
    )


def _action_phrase(prev_action_name: str) -> str:
    lowered = prev_action_name.lower()
    if "initial" in lowered:
        return "The episode has just started. No host has been compromised yet."
    if "scan" in lowered:
        return "A subnet scan was performed, increasing network discovery and reachability knowledge."
    if "exploit" in lowered:
        return "A remote exploit was attempted against a discovered service to gain user access."
    if "root" in lowered or "privilege" in lowered:
        return "Privilege escalation succeeded and root-level control was obtained on a host."
    return f"The previous action was {prev_action_name}."


def obs_to_text(
    obs,
    action_space_n: int,
    prev_action_name: str,
    step: int,
    scenario_id: str,
) -> str:
    """
    Convert a NASim flat observation into a compact natural-language state.

    NASim's flat vector layout varies by scenario and observation settings. This
    function therefore records robust aggregate signals from the vector while
    preserving the human-readable optimal-path label supplied by the benchmark.
    """
    summary = _summarize_obs(obs)
    action_phrase = _action_phrase(prev_action_name)

    return (
        f"Scenario {scenario_id}, step {step}. {action_phrase} "
        f"Observation vector has {summary.length} features, {summary.nonzero} active "
        f"entries, active ratio {summary.positive_ratio:.3f}, total signal "
        f"{summary.total:.3f}, max value {summary.max_value:.3f}. "
        f"The action space contains {action_space_n} possible actions. "
        "Progress should increase as the agent discovers hosts, exploits services, "
        "escalates privileges, and reaches root access on the target."
    )
