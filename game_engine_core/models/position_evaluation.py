from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionEvaluation:
    value: float
    """Value head output: position evaluation in [-1.0, 1.0] from the active player's perspective."""

    policy: Mapping[str, float]
    """Policy head output: prior probabilities over legal moves, keyed by str(ply).

    Required, and must contain an entry for every legal ply of the evaluated
    position — MCTS expands a leaf by attaching all of its children at once, and
    each one needs its prior at construction, so there is no path through the
    search where a prior is optional. An evaluator with no policy head should
    return a uniform distribution over the legal plies (see NullEvaluator);
    doing so here rather than in the engine keeps the cost on the evaluators
    that need it.
    """
