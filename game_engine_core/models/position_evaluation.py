from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionEvaluation:
    value: float
    """Value head output: position evaluation in [-1.0, 1.0] from the active player's perspective."""

    policy: Mapping[str, float] | None = None
    """Policy head output: prior probabilities over legal moves, keyed by str(ply).

    Optional. When None, the MCTS engine assigns a uniform prior to every move,
    which is equivalent to standard UCT (UCB1) behaviour. Evaluators that do not
    implement a policy head should leave this as None (the default).
    """
