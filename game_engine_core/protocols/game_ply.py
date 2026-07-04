from typing import Protocol

class GamePly(Protocol):
    """A single action taken by one player.

    The framework uses str(ply) as a ply's identity key: in evaluator policy
    dictionaries (see PositionEvaluation.policy), MCTS visit distributions, and
    training targets. str(ply) must therefore be unique among a position's legal
    plies — two legal plies that stringify identically would be silently merged.
    """

    def __str__(self) -> str:
        """Human readable representation of this ply. Can be used for display and also for player move input.

        Must be unique among the legal plies of any single position (see class docstring)."""
        ...
