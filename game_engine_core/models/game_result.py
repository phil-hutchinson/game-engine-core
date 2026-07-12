from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GameResult:
    outcome: Literal[1, 0, -1]
    result_reason: str  # the terminal position's outcome_reason (why the game ended)
    opening_board: str
    game_log: Sequence[tuple[str, str]]  # (ply annotation, board after ply)
