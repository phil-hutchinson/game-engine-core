from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class GameResult:
    outcome: Literal[1, 0, -1]
    opening_board: str
    game_log: Sequence[tuple[str, str]]  # (ply annotation, board after ply)