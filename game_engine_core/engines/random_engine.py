import random
from typing import Any

from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition


class RandomEngine[TGamePly: GamePly, TGamePosition: GamePosition[Any]]:

    def select_ply(self, game_position: TGamePosition) -> TGamePly:
        plies = list(game_position.legal_plies)
        if not plies:
            raise ValueError("No legal plies available")
        return random.choice(plies)
