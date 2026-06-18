import random
from typing import Any

from game_engine_core.protocols.game_ply import GamePly
from game_engine_core.protocols.game_position import GamePosition


class RandomEngine[TGamePly: GamePly, TGamePosition: GamePosition[Any]]:

    def select_ply(self, game_position: TGamePosition) -> TGamePly:
        plies = list(game_position.legal_plies)
        if not plies:
            raise ValueError("No legal plies available")
        return random.choice(plies)
