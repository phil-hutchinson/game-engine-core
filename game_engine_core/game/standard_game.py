from typing import Literal,Tuple,List
from ..protocols.player import Player
from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition
from ..protocols.game_ui import GameUI
from ..models.game_result import GameResult

class StandardGame[TPly: GamePly, TPosition: GamePosition[GamePly]]:

    def __init__(
        self,
        initial_position: TPosition,
        players: dict[Literal[1, -1], Player[TPly, TPosition]],
        game_ui: GameUI[TPly, TPosition],
    ):
        self._initial_position = initial_position
        self._players = players
        self._game_ui = game_ui

    def run(self) -> GameResult:
        position = self._initial_position
        opening_board = self._game_ui.text_board(position)
        game_log: List[Tuple[str, str]] = []

        while position.outcome is None:
            active_id = position.active_player_id
            player = self._players[active_id]

            if player.render_before_ply:
                self._game_ui.render_board(position)

            ply = player.select_ply(position)
            position = position.apply_ply(ply)
            game_log.append((str(ply), self._game_ui.text_board(position)))

        return GameResult(
            outcome=position.outcome,
            opening_board=opening_board,
            game_log=game_log,
        )