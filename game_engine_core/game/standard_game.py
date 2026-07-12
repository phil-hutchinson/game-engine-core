from typing import Any, Literal

from ..models.game_result import GameResult
from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition
from ..protocols.game_ui import GameUI
from ..protocols.player import Player


class StandardGame[TPly: GamePly, TPosition: GamePosition[Any]]:

    def __init__(
        self,
        initial_position: TPosition,
        players: dict[Literal[1, -1], Player[TPly, TPosition]],
        game_ui: GameUI[TPly, TPosition],
        render_final_board: bool = True,
    ):
        self._initial_position = initial_position
        self._players = players
        self._game_ui = game_ui
        self._render_final_board = render_final_board

    def run(self) -> GameResult:
        position = self._initial_position
        opening_board = self._game_ui.text_board(position)
        game_log: list[tuple[str, str]] = []

        while position.outcome is None:
            active_id = position.active_player_id
            player = self._players[active_id]

            if player.render_before_ply:
                self._game_ui.render_board(position)

            ply = player.select_ply(position)
            position = position.apply_ply(ply)
            game_log.append((str(ply), self._game_ui.text_board(position)))

        if self._render_final_board:
            self._game_ui.render_board(position)

        # position stores outcome relative to the player to move - change to non-relative value for output purposes
        # (the Literal annotation gives the checker the context to evaluate the product as literal math)
        absolute_outcome: Literal[1, 0, -1] = position.outcome * position.active_player_id
        # The GamePosition contract guarantees a reason once outcome is non-None.
        result_reason = position.outcome_reason
        assert result_reason is not None
        return GameResult(
            outcome=absolute_outcome,
            result_reason=result_reason,
            opening_board=opening_board,
            game_log=game_log,
        )