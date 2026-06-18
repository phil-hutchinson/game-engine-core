from ..protocols.player import Player
from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition
from ..protocols.game_ui import GameUI

class HumanPlayer[TPly: GamePly, TPosition: GamePosition[GamePly]](Player[TPly, TPosition]):
    
    def __init__(self, game_ui: GameUI[TPly, TPosition], name: str, render_before_ply: bool = True):
        self._game_ui = game_ui
        self._name = name
        self._render_before_ply = render_before_ply

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def render_before_ply(self) -> str:
        return self._render_before_ply
        
    def select_ply(self, position: TPosition) -> TPly:
        return self._game_ui.get_next_ply(position)