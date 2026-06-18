from typing import Any

from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition
from ..protocols.game_engine import GameEngine
from ..protocols.player import Player

class AIPlayer[TPly: GamePly, TPosition: GamePosition[Any]](Player[TPly, TPosition]):
    def __init__(self, engine: GameEngine[TPly, TPosition], name: str, render_before_ply: bool = False):
        self._engine = engine
        self._name = name
        self._render_before_ply = render_before_ply
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def render_before_ply(self) -> bool:
        return self._render_before_ply
        
    def select_ply(self, position: TPosition) -> TPly:
        return self._engine.select_ply(position)