from game_engine_core.protocols.game_ply import GamePly

class TicTacToePly(GamePly):
    
    def __init__(self, square: int):
        if not 1 <= square <= 9:
            raise ValueError(f"Square must be between 1 and 9, got {square}")
        self._square = square
    
    @property
    def square(self) -> int:
        """Board location 1-9, left to right, top to bottom. e.g. top-right is 3, bottom-left is 7."""
        return self._square
    
    def __str__(self) -> str:
        return str(self._square)