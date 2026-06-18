from typing import Protocol

class GamePly(Protocol):
    def __str__(self) -> str:
        """Human readable representation of this ply. Can be used for display and also for player move input."""
        ...
