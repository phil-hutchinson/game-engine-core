from dataclasses import dataclass

@dataclass(frozen=True)
class PositionEvaluation:
    value: float
    """Evaluation between -1.0 and 1.0. Higher values indicate better position for the active player (current-player-relative)."""
    
