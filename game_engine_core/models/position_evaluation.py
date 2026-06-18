from dataclasses import dataclass

@dataclass(frozen=True)
class PositionEvaluation:
    value: float
    """Evaluation between -1.0 and 1.0. Higher values indicate better position for player 1."""
    
    evaluation_cost: float
    """Recourse cost of this evaluation."""
    
