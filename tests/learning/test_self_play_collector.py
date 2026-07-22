"""SelfPlayCollector tests, centred on target-value signs.

With the Nim fixture in forced-line mode (takes of exactly 1) every game is fully
deterministic: from pile 3 the plies are P1 (3->2), P2 (2->1), P1 (1->0), so
player 1 takes the last token and wins. That makes every sample's target value
exactly predictable: the terminal outcome is stated from the perspective of the
player who would move next (the loser here), so the last recorded step — made by
the *other* player — gets -final_outcome, alternating backwards from there.
"""

from typing import Any

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator
from game_engine_learning.self_play_collector import SelfPlayCollector
from tests.core.nim_fixture import NimPly, NimPosition

from .nim_nn import NimMLP, NimNNEvaluator


def _collector(starting_pile: int) -> SelfPlayCollector[NimPly, NimPosition]:
    def engine_factory() -> MCTSEngine[NimPly, NimPosition, Any]:
        return MCTSEngine(evaluator=NullEvaluator(), iterations=10)

    return SelfPlayCollector(
        evaluator=NimNNEvaluator(model=NimMLP()),
        engine_factory=engine_factory,
        position_factory=lambda: NimPosition(pile=starting_pile, takes=(1,)),
    )


def test_target_values_alternate_back_from_the_winner() -> None:
    # Pile 3: player 1 wins. Samples are emitted last step first; the winner's
    # steps carry +1, the loser's -1.
    samples = _collector(starting_pile=3).collect(n_games=1)
    assert [sample.target_value for sample in samples] == [1.0, -1.0, 1.0]


def test_target_values_when_player_two_wins() -> None:
    # Pile 4: player 2 takes the last token, so the signs shift by one ply.
    samples = _collector(starting_pile=4).collect(n_games=1)
    assert [sample.target_value for sample in samples] == [1.0, -1.0, 1.0, -1.0]


def test_encodings_pair_with_their_steps() -> None:
    # Samples run last step first, so the recorded pile sizes are 1, 2, 3 —
    # confirming each target value is attached to the right position.
    samples = _collector(starting_pile=3).collect(n_games=1)
    assert [float(sample.encoded_position[0]) for sample in samples] == [1.0, 2.0, 3.0]


def test_policy_targets_cover_the_legal_plies() -> None:
    samples = _collector(starting_pile=3).collect(n_games=1)
    # Forced line: one legal ply per position, so every visit lands on it.
    assert all(sample.target_policy == {"1": 1.0} for sample in samples)


def test_policy_transform_reframes_targets_using_the_position() -> None:
    # A transform keyed on active_player_id proves the position is in scope at
    # capture. Over the forced pile-3 line the mover alternates 1, -1, 1, so the
    # re-keyed distributions differ per step in a way only the position reveals.
    def prefix_with_mover(
        position: NimPosition, policy: dict[str, float]
    ) -> dict[str, float]:
        return {f"{position.active_player_id}:{ply}": p for ply, p in policy.items()}

    collector = SelfPlayCollector(
        evaluator=NimNNEvaluator(model=NimMLP()),
        engine_factory=lambda: MCTSEngine(evaluator=NullEvaluator(), iterations=10),
        position_factory=lambda: NimPosition(pile=3, takes=(1,)),
        policy_transform=prefix_with_mover,
    )
    samples = collector.collect(n_games=1)
    # Emitted last step first: pile 1 (player 1), pile 2 (player -1), pile 3 (player 1).
    assert [sample.target_policy for sample in samples] == [
        {"1:1": 1.0},
        {"-1:1": 1.0},
        {"1:1": 1.0},
    ]


def test_collect_accumulates_across_games() -> None:
    samples = _collector(starting_pile=3).collect(n_games=2)
    assert len(samples) == 6
    assert [sample.target_value for sample in samples] == [1.0, -1.0, 1.0] * 2
