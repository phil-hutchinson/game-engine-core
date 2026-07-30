"""SelfPlayCollector tests, centred on target-value signs.

With the Nim fixture in forced-line mode (takes of exactly 1) every game is fully
deterministic: from pile 3 the plies are P1 (3->2), P2 (2->1), P1 (1->0), so
player 1 takes the last token and wins. That makes every sample's target value
exactly predictable: the terminal outcome is stated from the perspective of the
player who would move next (the loser here), so the last recorded step — made by
the *other* player — gets -final_outcome, alternating backwards from there.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

from torch import Tensor

from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator
from game_engine_core.game.batch_position_processor import BatchPositionProcessor
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


def _piles(*starting_piles: int) -> Callable[[], NimPosition]:
    """A position factory handing out a different starting pile per game, in slot order.

    The factory stays scalar under the fleet — a game's starting position is a
    per-game concern — so successive calls are what distinguish one slot from another.
    """
    remaining = iter(starting_piles)
    return lambda: NimPosition(pile=next(remaining), takes=(1,))


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
        positions: Sequence[NimPosition], policies: Sequence[dict[str, float]]
    ) -> Sequence[dict[str, float]]:
        return [
            {f"{position.active_player_id}:{ply}": p for ply, p in policy.items()}
            for position, policy in zip(positions, policies, strict=True)
        ]

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


def test_policy_transform_receives_one_batch_per_turn_paired_by_index() -> None:
    # The transform is called once per fleet-turn with every live game's position,
    # and a transform that re-keys by index must land each result on the game that
    # row came from. Piles 3, 4 and 5 finish at turns 3, 4 and 5, so the batch
    # narrows as games retire — and a game's row moves when an earlier game leaves,
    # which is what makes a misrouted result visible rather than coincidentally right.
    received: list[tuple[Sequence[NimPosition], Sequence[dict[str, float]]]] = []

    def record_and_tag_by_index(
        positions: Sequence[NimPosition], policies: Sequence[dict[str, float]]
    ) -> Sequence[dict[str, float]]:
        received.append((positions, policies))
        return [
            {f"row{i}:{ply}": p for ply, p in policy.items()}
            for i, policy in enumerate(policies)
        ]

    collector = SelfPlayCollector(
        evaluator=NimNNEvaluator(model=NimMLP()),
        engine_factory=lambda: MCTSEngine(evaluator=NullEvaluator(), iterations=10),
        position_factory=_piles(3, 4, 5),
        policy_transform=record_and_tag_by_index,
    )
    samples = collector.collect(n_games=3)

    for positions, policies in received:
        assert len(positions) == len(policies)
    # One call per turn, each carrying the live games in slot order: all three
    # count down together until the pile-3 game retires, then the pile-4 one.
    assert [[position.pile for position in positions] for positions, _ in received] == [
        [3, 4, 5],
        [2, 3, 4],
        [1, 2, 3],
        [1, 2],
        [1],
    ]
    # Slot order in, slot order out. Each game's samples run newest step first, so
    # the tags read backwards through that game's turns: the pile-5 game was row 2
    # until the pile-3 game left, then row 1, then row 0 on its final turn.
    assert [sample.target_policy for sample in samples] == [
        {"row0:1": 1.0}, {"row0:1": 1.0}, {"row0:1": 1.0},
        {"row0:1": 1.0}, {"row1:1": 1.0}, {"row1:1": 1.0}, {"row1:1": 1.0},
        {"row0:1": 1.0}, {"row1:1": 1.0}, {"row2:1": 1.0}, {"row2:1": 1.0}, {"row2:1": 1.0},
    ]


def test_a_fleet_matches_the_same_games_played_one_at_a_time() -> None:
    # The fleet changes only the driving. A fleet of three deterministic games must
    # return exactly what three separate one-game collects returned — same samples,
    # same order — which also pins that equal starting positions stay independent.
    sequential = [
        sample
        for _ in range(3)
        for sample in _collector(starting_pile=3).collect(n_games=1)
    ]
    fleet = _collector(starting_pile=3).collect(n_games=3)

    assert len(fleet) == len(sequential)
    for fleet_sample, sequential_sample in zip(fleet, sequential, strict=True):
        assert fleet_sample.target_value == sequential_sample.target_value
        assert fleet_sample.target_policy == sequential_sample.target_policy
        assert fleet_sample.encoded_position.equal(sequential_sample.encoded_position)


def test_games_of_different_lengths_retire_without_disturbing_each_other() -> None:
    # Piles 5, 3 and 4 run 5, 3 and 4 turns, so the games retire in the order slot 1,
    # slot 2, slot 0. Each game's values must alternate back from its own winner and
    # its encodings count down its own pile, unaffected by the others' lengths — and
    # the deliberately unsorted piles mean returning the buckets in the order the
    # games finished would fail here, where an ascending fleet would coincide with
    # slot order and hide it.
    collector = SelfPlayCollector(
        evaluator=NimNNEvaluator(model=NimMLP()),
        engine_factory=lambda: MCTSEngine(evaluator=NullEvaluator(), iterations=10),
        position_factory=_piles(5, 3, 4),
    )
    samples = collector.collect(n_games=3)

    # Slot order, each game newest step first: pile 5 (player 1 wins), pile 3
    # (player 1 wins), pile 4 (player 2 wins).
    assert [sample.target_value for sample in samples] == [
        1.0, -1.0, 1.0, -1.0, 1.0,
        1.0, -1.0, 1.0,
        1.0, -1.0, 1.0, -1.0,
    ]
    assert [float(sample.encoded_position[0]) for sample in samples] == [
        1.0, 2.0, 3.0, 4.0, 5.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0, 4.0,
    ]


def test_one_engine_serves_every_game_in_the_collect() -> None:
    # The fleet hands one engine every live game's position at once, so the engine
    # is per-collect rather than per-game. Safe because the training path retains
    # nothing between calls, and required because lockstep needs a single shared
    # iteration budget.
    engines: list[MCTSEngine[NimPly, NimPosition, Any]] = []

    def engine_factory() -> MCTSEngine[NimPly, NimPosition, Any]:
        engine: MCTSEngine[NimPly, NimPosition, Any] = MCTSEngine(
            evaluator=NullEvaluator(), iterations=10
        )
        engines.append(engine)
        return engine

    collector = SelfPlayCollector(
        evaluator=NimNNEvaluator(model=NimMLP()),
        engine_factory=engine_factory,
        position_factory=lambda: NimPosition(pile=3, takes=(1,)),
    )
    collector.collect(n_games=3)

    assert len(engines) == 1


class _WidthRecordingEvaluator(NimNNEvaluator):
    """Records the width of every encode_positions call, then encodes as usual."""

    def __init__(self, widths: list[int]):
        super().__init__(model=NimMLP())
        self._widths = widths

    def encode_positions(self, positions: Sequence[NimPosition]) -> Tensor:
        self._widths.append(len(positions))
        return super().encode_positions(positions)


class _WidthRecordingBatchProcessor(BatchPositionProcessor[NimPly, NimPosition]):
    """Records the width of each seam call, then delegates to the base loop."""

    def __init__(self, outcome_widths: list[int], apply_widths: list[int]):
        self._outcome_widths = outcome_widths
        self._apply_widths = apply_widths

    def outcomes(self, positions: Sequence[NimPosition]) -> Sequence[Literal[1, 0, -1] | None]:
        self._outcome_widths.append(len(positions))
        return super().outcomes(positions)

    def apply_plies(
        self, positions: Sequence[NimPosition], plies: Sequence[NimPly]
    ) -> Sequence[NimPosition]:
        self._apply_widths.append(len(positions))
        return super().apply_plies(positions, plies)


class _WidthRecordingEngine(MCTSEngine[NimPly, NimPosition, Any]):
    """Records the width of every fleet search, then searches as usual."""

    def __init__(self, widths: list[int]):
        super().__init__(evaluator=NullEvaluator(), iterations=10)
        self._widths = widths

    def select_plies_for_training(
        self, positions: Sequence[NimPosition]
    ) -> Sequence[tuple[NimPly, dict[str, float]]]:
        self._widths.append(len(positions))
        return super().select_plies_for_training(positions)


def test_each_seam_is_called_once_per_turn_at_the_live_fleet_width() -> None:
    # The point of the fleet: a turn is one call of width N through each seam, not N
    # calls of width one. Piles 5, 3 and 4 take 5, 3 and 4 turns, so the fleet plays
    # 5 turns and 12 plies in total — a width-one driver would show 12 calls of width
    # 1 through every seam below, and the terminal test would be called 15 times.
    search_widths: list[int] = []
    encode_widths: list[int] = []
    transform_widths: list[int] = []
    outcome_widths: list[int] = []
    apply_widths: list[int] = []

    def record_transform_width(
        positions: Sequence[NimPosition], policies: Sequence[dict[str, float]]
    ) -> Sequence[dict[str, float]]:
        transform_widths.append(len(positions))
        return policies

    collector = SelfPlayCollector(
        evaluator=_WidthRecordingEvaluator(encode_widths),
        engine_factory=lambda: _WidthRecordingEngine(search_widths),
        position_factory=_piles(5, 3, 4),
        policy_transform=record_transform_width,
        batch_ops=_WidthRecordingBatchProcessor(outcome_widths, apply_widths),
    )
    collector.collect(n_games=3)

    # Five turns, narrowing as the pile-3 game retires after turn 3 and the pile-4
    # game after turn 4.
    assert search_widths == [3, 3, 3, 2, 1]
    assert encode_widths == search_widths
    assert transform_widths == search_widths
    assert apply_widths == search_widths
    # One terminal test per turn plus the final one that empties the fleet, each
    # covering the games still live at the top of that turn.
    assert outcome_widths == [3, 3, 3, 3, 2, 1]


def test_collect_accumulates_across_games() -> None:
    samples = _collector(starting_pile=3).collect(n_games=2)
    assert len(samples) == 6
    assert [sample.target_value for sample in samples] == [1.0, -1.0, 1.0] * 2
