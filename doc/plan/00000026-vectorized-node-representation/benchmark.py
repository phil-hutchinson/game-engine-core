"""Search micro-benchmark for issue #26 (vectorised node representation).

A measuring instrument for this story, not a deliverable: it exists to give
every step of the implementation plan a before-number, and to show whether the
vectorised representation pays for itself at a realistic branching factor. It is
not part of the test suite and is not collected by pytest (``testpaths`` covers
``tests`` and ``examples`` only).

Four cells: a narrow position (the Nim fixture, branching 2) and a wide one (the
TicTacToe example, branching up to 9), each at a fleet of 1 and a fleet of 64.
Both fleet sizes go through ``select_plies_for_training`` so fleet size is the
only variable between them — ``select_ply`` differs by retaining its root, which
would confound the comparison.

Everything is fixed-seed and the evaluator is a cheap deterministic stand-in, so
two runs of this script on an idle machine should agree to within a few percent
on the timings and exactly on the result signatures.

Run it from anywhere:

    python doc/plan/00000026-vectorized-node-representation/benchmark.py

Reported per cell:

    wall        seconds for one full ``select_plies_for_training`` call. The
                headline is the fastest repeat, not the median: the noise here
                is one-sided (a GC pause or another process lands on a repeat
                and only ever adds time), so the minimum is the most stable
                estimator of the work itself. The median and the spread are
                printed alongside as the check on that — a large spread means
                the machine was busy and the run should not be compared.
    iters/s     tree-iterations per second: ``iterations x fleet / wall``. This
                is the throughput number to compare across steps, and it is
                comparable across fleet sizes — the plain wave rate is not,
                since one wave at fleet 64 is 64 trees' worth of work.
    signature   a checksum over the search result of slot 0. The story's
                non-goal is that search results do not move, so this must stay
                identical across every step of the plan for a given cell.
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The benchmark lives beside the plan it belongs to rather than in a package, so
# the repo root has to be put on the path before the fixtures can be imported.
# game_engine_core itself is installed editable and needs no help.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tictactoe.tictactoe_position import TicTacToePosition  # noqa: E402
from game_engine_core.engines.mcts_engine import MCTSEngine  # noqa: E402
from game_engine_core.models.position_evaluation import PositionEvaluation  # noqa: E402
from game_engine_core.protocols.game_position import GamePosition  # noqa: E402
from tests.core.nim_fixture import NimPosition  # noqa: E402

DEFAULT_ITERATIONS = 800
DEFAULT_SEED = 20260730
FLEET_SIZES = (1, 64)

# Fleet 64 does 64 trees' work per repeat, so it earns a stable measurement in
# far fewer of them than fleet 1 does — a fleet-1 repeat is tens of milliseconds
# and cheap to take a lot of.
REPEATS_BY_FLEET = {1: 15, 64: 3}

# Deep enough that descent has somewhere to go: takes of 1-2 give a branching
# factor of 2 at every non-trivial pile, which is the narrow case the plan warns
# numpy's per-call overhead may lose at.
NIM_PILE = 21


class SeededEvaluator:
    """Deterministic stand-in for a network, priced to stay out of the way.

    Draws a value and a prior per legal ply from a single seeded stream rather
    than from the position, which keeps it to a handful of RNG calls per
    evaluation — the point is to time the tree, not the evaluator. Drawing
    sequentially rather than per-position makes the output a function of the
    order evaluations are requested in, so identical search behaviour gives
    identical numbers and any drift in descent shape shows up in the signature.

    Priors are non-uniform on purpose: a uniform policy (as ``NullEvaluator``
    gives) leaves the PUCT exploration term constant across siblings, which is
    not the array the vectorised selection will actually be scoring.
    """

    def __init__(self, seed: int):
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def evaluate_positions(self, positions: Sequence[Any]) -> Sequence[PositionEvaluation]:
        evaluations: list[PositionEvaluation] = []
        for position in positions:
            legal_plies = position.legal_plies
            weights = [self._rng.random() for _ in legal_plies]
            total = sum(weights)
            if total == 0.0:
                total = 1.0
            evaluations.append(PositionEvaluation(
                value=self._rng.uniform(-1.0, 1.0),
                policy={
                    str(ply): weight / total
                    for ply, weight in zip(legal_plies, weights, strict=True)
                },
            ))
        return evaluations


@dataclass(frozen=True)
class Cell:
    """One measurement: a position shape at a fleet size."""

    name: str
    fleet: int
    make_position: Callable[[], GamePosition[Any]]


@dataclass(frozen=True)
class Measurement:
    cell: Cell
    iterations: int
    walls: list[float]
    signature: str

    @property
    def median_wall(self) -> float:
        return statistics.median(self.walls)

    @property
    def best_wall(self) -> float:
        return min(self.walls)

    @property
    def spread_pct(self) -> float:
        """Slowest repeat's excess over the fastest, as a percentage.

        A wide spread means the machine was busy and the numbers should not be
        compared against another run.
        """
        return (max(self.walls) - min(self.walls)) / min(self.walls) * 100.0

    @property
    def tree_iterations_per_second(self) -> float:
        return self.iterations * self.cell.fleet / self.best_wall


def _result_signature(results: Sequence[tuple[Any, dict[str, float]]]) -> str:
    """Checksum the search result of slot 0, to six decimal places.

    Full float equality would make the signature hostage to reassociation — the
    vectorised sum of Step 3 need not add in the same order as the Python loop it
    replaces. Six places is far tighter than any behavioural change would be and
    loose enough to survive that.
    """
    ply, policy = results[0]
    rendered = f"{ply}|" + ",".join(
        f"{key}:{value:.6f}" for key, value in sorted(policy.items())
    )
    return f"{zlib.crc32(rendered.encode()):08x}"


def measure(cell: Cell, iterations: int, seed: int, repeats: int) -> Measurement:
    positions = [cell.make_position() for _ in range(cell.fleet)]
    evaluator = SeededEvaluator(seed)

    walls: list[float] = []
    signature = ""
    for _ in range(repeats):
        # A fresh engine and a rewound evaluator per repeat: select_plies_for_training
        # retains nothing between calls, but the evaluator's stream would otherwise
        # carry over and each repeat would search a different tree.
        evaluator.reset()
        random.seed(seed)
        engine: MCTSEngine[Any, Any, Any] = MCTSEngine(evaluator=evaluator, iterations=iterations)

        # Nodes hold their parent and their parent holds them, so a discarded
        # tree is a reference cycle that only the cyclic collector can reclaim.
        # Without this the previous repeat's fleet is collected somewhere inside
        # this one's timing, which is where the outlier repeats came from.
        gc.collect()

        start = time.perf_counter()
        results = engine.select_plies_for_training(positions)
        walls.append(time.perf_counter() - start)

        signature = _result_signature(list(results))

    return Measurement(cell=cell, iterations=iterations, walls=walls, signature=signature)


def warm_up(seed: int) -> None:
    """Search each position shape once, small, to take first-call costs out of the timings."""
    for make_position in (_new_nim, _new_tictactoe):
        evaluator = SeededEvaluator(seed)
        engine: MCTSEngine[Any, Any, Any] = MCTSEngine(evaluator=evaluator, iterations=50)
        engine.select_plies_for_training([make_position()])


def _new_nim() -> GamePosition[Any]:
    return NimPosition(pile=NIM_PILE)


def _new_tictactoe() -> GamePosition[Any]:
    return TicTacToePosition.new_game()


def cells() -> list[Cell]:
    return [
        Cell(name=name, fleet=fleet, make_position=make_position)
        for name, make_position in (("narrow (nim)", _new_nim), ("wide (tictactoe)", _new_tictactoe))
        for fleet in FLEET_SIZES
    ]


def report(measurements: Sequence[Measurement], iterations: int, seed: int) -> None:
    print(f"\nMCTS search benchmark - iterations={iterations} seed={seed}")
    print(f"{'cell':<18} {'fleet':>5} {'repeats':>7} {'wall (s)':>10} "
          f"{'median':>10} {'spread':>7} {'iters/s':>12}  signature")
    print("-" * 92)
    for measurement in measurements:
        print(
            f"{measurement.cell.name:<18} {measurement.cell.fleet:>5} "
            f"{len(measurement.walls):>7} {measurement.best_wall:>10.4f} "
            f"{measurement.median_wall:>10.4f} {measurement.spread_pct:>6.1f}% "
            f"{measurement.tree_iterations_per_second:>12,.0f}  {measurement.signature}"
        )
    print("\nwall is the fastest repeat; iters/s is iterations x fleet / wall.")
    print("signature must not change across the steps of this story.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"search budget per tree (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed for the evaluator and ply choice (default: {DEFAULT_SEED})")
    parser.add_argument("--repeats", type=int, default=None,
                        help="repeats per cell (default: 7 at fleet 1, 3 at fleet 64)")
    args = parser.parse_args()

    warm_up(args.seed)

    measurements = [
        measure(
            cell,
            iterations=args.iterations,
            seed=args.seed,
            repeats=args.repeats if args.repeats is not None else REPEATS_BY_FLEET[cell.fleet],
        )
        for cell in cells()
    ]
    report(measurements, iterations=args.iterations, seed=args.seed)


if __name__ == "__main__":
    main()
