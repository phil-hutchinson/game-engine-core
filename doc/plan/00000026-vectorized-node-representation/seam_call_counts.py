"""Count every call the engine makes across the seam (issue #26, Step 6).

The timings in ``benchmark.md`` are what this story cost. This is what it
bought, and it does not show up in a wall clock here because both example games
construct a position almost for free — so the fixtures show all of the cost and
none of the benefit.

Wraps the evaluator and a ``BatchPositionProcessor`` in counters and runs one
benchmark cell, reporting per seam method how many calls were made and the total
and maximum width of those calls. Width is the number that matters: it is what
an implementation batching ``apply_plies`` onto a device would be sizing for.

Run it against the branch tip:

    python doc/plan/00000026-vectorized-node-representation/seam_call_counts.py wide 64

Getting the before-numbers needs a little care, because the editable install
registers an import hook that resolves ``game_engine_core`` to the working tree
no matter what ``sys.path`` says. Check the baseline out into a worktree, copy
this file in, and disable the hook by skipping user site-packages:

    git worktree add /tmp/baseline 5dcedf9
    cp doc/plan/00000026-*/seam_call_counts.py /tmp/baseline/doc/plan/00000026-*/
    cd /tmp/baseline && PYTHONNOUSERSITE=1 python doc/plan/00000026-*/seam_call_counts.py wide 64

Confirm it worked before believing the output: the baseline commit has no numpy
import in ``mcts_engine.py``, so ``grep -c 'np\\.' `` on the resolved module is
the check.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from game_engine_core.engines.mcts_engine import MCTSEngine  # noqa: E402
from game_engine_core.game.batch_position_processor import (  # noqa: E402
    BatchPositionProcessor,
)
from game_engine_core.models.position_evaluation import PositionEvaluation  # noqa: E402

# The cell definitions and the evaluator live in benchmark.py, which sits beside
# this file and is not importable as a module.
_BENCH_PATH = Path(__file__).resolve().parent / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_benchmark", _BENCH_PATH)
assert _spec is not None and _spec.loader is not None
_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["_benchmark"] = _benchmark  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(_benchmark)

CALLS: dict[str, list[int]] = defaultdict(list)


class CountingProcessor(BatchPositionProcessor[Any, Any]):
    """Records the width of every seam call, then defers to the default loop."""

    def outcomes(self, positions: Sequence[Any]) -> Sequence[Literal[1, 0, -1] | None]:
        CALLS["outcomes"].append(len(positions))
        return super().outcomes(positions)

    def legal_plies(self, positions: Sequence[Any]) -> Sequence[Sequence[Any]]:
        CALLS["legal_plies"].append(len(positions))
        return super().legal_plies(positions)

    def apply_plies(self, positions: Sequence[Any], plies: Sequence[Any]) -> Sequence[Any]:
        CALLS["apply_plies"].append(len(positions))
        return super().apply_plies(positions, plies)


class CountingEvaluator:
    def __init__(self, inner: Any):
        self._inner = inner

    def evaluate_positions(self, positions: Sequence[Any]) -> Sequence[PositionEvaluation]:
        CALLS["evaluate_positions"].append(len(positions))
        return self._inner.evaluate_positions(positions)


def main() -> None:
    cell_name = sys.argv[1] if len(sys.argv) > 1 else "wide"
    fleet = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    make_position = _benchmark._new_tictactoe if cell_name == "wide" else _benchmark._new_nim
    positions = [make_position() for _ in range(fleet)]

    random.seed(_benchmark.DEFAULT_SEED)
    engine: MCTSEngine[Any, Any, Any] = MCTSEngine(
        evaluator=CountingEvaluator(_benchmark.SeededEvaluator(_benchmark.DEFAULT_SEED)),
        iterations=_benchmark.DEFAULT_ITERATIONS,
        batch_ops=CountingProcessor(),
    )
    engine.select_plies_for_training(positions)

    print(f"\n{cell_name} cell, fleet {fleet}, {_benchmark.DEFAULT_ITERATIONS} iterations")
    print(f"{'seam method':<22} {'calls':>8} {'total width':>13} {'max width':>10} {'mean width':>11}")
    print("-" * 68)
    for name in ("evaluate_positions", "outcomes", "legal_plies", "apply_plies"):
        widths = CALLS[name]
        if not widths:
            print(f"{name:<22} {0:>8} {'-':>13} {'-':>10} {'-':>11}")
            continue
        print(f"{name:<22} {len(widths):>8} {sum(widths):>13,} {max(widths):>10} "
              f"{sum(widths) / len(widths):>11.1f}")


if __name__ == "__main__":
    main()
