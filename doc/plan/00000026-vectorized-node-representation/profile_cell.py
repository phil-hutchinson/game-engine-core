"""cProfile one benchmark cell (issue #26, Step 6).

``benchmark.py`` says how fast a cell is. This says where the time went, which
is what attributes a regression to a particular line rather than to a step. It
is the source of the "where the time goes" table in ``benchmark.md`` — and of
the slots-scored-per-iteration figures, which are the ``ncalls`` on
``child_puct_values`` divided by iterations x fleet.

    python doc/plan/00000026-vectorized-node-representation/profile_cell.py wide 64
    python doc/plan/00000026-vectorized-node-representation/profile_cell.py narrow 64

Read the numbers as shares, not as absolute times: cProfile's per-call overhead
is large relative to the array operations being measured here, so it inflates
everything, and it inflates the most-called functions most.
"""

from __future__ import annotations

import cProfile
import importlib.util
import pstats
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from game_engine_core.engines.mcts_engine import MCTSEngine  # noqa: E402

_BENCH_PATH = Path(__file__).resolve().parent / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_benchmark", _BENCH_PATH)
assert _spec is not None and _spec.loader is not None
_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["_benchmark"] = _benchmark  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(_benchmark)

ROWS = 14


def main() -> None:
    cell_name = sys.argv[1] if len(sys.argv) > 1 else "wide"
    fleet = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    make_position = _benchmark._new_tictactoe if cell_name == "wide" else _benchmark._new_nim
    positions = [make_position() for _ in range(fleet)]

    random.seed(_benchmark.DEFAULT_SEED)
    engine: MCTSEngine[Any, Any, Any] = MCTSEngine(
        evaluator=_benchmark.SeededEvaluator(_benchmark.DEFAULT_SEED),
        iterations=_benchmark.DEFAULT_ITERATIONS,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    engine.select_plies_for_training(positions)
    profiler.disable()

    tree_iterations = _benchmark.DEFAULT_ITERATIONS * fleet
    print(f"\n=== {cell_name} cell, fleet {fleet}, {tree_iterations:,} tree-iterations ===")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(ROWS)


if __name__ == "__main__":
    main()
