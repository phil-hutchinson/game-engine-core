"""Fixed-seed self-play ply sequences (issue #26, Step 6).

The story's non-goal is that search results do not move. ``benchmark.py``'s
result signatures are most of that argument, but they only exercise
``select_plies_for_training``, which builds bare roots and retains nothing —
so they cannot catch a re-rooting regression, and Step 4 changed ``observe_ply``.

This plays a full game to completion through the *play* path instead:
``select_ply`` then ``observe_ply``, one engine taking both sides at
temperature 0, so the sequence is a pure function of the seed. Run it before and
after the branch and diff the output.

    python doc/plan/00000026-vectorized-node-representation/ply_sequences.py

For the before-numbers, see the worktree procedure in ``seam_call_counts.py`` —
the editable install's import hook has to be disabled or the baseline run will
silently exercise the working tree's engine.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tictactoe.tictactoe_position import TicTacToePosition  # noqa: E402
from game_engine_core.engines.mcts_engine import MCTSEngine  # noqa: E402
from game_engine_core.protocols.game_position import GamePosition  # noqa: E402
from tests.core.nim_fixture import NimPosition  # noqa: E402

# SeededEvaluator lives in benchmark.py, which is not importable as a module.
# Sharing it means both sides of a before/after comparison draw from an
# identical stream.
_BENCH_PATH = Path(__file__).resolve().parent / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_benchmark", _BENCH_PATH)
assert _spec is not None and _spec.loader is not None
_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["_benchmark"] = _benchmark  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(_benchmark)

SEED = 20260730
ITERATIONS = 200
NIM_PILE = 21


def play(name: str, position: GamePosition[Any]) -> None:
    random.seed(SEED)
    engine: MCTSEngine[Any, Any, Any] = MCTSEngine(
        evaluator=_benchmark.SeededEvaluator(SEED), iterations=ITERATIONS, temperature=0.0
    )
    engine.reset()

    plies: list[str] = []
    while position.outcome is None:
        ply = engine.select_ply(position)
        next_position = position.apply_ply(ply)
        engine.observe_ply(position, ply, next_position)
        plies.append(str(ply))
        position = next_position

    print(f"{name}: outcome={position.outcome} reason={position.outcome_reason}")
    print(f"{name}: {' '.join(plies)}")


def main() -> None:
    play("tictactoe", TicTacToePosition.new_game())
    play("nim", NimPosition(pile=NIM_PILE))


if __name__ == "__main__":
    main()
