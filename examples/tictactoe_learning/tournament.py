"""Round-robin tournament between training checkpoints and reference players.

Each checkpoint saved by train.py (--checkpoint-every) enters as its own
player, so the cross-table directly shows whether more training produced a
stronger network. Checkpoints are read from a single training run — the most
recent run folder by default, or one named via --checkpoints-dir — so networks
from different training runs never mix. Two reference players anchor the
scale: random (any checkpoint should beat it) and the heuristic-evaluator MCTS
from the plain tictactoe example. Results land in a timestamped folder under
examples/tictactoe_learning/tournaments/ (gitignored — derived artifacts,
like weights).

Usage:
    python -m examples.tictactoe_learning.tournament
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from examples.tictactoe.tictactoe_heuristic_evaluator import TicTacToeHeuristicEvaluator
from examples.tictactoe.tictactoe_ply import TicTacToePly
from examples.tictactoe.tictactoe_position import TicTacToePosition
from examples.tictactoe.tictactoe_ui import TicTacToeUI
from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.engines.random_engine import RandomEngine
from game_engine_core.players.ai_player import AIPlayer
from game_engine_core.protocols.player import Player
from game_engine_core.tournament.reporting import write_tournament_report
from game_engine_core.tournament.tournament import Tournament
from game_engine_learning.checkpoints import (
    Checkpoint,
    discover_checkpoints,
    latest_run_directory,
)

from .tictactoe_mlp import TicTacToeMLP
from .tictactoe_nn_evaluator import TicTacToeNNEvaluator
from .train import RUNS_DIR

TOURNAMENTS_DIR = Path(__file__).parent / "tournaments"


def _checkpoint_player(
    checkpoint: Checkpoint, mcts_iterations: int, temperature: float
) -> AIPlayer[TicTacToePly, TicTacToePosition]:
    model = TicTacToeMLP()
    model.load_state_dict(torch.load(checkpoint.path, weights_only=True))
    engine: MCTSEngine[TicTacToePly, TicTacToePosition, TicTacToeNNEvaluator] = MCTSEngine(
        evaluator=TicTacToeNNEvaluator(model=model),
        iterations=mcts_iterations,
        temperature=temperature,
    )
    return AIPlayer(engine=engine, name=f"nn-iter-{checkpoint.iteration:02d}")


def _reference_players(
    mcts_iterations: int, temperature: float
) -> list[Player[TicTacToePly, TicTacToePosition]]:
    """Extra (non-checkpoint) entrants — any Player can be added here."""
    random_engine: RandomEngine[TicTacToePly, TicTacToePosition] = RandomEngine()
    heuristic_engine: MCTSEngine[
        TicTacToePly, TicTacToePosition, TicTacToeHeuristicEvaluator
    ] = MCTSEngine(
        evaluator=TicTacToeHeuristicEvaluator(),
        iterations=mcts_iterations,
        temperature=temperature,
    )
    return [
        AIPlayer(engine=random_engine, name="random"),
        AIPlayer(engine=heuristic_engine, name="heuristic-mcts"),
    ]


def main(
    games_per_pairing: int = 2,
    mcts_iterations: int = 100,
    temperature: float = 0.0,
    checkpoints_dir: Path | None = None,
) -> None:
    if checkpoints_dir is None:
        checkpoints_dir = latest_run_directory(RUNS_DIR)
        if checkpoints_dir is None:
            raise SystemExit(
                f"No training runs found under {RUNS_DIR}.\n"
                "Run `python -m examples.tictactoe_learning.train --checkpoint-every N` first."
            )
    checkpoints = discover_checkpoints(checkpoints_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {checkpoints_dir}.")
    print(f"Using checkpoints from {checkpoints_dir}")

    players = [
        _checkpoint_player(checkpoint, mcts_iterations, temperature)
        for checkpoint in checkpoints
    ] + _reference_players(mcts_iterations, temperature)

    tournament: Tournament[TicTacToePly, TicTacToePosition] = Tournament(
        players=players,
        position_factory=lambda p1, p2: TicTacToePosition.new_game(),
        game_ui=TicTacToeUI(),
        games_per_pairing=games_per_pairing,
    )
    print(f"Running round-robin: {len(players)} players, {games_per_pairing} games per pairing...")
    result = tournament.run()

    output_dir = TOURNAMENTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    write_tournament_report(result, output_dir, notes=[f"Checkpoints: {checkpoints_dir}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Round-robin tournament between training checkpoints and reference players."
    )
    parser.add_argument(
        "--games-per-pairing",
        type=int,
        default=2,
        help="Games per pairing (sides alternate). With temperature 0 play is "
        "deterministic, so more than 2 only adds variety if --temperature > 0.",
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=100,
        help="MCTS iterations per ply for every MCTS-backed player.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ply-selection temperature for MCTS-backed players. 0.0 (default) "
        "always picks the top move; small values (~0.3) add variety.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Training run folder to load checkpoints from. Defaults to the most "
        "recent run under weights/runs/.",
    )
    args = parser.parse_args()
    main(
        games_per_pairing=args.games_per_pairing,
        mcts_iterations=args.mcts_iterations,
        temperature=args.temperature,
        checkpoints_dir=args.checkpoints_dir,
    )
