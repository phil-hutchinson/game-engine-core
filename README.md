# game-engine-core

A game-agnostic Python engine framework for building board and turn-based games with pluggable AI players. Provides the core abstractions for game state, player protocols, and search-based AI — including a Monte Carlo Tree Search (MCTS) engine with PUCT selection and a policy/value head interface ready for neural network integration.

> **Alpha.** The API is not stable. Breaking changes — to protocols, models, and engine behaviour — land in any release, including patch-level version bumps, and are not signalled by the version number. Pin an exact commit if you need stability.

## What's in the box

| Package | Purpose |
|---|---|
| `game_engine_core.protocols` | Abstract interfaces: `GamePosition`, `GamePly`, `GameEngine`, `Player`, `PositionEvaluator`, `GameUI`, `GameLogging` |
| `game_engine_core.game` | `StandardGame` — the main game loop wiring players, engine, and UI together; `BatchPositionProcessor` — the batch seam for position operations |
| `game_engine_core.engines` | `MCTSEngine` (PUCT selection, configurable iterations, retains its search tree across plies for play, searches N games in lockstep for training), `RandomEngine` |
| `game_engine_core.players` | `AIPlayer`, `HumanPlayer` |
| `game_engine_core.evaluators` | `NullEvaluator` — uniform prior, used as a baseline |
| `game_engine_core.models` | `GameResult`, `PositionEvaluation` (value + policy) |
| `game_engine_core.tournament` | `Tournament` — round-robin runner for any set of players, with standings/cross-table aggregation and report writing |
| `game_engine_learning` | Optional. `NeuralNetworkEvaluator` base class, self-play data collection, training loop, and checkpoint file helpers. Requires PyTorch. |

## Quick start

```python
from game_engine_core.game.standard_game import StandardGame
from game_engine_core.engines.mcts_engine import MCTSEngine
from game_engine_core.evaluators.null_evaluator import NullEvaluator
from game_engine_core.players.ai_player import AIPlayer

engine = MCTSEngine(evaluator=NullEvaluator(), iterations=200_000)
game = StandardGame(
    initial_position=MyGamePosition.new_game(),
    players={
        1: AIPlayer(engine=engine, name="AI (X)"),
        -1: AIPlayer(engine=engine, name="AI (O)"),
    },
    game_logging=MyGameLogging(),  # board snapshots + ply annotations for the game record
    game_ui=MyGameUI(),  # interactive display; omit for headless play
)
result = game.run()
```

See [`examples/tictactoe`](examples/tictactoe) for a complete working implementation.

## Implementing a game

1. Subclass `GamePosition` — represent your board state, enumerate legal moves, and report the outcome (with a reason) once the game ends.
2. Subclass `GamePly` — represent a single move.
3. Implement `GameLogging` — a text board rendering and a per-ply log annotation (`str(ply)` is a valid trivial annotation).
4. Optionally implement `PositionEvaluator` — plug in a heuristic or neural network to guide MCTS. Its one method, `evaluate_positions`, takes a sequence of positions and returns one `PositionEvaluation` per position, index-aligned.
5. Wire it together with `StandardGame`.

The engine never touches game-specific logic. Everything game-specific lives behind the `GamePosition` and `PositionEvaluator` protocols.

`MCTSEngine` and `SelfPlayCollector` reach positions only through a `BatchPositionProcessor` (constructor argument `batch_ops`, defaulting to a base instance whose `outcomes`/`legal_plies`/`apply_plies` loop the corresponding `GamePosition` member). It exists so a game whose positions can be scored several at once — vectorised legality, a single batched call into an external engine — can override just the method that benefits and inherit the rest; this is the expected steady state, not a fallback for simple games. `GamePosition` keeps its own scalar `outcome`/`legal_plies`/`apply_ply`, which is what lets a position implementation delegate one of them to a batch-of-one call into a `BatchPositionProcessor` it holds — but only into a method that processor actually overrides. Delegating into the inherited loop instead calls straight back into the same scalar property and recurses until the stack overflows; `BatchPositionProcessor.overridden_methods()` reports which methods a subclass overrode, so a delegating position can assert the assumption at construction rather than discovering it mid-search.

The examples ship with their own pytest suites ([`examples/tictactoe/tests`](examples/tictactoe/tests), [`examples/tictactoe_learning/tests`](examples/tictactoe_learning/tests)) that double as a model for testing your own game implementation — position legality and outcome checks, evaluator sanity tests, and an engine-vs-position integration test.

## MCTS and neural network support

`MCTSEngine` uses PUCT selection and full node expansion: each search iteration descends to a leaf, evaluates it exactly once, and attaches a child for every legal ply seeded with its prior. It accepts any `PositionEvaluator` implementation, so a neural network policy/value head can be dropped in without changing the search logic. The evaluator returns a `PositionEvaluation` with a scalar value (from the current player's perspective) and a policy dict mapping plies to prior probabilities.

The engine has two entry points. `select_ply` is the play surface — one game, the `GameEngine` protocol method, retaining its tree across plies. `select_plies_for_training` is the training surface: hand it the current position of N independent games and it searches all N in lockstep, returning one `(ply, visit_distribution)` pair per game, index-aligned by slot. Because the games advance one iteration together, every iteration gathers one leaf per game into a single `evaluate_positions` call instead of N calls of width one — which is what lets a batched evaluator saturate a GPU. A single game is simply the fleet at N = 1, and both entry points run the same iteration.

The policy is **required** and must cover every legal ply of the position, because expansion needs a prior for each child it creates. An evaluator with no policy head should return a uniform distribution over the legal plies, as `NullEvaluator` does.

Within a game, `MCTSEngine` retains its search tree between plies: `StandardGame` reports every applied ply back to the engine via `observe_ply`, and the engine re-roots onto the corresponding child instead of rebuilding from scratch, carrying forward whatever the search accumulated below that ply — visit counts, cached evaluations, and priors. A ply the search never descended into carries forward its prior alone, since expansion created the child but nothing below it exists yet. This is most valuable with a neural network evaluator, where re-evaluating positions is the dominant cost. Retention is per-game — `StandardGame` resets it at the start of each game, so reusing a player/engine across games (as `Tournament` does) never leaks a tree between games. Retention applies to `select_ply` only: `select_plies_for_training` builds bare roots on every call and retains nothing, so each ply's search is independent.

`game_engine_learning` provides self-play loops, training infrastructure, and a `NeuralNetworkEvaluator` base class. Subclass `NeuralNetworkEvaluator` and implement `encode_positions` (positions → one stacked input tensor) and `decode_policies` (policy logits + positions → one probability distribution per position) — the base class stacks the batch, runs a single forward pass in eval mode, and assembles the `PositionEvaluation`s. See [`examples/tictactoe_learning`](examples/tictactoe_learning) for a complete example.

## Tournaments

`game_engine_core.tournament` plays a round-robin between any `Player` implementations (sides alternate within each pairing) and reports standings, a cross-table, and per-game JSON logs. Each game's starting position comes from a factory called with the two participants in side order, giving games whose opening position depends on per-player state a place to build it — factories that don't need the players just ignore the arguments. Its main use is measuring training progress: save checkpoints during training (`train.py --checkpoint-every N` in the learning example), then enter each checkpoint as a player — the standings show whether later checkpoints actually beat earlier ones. See [`examples/tictactoe_learning/tournament.py`](examples/tictactoe_learning/tournament.py).

## Requirements

- Python 3.12+
- PyTorch 2.0+ (only required for `game_engine_learning`)

## Installation

The package is not yet published to PyPI; install it directly from GitHub:

```bash
pip install "git+https://github.com/phil-hutchinson/game-engine-core.git"
```

To include the optional learning module (pulls in PyTorch):

```bash
pip install "game-engine-core[learning] @ git+https://github.com/phil-hutchinson/game-engine-core.git"
```

Or from a local clone of the repository: `pip install .` (or `pip install ".[learning]"`).

## Contributing

Contributor setup is separate from the installation above — the repo ships a VS Code Dev Container that provisions the full development environment automatically. See [CONTRIBUTING.md](CONTRIBUTING.md).
