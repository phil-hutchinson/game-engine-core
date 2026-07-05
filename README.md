# game-engine-core

A game-agnostic Python engine framework for building board and turn-based games with pluggable AI players. Provides the core abstractions for game state, player protocols, and search-based AI — including a Monte Carlo Tree Search (MCTS) engine with PUCT selection and a policy/value head interface ready for neural network integration.

## What's in the box

| Package | Purpose |
|---|---|
| `game_engine_core.protocols` | Abstract interfaces: `GamePosition`, `GamePly`, `GameEngine`, `Player`, `PositionEvaluator`, `GameUI` |
| `game_engine_core.game` | `StandardGame` — the main game loop wiring players, engine, and UI together |
| `game_engine_core.engines` | `MCTSEngine` (PUCT selection, configurable iterations), `RandomEngine` |
| `game_engine_core.players` | `AIPlayer`, `HumanPlayer` |
| `game_engine_core.evaluators` | `NullEvaluator` — uniform prior, used as a baseline |
| `game_engine_core.models` | `GameResult`, `PositionEvaluation` (value + policy) |
| `game_engine_learning` | Optional. `NeuralNetworkEvaluator` base class, self-play data collection, and training loop. Requires PyTorch. |

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
    game_ui=MyGameUI(),
)
result = game.run()
```

See [`examples/tictactoe`](examples/tictactoe) for a complete working implementation.

## Implementing a game

1. Subclass `GamePosition` — represent your board state and enumerate legal moves.
2. Subclass `GamePly` — represent a single move.
3. Optionally subclass `PositionEvaluator` — plug in a heuristic or neural network to guide MCTS.
4. Wire it together with `StandardGame`.

The engine never touches game-specific logic. Everything game-specific lives behind the `GamePosition` and `PositionEvaluator` protocols.

## MCTS and neural network support

`MCTSEngine` uses PUCT selection (the same formula as AlphaZero). It accepts any `PositionEvaluator` implementation, so a neural network policy/value head can be dropped in without changing the search logic. The evaluator returns a `PositionEvaluation` with a scalar value (from the current player's perspective) and a policy dict mapping moves to prior probabilities.

`game_engine_learning` provides self-play loops, training infrastructure, and a `NeuralNetworkEvaluator` base class. Subclass `NeuralNetworkEvaluator` and implement `encode_position` and `decode_policy` — the base class handles the forward pass and assembles the `PositionEvaluation`. See [`examples/tictactoe_learning`](examples/tictactoe_learning) for a complete example.

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
