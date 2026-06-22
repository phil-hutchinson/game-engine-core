# Story: Add ML Learning Package (issue #1)

## Goal

Create a `game_engine_learning` sibling package that provides reusable self-play training infrastructure. Any game repo can plug in a neural network evaluator and get the full training loop for free, without duplicating the plumbing. Validate the package end-to-end with a TicTacToe learning example.

## Motivation

`game_engine_core` already defines `PositionEvaluator` as the seam between game logic and evaluation. Everything behind that seam that is game-agnostic — running self-play, collecting training samples, gradient descent — can live in a shared package. The alternative is each game repo reimplementing the same training scaffolding.

`game_engine_learning` is a separate package (not a subpackage of `game_engine_core`) so that the PyTorch dependency is opt-in.

## Scope

### 1. `game_engine_learning` package

#### `NeuralNetworkEvaluator` (abstract base class)

The main extension point. Wraps a PyTorch model and implements `PositionEvaluator`. Game repos subclass this and provide:

- `encode_position(position) -> Tensor` — converts a `GamePosition` to a model input tensor
- `decode_policy(output_tensor, legal_plies) -> dict[str, float]` — masks illegal moves and normalises logits to probabilities

The base class handles the forward pass and assembles the returned `PositionEvaluation`.

#### `TrainingSample`

Dataclass holding one labelled example from self-play:

- `encoded_position: Tensor`
- `target_value: float`
- `target_policy: dict[str, float]`

#### `SelfPlayCollector`

Runs N complete games between two MCTS agents and accumulates `TrainingSample` instances. Parameterised with an engine factory and a game factory; no game-specific logic inside.

#### `TrainingLoop`

Gradient descent over a collection of `TrainingSample`s. Optimises value and policy heads jointly. Parameterised with a model, an optimizer, and loss functions; no game-specific logic inside.

### 2. `MCTSEngine` temperature fix

The existing `TEMPERATURE` class constant has a TODO marking it as incomplete. Self-play requires temperature > 0 to generate diverse training data. This story replaces the class constant with an `__init__` parameter (`temperature: float = 0.0`).

### 3. `examples/tictactoe_learning/` example

A new example that imports game types from `examples/tictactoe/` and demonstrates end-to-end training:

- **`TicTacToeMLP`** — a small PyTorch MLP with value and policy heads (inputs: 9 board squares; outputs: scalar value + 9-element policy logits)
- **`TicTacToeNNEvaluator`** — subclasses `NeuralNetworkEvaluator`, implements `encode_position` and `decode_policy`
- **`train.py`** — runs self-play and training loop iterations; saves model weights to `examples/tictactoe_learning/weights/model.pt` (gitignored — weights are a derived artifact, not source)
- **`main.py`** — loads weights from the above path and plays a game (human vs. trained AI, or trained AI vs. another agent); fails with a clear message if weights are not found ("run train.py first")

## Out of Scope

- Replay buffer with prioritised sampling (uniform sampling is sufficient for TicTacToe)
- Distributed self-play or async data collection
- Network architecture search or hyperparameter tuning infrastructure
- Any game other than TicTacToe for validation

