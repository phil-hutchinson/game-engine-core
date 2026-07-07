# Story: Add Tournament Runner (issue #8)

## Goal

Add a reusable, game-agnostic **tournament runner** to `game_engine_core` that plays a round-robin among an arbitrary set of players and reports standings, a cross-table, and per-game logs. Pair it with checkpointing in the training flow so a network trained for progressively more iterations can be entered as multiple distinct players. Validate end-to-end with TicTacToe: a tournament between training checkpoints (plus any extra players added at tournament creation) should make ML progress visible — later checkpoints beating earlier ones.

## Motivation

The repo can train a network via self-play, but has no way to answer "is it actually getting stronger?" beyond watching loss curves. Head-to-head results between checkpoints of the same network are a direct, loss-agnostic strength signal (a scaled-down version of the evaluation ladders used in AlphaZero-style pipelines).

Everything a tournament needs already exists behind stable seams: `StandardGame` plays one headless game between two `Player`s and returns a `GameResult`, and `Player` is game-agnostic. A round-robin scheduler, result aggregation, and reporting are pure plumbing that every game repo would otherwise reimplement — so they belong in `game_engine_core` (the tournament itself has no ML dependency; any `Player` can enter).

## Scope

### 1. `game_engine_core` tournament support

#### `Tournament` (round-robin runner)

Runs a round-robin among N named players:

- Every pair of players meets in `games_per_pairing` games, alternating which player takes side `1` (moves first), so first-move advantage cancels out.
- Players are supplied at tournament creation as a list — checkpoints, baselines, and any other `Player` implementations mix freely.
- Games run headless via `StandardGame` (no board rendering; `AIPlayer` already defaults `render_before_ply=False`). A `GameUI` is still supplied — its `text_board` feeds the game logs.
- Parameterised with a position factory; no game-specific logic inside.

#### Result models

Dataclasses capturing the outcome of a completed tournament:

- per-game record: the two players, who had side 1, and the `GameResult`
- standings: wins/draws/losses and points (1 / 0.5 / 0) per player, sorted
- cross-table: aggregate score of each player vs. each other player

#### Tournament reporting

Writes a completed tournament to a **timestamped output folder** (one folder per tournament run; path supplied by the caller):

- `standings.md` — standings table and cross-table
- `games/` — one JSON file per game: player names, side assignment, outcome, and the ply-by-ply log (plies are already strings, so `GameResult.game_log` serialises naturally)

A summary (standings + cross-table) is also printed to the console.

### 2. Checkpointing during training

Extend the training flow so weights are saved every `n` iterations (e.g. `checkpoint-00005.pt`, `checkpoint-00010.pt`, …) in addition to the final weights. Training and tournament stay decoupled: checkpoints are files on disk, and a tournament can be re-run with different settings without retraining.

The reusable part (naming/saving checkpoints alongside a training loop) should live in `game_engine_learning` if it carries its weight; the exact split between package and example is an implementation-plan decision.

### 3. TicTacToe tournament example

Extends `examples/tictactoe_learning/`:

- **`train.py`** gains a `--checkpoint-every n` option that writes checkpoint weight files to the (gitignored) weights directory.
- **`tournament.py`** — new entry point that:
  - loads each checkpoint into a `TicTacToeNNEvaluator` + `MCTSEngine` + `AIPlayer`, named after its training iteration (e.g. `nn-iter-05`)
  - demonstrates adding extra players at tournament creation — e.g. a `RandomEngine` player and the heuristic-evaluator MCTS player from `examples/tictactoe/` — as fixed reference points
  - runs the round-robin and writes results to a timestamped folder under a gitignored tournaments directory (derived artifacts, like weights)

**Definition of done for the example:** after training with checkpoints enabled, running `tournament.py` produces a standings/cross-table where strength ordering is inspectable — ideally later checkpoints outscore earlier ones and random loses to everyone.

## Out of Scope

- Tournament formats other than round-robin (Swiss, elimination, gauntlet)
- Elo or other rating computations — standings and cross-table only
- Parallel game execution — games run sequentially
- Games with other than two players (`StandardGame` is two-player; the tournament inherits that)
- Any game other than TicTacToe for validation
