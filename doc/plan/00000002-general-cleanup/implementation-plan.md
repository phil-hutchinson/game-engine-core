# Implementation Plan — General Cleanup

This plan addresses the peer-review findings from [story-peer-review.md](story-peer-review.md) that are small, low-risk, and self-contained enough to fix within this story: **#1, #5, #6, #7, #8, #9, #10**.

The type-level fixes (#5, #6) are verified via Pylance in the IDE, which embeds pyright — CLI pyright adoption is deferred (see below), so no new tooling is installed in this story.

## Out of scope — deferred to their own stories

These findings are marked **Requires Story** in the peer review:

- **#2 (MCTS root never evaluated)** — changes engine search behaviour and reopens the question of the learning example's play-time iteration budget; needs its own design/verification pass.
- **#3 (player-alternation assumption)** — requires a design decision between documenting the constraint as a protocol precondition and making the framework genuinely alternation-agnostic.
- **#4 (no test suite)** — a body of work in its own right; the pytest tooling recommendation goes with it.

Also deferred: the peer review's **pyright** and **ruff** tooling recommendations. CLI pyright requires a node runtime, which is unwanted on the host machine; both tools are planned to land in a future story that containerizes the development environment. Until then, Pylance provides interim type checking in the IDE.

## Steps

### Step 1 — Fix packaging: add the missing `game` package initializer (finding #1)

Add an empty `__init__.py` to `game_engine_core/game/` so setuptools discovers the subpackage and built distributions include `StandardGame`.

Depends on: nothing (ordered first because it is the only Critical finding).

Verification (improvised): From the project venv, run a one-liner invoking setuptools package discovery with the same include patterns as `pyproject.toml` and confirm `game_engine_core.game` now appears in the output. Then run `python -m examples.tictactoe --p1 random --p2 random` to confirm the package still imports and a game runs to completion.

### Step 2 — Fix the outcome type violation in `StandardGame` (finding #5)

Narrow the absolute-outcome computation in `StandardGame.run` so it satisfies `GameResult.outcome`'s `Literal[1, 0, -1]` declaration, with a brief comment noting the runtime math is unchanged.

Depends on: Step 1 (the game package should be properly importable before further edits to it).

Verification (manual): Open `standard_game.py` in the IDE and confirm Pylance no longer reports a type error on the `GameResult` construction, with no new diagnostics introduced. Run `python -m examples.tictactoe --p1 random --p2 random` and confirm the result line ("Player X wins!" / "Player O wins!" / "It's a draw!") still reports correctly.

### Step 3 — Fix the `Player` protocol type bound (finding #6)

Change `Player`'s position type-parameter bound to match the `GamePosition[Any]` convention used everywhere else in the codebase, so concrete instantiations like a TicTacToe-typed `Player` satisfy the bound.

Depends on: nothing (independent of Steps 1–2; ordered here to keep the two type-level fixes adjacent).

Verification (improvised): In a temporary scratch script (not committed), annotate a variable with a fully concrete `Player` instantiation using the TicTacToe types and assign a `HumanPlayer`/`AIPlayer` to it; confirm Pylance reports no error, and that it *does* report a bound violation before the fix is applied. Note this scratch check is improvised — it becomes reproducible outside the IDE when the containerization story adds CLI pyright, and a real regression test when the finding-#4 test-suite story lands.

### Step 4 — Modernise the `Callable` import in `SelfPlayCollector` (finding #10)

Move the `Callable` import in `self_play_collector.py` from `typing` to `collections.abc`, matching `training_loop.py` and the CONTRIBUTING.md convention.

Depends on: nothing (independent of Steps 1–3).

Verification (manual): Run `python -m examples.tictactoe_learning.selfplay` and confirm the self-play diagnostic completes and prints its summary statistics (total samples, value distribution, policy entropy), demonstrating the module imports and functions correctly.

### Step 5 — Document the `str(ply)` uniqueness contract on `GamePly` (finding #9)

Extend the `GamePly` protocol docstring to state that `str(ply)` must be unique among a position's legal plies, since the framework uses it as the identity key in policy dictionaries, visit distributions, and training targets.

Depends on: nothing (documentation-only; ordered here to keep code-behaviour changes in Steps 6–7 last).

Verification (manual): Documentation-only change — review the rendered docstring for accuracy and confirm Pylance shows no new diagnostics. No runtime behaviour to exercise.

### Step 6 — Weight epoch losses by batch size in `TrainingLoop` (finding #7)

Change the per-epoch loss aggregation so each batch contributes in proportion to its sample count, making reported epoch losses exact when the final batch is ragged.

Depends on: Step 4 (same package; keeps `game_engine_learning` edits sequential and separately verifiable).

Verification (improvised): Run a short training session with a sample count that is not a multiple of the batch size (e.g. `python -m examples.tictactoe_learning.train --iterations 1 --games 5`) and confirm losses report as sensible finite values. Then, in a temporary REPL session, train one epoch over a tiny hand-sized sample set with a batch size that forces a ragged final batch, and confirm the reported epoch loss equals the sample-weighted mean of the per-batch losses. Note this REPL check is improvised — it becomes a real unit test when the finding-#4 test-suite story lands.

### Step 7 — Randomise MCTS expansion order (finding #8)

Remove the deterministic end-first expansion bias in `MCTSEngine` by randomising the order in which unexplored plies are expanded.

Depends on: Steps 1–2 (core-package changes verified stable first). Ordered last because it is the only step in this story that intentionally alters engine behaviour (expansion order only; selection and backpropagation are untouched).

Verification (manual): Run `python -m examples.tictactoe --p1 heuristic --p2 random` a few times and confirm the heuristic player still plays sensibly and wins consistently. Then run `python -m examples.tictactoe_learning.selfplay` and confirm mean policy entropy remains close to the uniform baseline (random-weight network), showing the visit distributions are still well-formed.

## After completion

Update the Status and Resolution columns for findings #1, #5, #6, #7, #8, #9, #10 in [story-peer-review.md](story-peer-review.md) as each fix lands.
