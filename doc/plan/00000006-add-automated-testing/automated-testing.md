# Automated Testing

How this repository is tested: the suite layout, the strategies behind it, and how to run everything. Written as part of the add-automated-testing story; kept accurate against the suite as built.

## Running the tests

From the repository root, inside the dev container:

```bash
pytest                 # everything: package tests + example tests
pytest tests           # package tests only (game_engine_core, game_engine_learning)
pytest tests/core      # one package's tests
pytest examples        # example tests only
```

(Test paths are configured in `pyproject.toml`, so bare `pytest` from the root finds both suites.)

pytest is pinned in the dev-container image (same policy as ruff and pyright: bump the pin deliberately, in a commit of its own) and configured in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Layout: two suites

The suite is deliberately split in two:

```
tests/                              # package tests — the library's own regression suite
    core/                           #   game_engine_core (incl. nim_fixture.py, the fixture game)
    learning/                       #   game_engine_learning (incl. nim_nn.py, a minimal torch model/evaluator)
examples/tictactoe/tests/           # example tests — also an example of how to test
examples/tictactoe_learning/tests/
```

Support modules (`nim_fixture.py`, `nim_nn.py`) carry no `test_` prefix, so pytest does not collect them; they are imported by the test modules, the learning suite reaching the fixture game via `tests.core.nim_fixture`.

**Package tests (`tests/`)** protect the library. They are self-contained: no imports from `examples/`, so the packages are tested exactly as an external consumer receives them. `tests/` is not included in the distribution (setuptools discovery includes only `game_engine_core*` and `game_engine_learning*`).

**Example tests (under `examples/`)** protect the TicTacToe examples — and, like the examples themselves, they are documentation: they show a developer integrating this library what testing a concrete game implementation looks like (protocol-contract tests for a position class, evaluator sanity checks, an engine-vs-position integration test).

## The fixture game

The core protocols (`GamePosition`, `GamePly`) need a real implementation to test engines and collectors against, and the package tests must not reach into `examples/` for one. `tests/core/` therefore defines a minimal fixture game: **subtraction Nim** — a pile of tokens, each ply removes a permitted number, taking the last token wins, players strictly alternate (the framework's current alternation assumption — see general-cleanup finding #3).

Nim was chosen over something like a micro TicTacToe because its game theory is trivial to state in a test: with takes of 1–2, a pile that is a multiple of 3 is lost for the player to move, and anything else is won. That gives MCTS tests known-correct answers ("from pile 2, taking 2 is the forced win") without hand-built board states.

The permitted takes are configurable per instance, which yields two modes:

- **Choice mode** (takes 1–2): real branching for MCTS search tests.
- **Forced-line mode** (takes of exactly 1): one legal ply per position, so a whole game is deterministic and its winner is a parity function of the starting pile. Self-play tests use this to predict every training sample's target value exactly.

## What is covered, and why

The coverage centres on the sign-convention logic identified by general-cleanup peer-review finding #4 — the code that earlier reviews could only verify by hand:

| Area | Key assertions |
|---|---|
| Fixture game | Its own contract: current-player-relative outcome, legality, alternation |
| `StandardGame` | Relative outcome at the terminal position converts to the correct absolute `GameResult.outcome` for either winner; game log records every ply |
| `RandomEngine` | Selects only legal plies; raises on a position without plies |
| `MCTSEngine` | Forced win-in-1 is found with correctly signed search values; backpropagation flips the value sign per tree level; terminal nodes evaluate to their exact outcome; visit distribution covers all legal plies, sums to 1, uniform fallback; policy priors reach children; missing policy entry raises; temperature-0 picks the most-visited ply |
| `SelfPlayCollector` | Target values alternate backwards from `-final_outcome` (the last ply was made by the *other* player than the terminal perspective); encodings pair with their steps; policy targets present; samples accumulate across games |
| `TrainingLoop` | Reported epoch loss is the sample-weighted mean over batches, exact for ragged final batches (durable form of general-cleanup finding #7's scratch verification, using a zero learning rate to freeze the model); loss decreases over epochs with a real learning rate; empty input rejected |
| `NeuralNetworkEvaluator` | Value bounded, policy normalised over exactly the legal plies; inference runs in eval mode even when training left the model in train mode |
| TicTacToe example | Ply validation; win/draw/legality with correct relative signs; heuristic evaluator policy sanity; `tictactoe_policy_loss` column mapping; MCTS finds a win-in-1 on a real board (integration) |

## Conventions

- **Imports**: test code imports the code under test absolutely (`from game_engine_core... import ...`, `from examples.tictactoe... import ...`) — the same way a consumer would, and consistent with the ruff TID rules.
- **Determinism**: tests must not flake. Randomised behaviour is tested either through forced situations (a single legal ply; enough MCTS iterations that the answer is forced) or by asserting distribution-level properties, not by seeding and asserting exact sequences.
- **Behaviour under test is current behaviour**: deferred design issues (root-node evaluation, the alternation assumption) are not "fixed" by tests; fixture games satisfy today's preconditions.
- **Private access**: tests prefer public API. Where a convention is only observable internally (backpropagation sign flipping), a targeted test may drive a private method directly and says so in a comment.
- **Type checking**: `tests/` is in the pyright `include` list; test code passes the same pyright/ruff bar as the packages.
