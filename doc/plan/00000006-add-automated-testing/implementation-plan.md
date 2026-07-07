# Implementation Plan: Add Automated Testing

Story: [story.md](story.md)

The testing approach (suite layout, fixture-game strategy, conventions) is designed in [automated-testing.md](automated-testing.md); this plan sequences its construction. Every step's verification is automated — running the tests the step adds — so no manual testing pauses are needed. Each step also requires `pyright` and `ruff check .` to pass clean, per CONTRIBUTING.md.

### Step 1 — Pytest tooling and test scaffold

Add pytest as a pinned dev-container tool (same pinning policy as ruff: installed in the `.devcontainer/Dockerfile` image, version bumped deliberately) and configure it in `pyproject.toml` (`[tool.pytest.ini_options]`: test paths covering the package suite and the example suite). Add the `tests/` tree to the pyright `include` list so test code is type-checked. Create the `tests/` skeleton — `tests/core/` and `tests/learning/` packages — with a single trivial smoke test that imports the two library packages, proving discovery and configuration work end to end. Install pytest at the pinned version into the running container as well, since a Dockerfile change only takes effect on rebuild.

Depends on: nothing — this is the scaffolding every later step runs on.

Verification (automated): Run `pytest` from the repository root and confirm it discovers and passes the smoke test. Confirm `pyright` and `ruff check .` pass clean.

### Step 2 — Fixture game and core game-flow tests

Implement the shared test fixture game in `tests/core/`: a minimal Nim-style subtraction game (take 1–2 tokens from a pile; taking the last token wins; players strictly alternate) with a configurable set of legal takes — restricting it to a single take makes every game a forced line, which later steps use for deterministic outcome tests. Add tests that pin down the fixture's own contract (outcome perspective, legality, alternation), then use it to test `StandardGame` (relative-to-absolute outcome conversion for a player-1 win and a player-2 win, using a stub UI and scripted players) and `RandomEngine` (selects a legal ply; raises on a terminal position).

Depends on: Step 1 (test scaffold). The fixture game introduced here is the game every later package-test step builds on, so its own behaviour must be pinned first.

Verification (automated): Run `pytest tests/core` and confirm the new tests pass alongside the smoke test. Confirm `pyright` and `ruff check .` pass clean.

### Step 3 — MCTS engine tests

Test `MCTSEngine` against the fixture game, centred on the sign conventions the story exists to protect: a forced win in 1 is found (from a pile where taking everything wins); backpropagation alternates the value sign at each tree level and increments visits along the path; terminal positions are handled (evaluating a terminal node returns its exact outcome; `select_ply` on a position with no legal plies raises). Cover the visit distribution via the public `select_ply_with_policy`: it spans all legal plies including zero-visit ones, sums to 1, and falls back to uniform when nothing was visited. Cover policy priors: an evaluator that returns a policy sees it distributed to children as priors, and a policy missing a legal ply's entry raises. Cover temperature-0 selection picking the most-visited ply.

Depends on: Step 2 (fixture game with known game-theoretic values; single-take mode for deterministic trees).

Verification (automated): Run `pytest tests/core` and confirm all MCTS tests pass. Confirm `pyright` and `ruff check .` pass clean.

### Step 4 — Learning package tests

Test `game_engine_learning` in `tests/learning/`, using the fixture game plus a minimal torch model and a concrete `NeuralNetworkEvaluator` subclass defined in the test package. Coverage: `SelfPlayCollector` target-value signs — with the fixture in single-take mode the game is a forced line with a known winner, so every sample's target value is exactly predictable (last ply's sample is `-final_outcome`, alternating backwards) — plus sample counts and policy-target presence; `TrainingLoop` epoch-loss reporting is the sample-weighted mean of batch losses (verified with a zero learning rate so the model is frozen and per-batch losses can be computed independently — this makes the finding-#7 scratch verification durable), and rejection of an empty sample list; `NeuralNetworkEvaluator.evaluate_position` restores eval-mode inference even after the model was left in train mode.

Depends on: Step 3 (self-play drives the MCTS engine, whose fixture-game behaviour is now pinned).

Verification (automated): Run `pytest tests/learning` and confirm all learning tests pass. Confirm `pyright` and `ruff check .` pass clean.

### Step 5 — Example tests for TicTacToe

Add the separate example suite under `examples/`, written the way a library consumer would test their own game: `TicTacToePly` input validation; `TicTacToePosition` outcome and legality (win detection across rows/columns/diagonals with the current-player-relative sign, draw detection, legal-ply shrinkage, rejection of occupied squares, immutability of the source position); `TicTacToeHeuristicEvaluator` policy sanity (distribution over exactly the legal plies, summing to 1, winning move ranked highest); the learning example's `tictactoe_policy_loss` str(ply)-to-column mapping; and one integration test proving the stack composes — `MCTSEngine` with the `NullEvaluator` finds a win-in-1 on a real board.

Depends on: Step 1 (tooling) only; deliberately independent of the `tests/` fixture game so the suite stands alone as an example. Placed after the package steps because it re-uses testing patterns established there.

Verification (automated): Run `pytest examples` and confirm the example tests pass, then run bare `pytest` to confirm both suites run together. Confirm `pyright` and `ruff check .` pass clean.

### Step 6 — Testing documentation

Reconcile [automated-testing.md](automated-testing.md) with what was actually built (layout, commands, any decisions that shifted during implementation) and add a test-running section to `CONTRIBUTING.md` alongside the existing pyright/ruff instructions.

Depends on: Steps 1–5 (documents the finished suite, not the intended one).

Verification (manual): Follow the documented commands verbatim from the repository root and confirm each runs what the document says it runs.

### Step 7 — README check

Verify `README.md` is still accurate given the story's changes, updating it if warranted (the `/update-readme` command automates this). The story adds development tooling and test trees; anything the README says about repository layout or contributor workflow may need to reflect them.

Depends on: Steps 1–6 (the README can only be checked against the story's final shape).

Verification (manual): Read the resulting README diff (or confirmation that no change was needed) and confirm it matches the story's actual changes.

## After completion

Update this plan if any step's substance changed during implementation, and record each step's commit against it in the peer review when the story is reviewed.
