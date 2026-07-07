# Story: Add Automated Testing (issue #6)

## Goal

Add an automated test suite (pytest) covering the `game_engine_core` and `game_engine_learning` packages, plus a separate test suite for the examples that doubles as a demonstration of how a developer consuming the library would test their own game implementation.

## Motivation

This is the deferred finding #4 from the [general-cleanup peer review](../00000002-general-cleanup/story-peer-review.md): the codebase is dense with sign-convention logic — current-player-relative outcomes, per-level negation in MCTS backpropagation, outcome back-labelling in self-play — exactly the code earlier peer reviews had to hand-verify in prose. Regressions in any of these conventions currently go undetected. Several prior fixes were verified with throwaway scratch scripts (e.g. general-cleanup finding #7, whose resolution explicitly says "becomes a unit test when the #4 test-suite story lands"); this story turns that verification into durable, runnable coverage.

The pytest tooling itself was deferred from the containerization story (issue #4), which established the pattern this story follows: tools pinned in the dev container, configured in `pyproject.toml`.

## Scope

### 1. pytest tooling

Add pytest to the dev container (pinned, same policy as ruff/pyright) with configuration in `pyproject.toml`. Tests are development-only: nothing about the packages as consumed externally changes, and no test code ships in the distribution.

### 2. Package tests

Tests for the two library packages, self-contained (no dependency on `examples/`), using a minimal fixture game where a real game implementation is needed. Coverage centres on the peer-review finding's list:

- MCTS terminal handling and sign conventions — a forced win in 1 must be found
- MCTS visit-distribution normalisation
- `SelfPlayCollector` target-value signs
- `StandardGame` relative-to-absolute outcome conversion
- `TrainingLoop` sample-weighted epoch loss (the finding-#7 scratch verification, made durable)

### 3. Example tests

A separate suite testing the TicTacToe examples (`TicTacToePosition` outcome/legality among them — the remaining item from the finding's list). Beyond regression protection, this suite is itself an example: it shows a developer integrating the library how to test a concrete game implementation against the protocols.

### 4. Testing documentation

A new story artifact, `automated-testing.md` (alongside this story's planning files), documenting the testing approach: suite layout, the fixture-game strategy, conventions, and how to run everything. `CONTRIBUTING.md` gains test-running instructions next to the existing pyright/ruff ones.

## Out of Scope

- **CI pipeline** — running the suite automatically on push/PR remains a future story (same status as pyright/ruff)
- **Coverage tooling or thresholds** — `.gitignore` anticipates `.coverage`/`htmlcov/`, but measuring coverage is not part of this story
- **Behavioural fixes** — general-cleanup findings #2 (root node never evaluated) and #3 (strict-alternation assumption) remain deferred to their own stories; tests written here assert *current* behaviour and use strictly alternating fixture games

## After-completion notes

- **`.gitignore` for `.claude/settings.local.json`** (commit `ee222d5`) — added mid-branch at the developer's explicit request. It is inert development tooling with no interaction with the testing work and was too small to warrant its own story, so it was folded into this branch. Recorded here (per peer-review comment #6) to keep the plan-to-commit audit trail complete.
