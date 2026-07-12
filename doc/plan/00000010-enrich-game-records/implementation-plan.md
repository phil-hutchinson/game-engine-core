# Implementation Plan: Enrich Game Records (issue #10)

Implements [story.md](story.md): `result_reason` on game results, and the `GameLogging` protocol split with `ply_annotation`.

Sequencing notes:

- The two story changes are largely independent; the `result_reason` chain (Steps 1–2) comes first because it is smaller and does not reshape constructor signatures. The protocol split (Steps 3–4) follows, and reporting (Step 5) needs only Step 2.
- The story mandates breaking changes with no compatibility shims, so each protocol-changing step must update every in-repo implementer and caller in the same step to keep the suite green. Step 3 is the largest step for exactly this reason — it cannot be split without introducing a temporary shim.

### Step 1 — `outcome_reason` on `GamePosition` and its implementers

Add the required `outcome_reason` property to the `GamePosition` protocol: `None` while the game is ongoing, a short game-specific string once `outcome` is non-`None`. Implement it on both in-repo positions — `TicTacToePosition` (distinct reasons for a completed line vs. a full-board draw) and the Nim fixture's `NimPosition` (a single reason, since taking the last token is Nim's only ending). Extend the existing position test suites (`examples/tictactoe/tests/test_tictactoe_position.py`, `tests/core/test_nim_fixture.py`) to assert `outcome_reason` is `None` for ongoing positions and carries the expected string for each terminal case.

Depends on: nothing — this is the bottom of both dependency chains (Step 2 reads the property; game implementers must exist before anything consumes them).

Verification (automated): Run `pytest` and confirm the full suite passes, including the new terminal/ongoing `outcome_reason` assertions in both position test files.

### Step 2 — `result_reason` on `GameResult`, populated by `StandardGame`

Add the required `result_reason` field to `GameResult`, and have `StandardGame.run()` populate it from the terminal position's `outcome_reason`. Update the one test that constructs `GameResult` directly (`tests/core/test_tournament_reporting.py`'s markdown-escaping test) for the new field. Extend `tests/core/test_standard_game.py` to assert that a completed game's result carries the Nim fixture's reason string.

Depends on: Step 1 (`StandardGame` sources the reason from the position property introduced there). Step 5 depends on this field existing.

Verification (automated): Run `pytest` and confirm the suite passes, including the new assertion that `GameResult.result_reason` matches the terminal position's reason.

### Step 3 — `GameLogging` protocol, slimmed `GameUI`, and the new `StandardGame` seam

Introduce the `GameLogging` protocol (new module under `game_engine_core/protocols/`) with `text_board` and `ply_annotation(from_position, ply, to_position)`. Slim `GameUI` to the interactive pair (`render_board`, `get_next_ply`). Rework `StandardGame` to take a required `GameLogging` and an optional `GameUI` (`None` for headless play): `text_board` and `ply_annotation` calls go to the logging object, board rendering happens only when a UI is present, and the game log's annotation entry becomes `ply_annotation(...)` instead of `str(ply)`.

Because this is a breaking change with no shims, the same step updates every implementer and caller:

- `TicTacToeUI` implements both protocols on one class, inheriting both protocol bases explicitly — matching its existing explicit `GameUI` subclassing (`ply_annotation` trivially returns the ply's string).
- The Nim fixture's stub gains the `GameLogging` members (trivial annotation) alongside its existing UI members.
- Both example entry points (`examples/tictactoe/main.py`, `examples/tictactoe_learning/main.py`) pass the UI object in both roles.
- `Tournament._play_game` keeps compiling by passing its existing UI object as the logging argument with `game_ui=None` — its public constructor is untouched until Step 4.
- `tests/core/test_standard_game.py` adapts to the new signature, and gains coverage that `ply_annotation` — not `str(ply)` — is what lands in the game log, using a test-local logging stub whose annotations are distinguishable from the ply's string.

`HumanPlayer` is unaffected (it holds its own `GameUI` for `get_next_ply`).

Depends on: Step 2 (touches the same `StandardGame.run()` body; doing the smaller change first keeps this step's diff focused). Step 4 depends on the `GameLogging` type and `StandardGame`'s headless mode introduced here.

Verification (manual): Run `python -m examples.tictactoe --p1 heuristic --p2 random` and confirm the game plays to completion with boards rendered. Then run `pytest` and confirm the suite passes, including the new game-log annotation coverage.

### Step 4 — `Tournament` takes a `GameLogging` and runs headless

Change `Tournament`'s constructor to accept a `GameLogging` instead of a `GameUI`, and run games with `game_ui=None`. Update the class docstring (which currently says "the GameUI is only consulted for text_board") to describe the new arrangement. Update the tournament callers and tests: `examples/tictactoe_learning/tournament.py` passes the TicTacToe UI object as the logging parameter, and `tests/core/test_tournament.py` / `tests/core/test_tournament_reporting.py` adapt — the "render is forbidden" test can now assert headlessness more directly, since a logging-only object with no render member is sufficient to run a tournament.

Depends on: Step 3 (needs the `GameLogging` type and `StandardGame`'s optional-UI mode).

Verification (automated): Run `pytest` and confirm the tournament suites pass, including a check that tournaments run with an object implementing only `GameLogging`.

### Step 5 — `result_reason` in tournament reporting

Include `result_reason` in each per-game JSON file written by `write_tournament_report`, and refresh the reporting module's docstring (it currently explains that ply annotations are `str(ply)`). Extend `tests/core/test_tournament_reporting.py` to assert the reason appears in the parsed per-game JSON and matches the record's result.

Depends on: Step 2 (the field must exist on `GameResult`); placed after Step 4 so the reporting tests are only touched once the tournament seam is stable.

Verification (automated): Run `pytest` and confirm the reporting tests pass, including the new per-game JSON `result_reason` assertion.

### Step 6 — README and documentation check

Verify `README.md` against the story's changes and update where stale: the protocols table (add `GameLogging`), the quick-start `StandardGame` snippet (new constructor shape), and the tournament section if its description of UI wiring no longer matches. The `/update-readme` command automates this review. Also sweep remaining docstrings that describe the old arrangement.

Depends on: Steps 1–5 (documents the final state of the APIs).

Verification (manual): Read the updated README sections and confirm the quick-start snippet matches the actual `StandardGame` constructor, and the protocols table lists the new protocol. Run `pytest` one final time to confirm the whole suite is green.
