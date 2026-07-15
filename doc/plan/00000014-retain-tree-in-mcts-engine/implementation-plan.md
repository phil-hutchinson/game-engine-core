# Implementation Plan: Retain the MCTS tree between plies (issue #14)

Story: [story.md](story.md)

The code change spans the `MCTSEngine`, the `GameEngine`/`Player` protocols and
their implementations, and the `StandardGame` loop, and has already been
implemented by the developer as one cohesive change. This plan treats that
implementation as Step 1 and follows it with the verification, test-coverage,
quality-gate, and README steps the story still needs. The steps after Step 1 are
what carry the change from "written" to "verified and submittable".

### Step 1 — Code changes (implemented by developer)

Tree retention as described in the story, already in the working tree:

- `MCTSEngine` holds a retained `_root_node`; `select_ply` warm-starts from it
  (building a fresh root only when none is held); `observe_ply` re-roots onto the
  child matching `str(ply)` and detaches it (`parent`/`ply_from_parent` cleared),
  or clears the retained root when there is no match; `reset()` clears the root.
  Root creation is separated from tree growth internally, and
  `select_ply_with_policy` is unchanged in behaviour (always builds a fresh
  tree).
- `GameEngine` and `Player` protocols gain `observe_ply` and `reset`.
- `AIPlayer` forwards both to its engine; `HumanPlayer` and `RandomEngine`
  implement them as no-ops.
- `StandardGame.run` resets both players at game start and broadcasts every
  applied ply to both players via `observe_ply`.

Depends on: nothing — this is the root change everything else verifies.

Verification (automated): Run `pytest` and confirm the existing suite still
passes (behaviour of `select_ply`, `select_ply_with_policy`, and the game loop is
unchanged for the already-covered cases). This only establishes no regression;
the retention behaviour itself is pinned in Step 2.

### Step 2 — Test coverage for retention, reset, and the miss fallback

Add coverage in `tests/core/test_mcts_engine.py` (using the existing Nim fixture
where a concrete game is needed) for the behaviour this story introduces:

- **Re-rooting keeps the subtree.** After a `select_ply` followed by an
  `observe_ply` for the played ply, the engine's retained root is the former
  child for that ply, with `parent` and `ply_from_parent` cleared and its visit
  count / children / cached statistics preserved (not reset to zero).
- **Miss clears the root.** After `observe_ply` for a ply that has no matching
  child (e.g. an unexplored ply, or a ply observed before any search has run),
  the next `select_ply` produces a legal ply for the *incoming* position —
  demonstrating the fresh-root rebuild rather than a stale tree.
- **`reset()` clears the root.** After a search, `reset()` followed by
  `select_ply` on a fresh position behaves as a cold start.
- **Cross-game isolation through `StandardGame`.** A game run that reuses the same
  `AIPlayer`/engine instances (as a tournament does) yields legal, correct play
  in the second game — i.e. `reset()` is actually invoked at game start and no
  tree leaks between games. This can live in `tests/core/test_standard_game.py`
  if it fits better there.
- Update existing tests where failing, as needed

Depends on: Step 1 (the behaviour under test).

Verification (automated): Run `pytest tests/core/test_mcts_engine.py` (and
`tests/core/test_standard_game.py` if the isolation test lands there) — the new
tests pass, and temporarily reverting the reset-on-miss line (making `observe_ply`
keep a stale root on a miss) fails the miss-fallback test, confirming it really
pins the behaviour.

### Step 3 — Type check and lint

Run the quality gates over the full change per CONTRIBUTING.md.

Depends on: Steps 1–2 (all code, including new tests, must exist).

Verification (automated): Run `pyright` and `ruff check .` and confirm both are
clean — in particular that the new protocol methods are consistently typed across
`GameEngine`, `Player`, `AIPlayer`, `HumanPlayer`, `RandomEngine`, and
`MCTSEngine`.

### Step 4 — README check

Verify `README.md` is still accurate given the new `observe_ply`/`reset` methods
on the `GameEngine`/`Player` protocols. If the README documents the engine or
player interface, or shows a custom engine/player implementation, update it to
include the new methods; otherwise confirm no update is needed. The
`/update-readme` command automates this check.

Depends on: Steps 1–3 (the full diff must exist to review against the README).

Verification (manual): Run `/update-readme` (or review the diff against
`README.md` by hand) and confirm either no change is warranted or the updated
text reflects the new protocol methods.
