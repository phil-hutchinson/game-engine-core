# Implementation Plan: Widen Tournament Position Factory (issue #11)

Story: [story.md](story.md)

The change is a breaking signature widening of `Tournament`'s `position_factory`
from a zero-arg callable to one receiving the two participants in side order,
with no compatibility shim. Because there is no shim, the core change and the
in-repo callers must move in lockstep — a step that changed only one side would
leave the suite failing — so Step 1 updates them together and verifies with the
existing tests. New behaviour coverage and the README check follow as separate
steps.

### Step 1 — Widen the factory signature and update all in-repo callers

Change `Tournament.__init__` so `position_factory` takes the two `Player`
participants and returns the starting position, and have `_play_game` invoke it
with `(side_one, side_other)` — the same objects, in the same order, that it
assigns to sides 1 and -1. Update the `Tournament` docstring (and the
parameter's documentation) to state the side-order contract: the factory is
called once per game, the first argument holds side 1, and alternation within a
pairing is already applied; games that don't need the players ignore the
arguments. Update every in-repo caller to a two-parameter factory that ignores
its arguments: the factory lambdas in `tests/core/test_tournament.py` and
`tests/core/test_tournament_reporting.py`, the `counting_factory` test helper,
and `examples/tictactoe_learning/tournament.py` (whose bare
`TicTacToePosition.new_game` reference becomes a two-parameter wrapper).
`SelfPlayCollector` and its callers are untouched (non-goal).

Depends on: nothing — this is the root change everything else builds on. The
callers are updated in the same step because the breaking change with no shim
gives no intermediate state in which the suite passes.

Verification (automated): Run `pytest tests` — all existing tournament tests
pass unchanged in behaviour (they exercise scheduling, alternation, attribution,
and fresh-position-per-game through the new signature). Also run
`ruff check .` and `mypy` per CONTRIBUTING.md to confirm the widened type is
consistent at every call site.

### Step 2 — Test coverage for the side-order contract

Add coverage in `tests/core/test_tournament.py` that the factory's arguments
observably match the alternation `Tournament` applies: a recording factory
captures the player pair it receives for each game, and the test asserts (a)
the factory is invoked exactly once per game, and (b) for every game, the names
of the received `(side_one, side_other)` pair equal the side 1 / side -1 names
the corresponding `GameRecord` reports — across enough games per pairing that
alternation actually flips the order. Fold the existing call-counting test into
this coverage or keep it alongside, whichever reads better.

Depends on: Step 1 (the widened signature is what the recording factory
observes).

Verification (automated): Run `pytest tests/core/test_tournament.py` — the new
test passes, and deliberately breaking the pairing order in the assertion (or
temporarily swapping the arguments in `_play_game`) fails it, confirming the
test really pins the contract.

### Step 3 — README check

Verify `README.md` is still accurate. A search shows the README does not
currently show a `Tournament` construction, so the expected outcome is
confirming no update is needed; if review of the diff says otherwise, update
it. The `/update-readme` command automates this check.

Depends on: Steps 1–2 (the full diff must exist to review against the README).

Verification (manual): Run `/update-readme` (or review the diff against
`README.md` by hand) and confirm either no change is warranted or the updated
text matches the new factory signature.
