# Implementation Plan: Pass Position into decode_policy (issue #16)

Story: [story.md](story.md)

The change is a breaking signature widening of `decode_policy` from
`legal_plies: Sequence[TPly]` to `position: TPosition`, with no compatibility
shim. As with story #11, there is no intermediate state in which the suite
passes with only one side updated, so Step 1 moves the abstract method, its
caller, and every in-repo implementation together and verifies with the
existing tests. A dedicated coverage step follows to pin that `position` (not
a derived list) is what implementations now receive, then the README check.

### Step 1 — Widen the signature and update all in-repo implementations

Change `NeuralNetworkEvaluator.decode_policy`'s abstract signature so its
second parameter is `position: TPosition` instead of
`legal_plies: Sequence[TPly]`, and update its docstring to describe the new
parameter (implementations needing only the legal plies read
`position.legal_plies` themselves). Update `evaluate_position` to pass
`position` straight through instead of building `list(position.legal_plies)`.
Update every in-repo `decode_policy` implementation to the new signature,
deriving `legal_plies` from `position.legal_plies` internally where they
previously received it as a parameter: `tests/learning/nim_nn.py`'s
`NimNNEvaluator` and `examples/tictactoe_learning/tictactoe_nn_evaluator.py`'s
`TicTacToeNNEvaluator`. Update the direct-call tests in
`examples/tictactoe_learning/tests/test_nn_evaluator.py` to construct a
`TicTacToePosition` and pass it where a bare `legal_plies` sequence was passed
before. Drop the now-unused `Sequence` import from
`neural_network_evaluator.py` if nothing else in the file needs it.

Depends on: nothing — this is the root change everything else builds on.
Caller and implementations move together because the breaking change has no
shim.

Verification (automated): Run `pytest` — all existing tests pass unchanged in
behaviour (`test_neural_network_evaluator.py` exercises `decode_policy` only
indirectly through `evaluate_position`, so its assertions are an unchanged
regression check; the tictactoe direct-call tests pass under the new
signature). Also run `ruff check .` and `pyright` per CONTRIBUTING.md to
confirm the widened type is consistent at every call site.

### Step 2 — Test coverage that decode_policy receives the position itself

Add a test (in `examples/tictactoe_learning/tests/test_nn_evaluator.py`,
alongside the existing `decode_policy` tests) that would fail if
`evaluate_position` still passed only a derived list. A direct call to
`decode_policy` can't distinguish this — the new signature accepts a
`TicTacToePosition` either way — so the test instead exercises the production
`evaluate_position` → `decode_policy` wiring: a `_RecordingEvaluator` test
double subclasses `TicTacToeNNEvaluator`, overrides `decode_policy` to record
the object it receives (delegating to `super()` so real behaviour is still
exercised), and the test calls `evaluate_position` on it and asserts the
recorded object exposes `active_player_id` — a property that only exists on
the position itself, not on a bare `legal_plies` sequence.

Depends on: Step 1 (the new signature is what this test exercises).

Verification (automated): Run
`pytest examples/tictactoe_learning/tests/test_nn_evaluator.py` — the new/
adjusted test passes; temporarily reverting `evaluate_position` to pass
`list(position.legal_plies)` instead of `position` fails it, confirming the
test really pins the contract.

### Step 3 — README check

Verify `README.md` is still accurate. The README currently only names
`decode_policy` in prose (no signature shown), so the expected outcome is
confirming no update is needed; if review of the diff says otherwise, update
it. The `/update-readme` command automates this check.

Depends on: Steps 1–2 (the full diff must exist to review against the
README).

Verification (manual): Run `/update-readme` (or review the diff against
`README.md` by hand) and confirm either no change is warranted or the updated
text matches the new `decode_policy` signature.
