# Implementation Plan: True PUCT / full node expansion

Replaces one-child-per-iteration expansion with AlphaZero-style full expansion in
`MCTSEngine`. The sequence lands the behaviour change first (Step 1), then strips
the machinery it makes vestigial (Step 2), then re-checks search quality in the
example applications now that priors — not visit-order — drive descent (Steps 3–4),
then docs (Step 5).

## Ownership

The developer implements the engine changes (Steps 1 and 2 production code).
Claude writes and updates the tests for every step, and owns any changes to the
`examples/tictactoe` and `examples/tictactoe_learning` applications.

## Assumptions

Two points the story leaves open, resolved here so Step 1 is unambiguous. Raise
them before starting if either is wrong.

- **Priors when the evaluator supplies no policy.** The story says to remove the
  uniform-prior fallback, which is the `prior = 1.0` assignment that exists only
  because the root was never evaluated. A policy-less evaluator is still
  supported (`NullEvaluator` ships in the package, and `PositionEvaluation.policy`
  is documented as optional), so expansion still needs a prior in that case —
  this plan assumes an actual uniform distribution, `1 / len(legal_plies)`, rather
  than the current `1.0`. Note this is a real change in exploration scale: with
  nine legal plies the exploration term shrinks ninefold against today's
  behaviour, which is what Step 3 exists to measure.
- **What marks a leaf.** "Not yet evaluated" is observable as "has no children",
  because full expansion attaches every legal ply at once and a terminal position
  has no legal plies. This plan assumes the childless test rather than a new
  `is_expanded` flag, which keeps the node free of a field that would have to be
  maintained through re-rooting. A consequence: a terminal node is selected as a
  leaf on every iteration that reaches it, and is re-scored from its outcome each
  time — correct, and it keeps "exactly one evaluation per iteration" true without
  a special case.

## Step 1 — Evaluate-then-expand-all iteration

Rework the iteration in `mcts_engine.py` so one iteration is: descend from the
root by PUCT while the current node has children; evaluate the leaf reached
(outcome for a terminal position, evaluator otherwise); attach a child for every
legal ply of that leaf, each seeded with its prior from the returned policy;
backpropagate the value from that leaf. Backpropagation now starts at the
evaluated leaf itself rather than at a newly created child, so a fresh root is
evaluated and expanded by iteration 1 and receives real priors for its children.
The existing policy-completeness check (raise when a legal ply is missing from
the policy) moves to full expansion, where it covers every legal ply in one pass.
Keep the `is_fully_expanded` / `unexplored_plies` machinery in place for now if it
simplifies the diff; Step 2 removes it.

Depends on: nothing (entry point). Steps 2–5 all depend on this behaviour being
correct.

Tests (Claude): several existing assertions in `tests/core/test_mcts_engine.py`
encode the old shape and must be re-pinned to the new one — notably the visit
accounting in `test_search_values_carry_correct_signs` (the root now consumes one
iteration of its own, so children sum to `iterations - 1`), the one-iteration
premise in `test_visit_distribution_includes_zero_visit_plies` (after one
iteration every child exists at zero visits, so the distribution falls to the
uniform branch), and `test_observe_ply_miss_on_unexplored_child_clears_root_and_rebuilds`
(no legal ply is ever unexplored after the root is expanded, so the miss must be
constructed deliberately rather than by running a single iteration). New tests to
add: the root is evaluated and fully expanded after one iteration, with priors
taken from the evaluator's policy rather than a uniform default; the evaluator is
called exactly once per iteration (a counting evaluator); and — the point of the
story — a child with a dominant prior is re-selected while a low-prior sibling
stays at zero visits, which the old code could not produce.

Verification (automated): Run `pytest tests/core/test_mcts_engine.py` and confirm
the updated suite is green, including the new prior-dominance and
one-evaluation-per-iteration tests. Then run `pytest` to confirm nothing else in
the package depends on the old expansion shape.

## Step 2 — Remove the machinery full expansion makes vestigial

Delete `unexplored_plies` and `is_fully_expanded` from `MCTSNode`, the expansion
`random.shuffle` (expansion order no longer carries any bias, since all children
are created together), the `node.policy` field (the policy is consumed at
expansion and never read again — the non-goals allow discarding it), and the
"include unexplored plies with 0 visits" reconciliation in `_visit_distribution`,
which can now read counts straight off the children — note the uniform fallback
for a root with no children at all (`iterations=0`) still needs `legal_plies` and
stays. Update the docstrings that describe
the old model: `MCTSNode.puct_value`'s "reduces to UCT" note and
`PositionEvaluation.policy`'s description of the uniform fallback both need to
match the resolved prior semantics.

Depends on: Step 1 (nothing may reference these once the new flow is the only
flow).

Tests (Claude): remove or rewrite any test that reaches for the deleted
attributes; keep the zero-visit and uniform-fallback distribution tests, since
those behaviours survive even though their implementation simplifies.

Verification (automated): Run `pytest` and confirm it stays green
with no behaviour change from Step 1 — this step is pure removal, so a green run
that also still passes the new Step 1 tests is the signal. Run `pyright` to confirm no dangling
references to the removed fields.

## Step 3 — Re-check search strength in the TicTacToe example

With priors normalised, the exploration term is scaled by `1/K` where it was
previously scaled by `1.0` for policy-less evaluators, so the effective
exploration constant has changed for every `NullEvaluator`-driven search — the
200,000-iteration null-evaluator engine in `examples/tictactoe/main.py` and the
existing MCTS integration test. Measure whether the example still plays at the
strength it did, and adjust the example's exploration constant or iteration count
if it does not. Do not silently retune the engine default to mask a regression:
if the default `exploration_constant` is genuinely wrong for normalised priors,
that is a deliberate change to make here with the measurement that justifies it.

Depends on: Steps 1–2 (measuring the finished search).

Tests / example updates (Claude): the TicTacToe MCTS integration test is the
existing strength check; extend the example's test coverage if the win-in-one
case turns out to be too weak a signal to detect a strength regression (e.g. a
position where the correct ply requires depth rather than a one-ply tactic).

Verification (manual): Run `pytest examples/tictactoe` and confirm it passes.
Then play the example directly — `python -m examples.tictactoe` — against both the
null-evaluator and heuristic-evaluator engines and confirm the AI still blocks
threats and takes wins as before. A pure-MCTS TicTacToe player should be
unbeatable; anything less is a regression to investigate before proceeding.

## Step 4 — Re-check the learning path

Confirm the neural-network path behaves under full expansion: the NN evaluator is
now called once per iteration including at the root, and every root child receives
a genuine network prior for the first time (previously only deeper nodes did), so
the self-play visit distributions and their entropy are expected to shift. The
concern to rule out is a broken or degenerate policy target, not a changed one.

Depends on: Step 3 (the search itself is settled before judging what it produces).

Tests (Claude): the learning suites (`tests/learning/`) and any example tests
covering self-play collection.

Verification (manual): Run `pytest` for a full green suite, then run
`python -m examples.tictactoe_learning.selfplay` and confirm the reported mean
policy entropy is finite, below the uniform baseline it prints alongside, and that
the collected samples cover all legal plies per position. A short
`python -m examples.tictactoe_learning.train` run should still show loss
decreasing.

## Step 5 — README and docs check

Check `README.md` against the finished change. The MCTS section describes PUCT
selection, the policy/value head interface, and what tree retention carries
forward; the expansion model and the "uniform prior when policy is None" claim are
the parts most likely to be stale. Update if warranted, or record that no update
is needed.

Depends on: Steps 1–4 (documents the settled behaviour).

Verification (manual): Run `/update-readme` (or review the branch diff against
`README.md`) and confirm the README either is updated to describe full expansion
and the resolved prior semantics, or is verified as already accurate. Run `pytest` once more to confirm the branch is green before closing
the story.
