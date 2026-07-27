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

## Decisions

Two points the story leaves open, resolved here so Step 1 is unambiguous.

- **A policy is required; there is no engine-side fallback.** The story says to
  remove the uniform-prior fallback (the `prior = 1.0` assignment, which existed
  only because the root was never evaluated). Rather than replace it with a
  uniform fallback inside the engine, the engine now *requires* every evaluation
  to carry a policy, and `NullEvaluator` supplies a uniform one itself. The
  reason: under full expansion a prior is structural — every child needs one at
  construction, on every path — so "optional policy" is a false option, since
  something must invent the numbers regardless. The only question is who, and
  putting it in the evaluator that declined a policy head keeps the engine's hot
  path branch-free and keeps the fleet wave's bulk expansion (#23) free of a
  per-game special case. Two consequences to accept deliberately: this makes
  `PositionEvaluation.policy` a required field, which is a contract change the
  story's non-goals arguably fence off as #22's (judged fair game at `0.1.0` with
  three in-repo implementations); and `NullEvaluator` must now generate
  `legal_plies` to build its uniform policy, which the engine then generates again
  when expanding — real duplicated work on that path, accepted because the cost
  lands on the evaluator that opted out.
- **The "reduces to UCT" claim is retired, not relocated.** It was never true of
  the formula: UCB1's exploration term is unbounded as visits approach zero, which
  is what forces every sibling to be tried once, whereas PUCT's is finite at zero
  visits. The old engine only behaved that way because one-child-per-iteration
  expansion did the round-robin by hand — so this story is precisely what
  invalidates the claim. `MCTSNode.puct_value` and `PositionEvaluation.policy`
  both assert it today and are corrected in Step 1, since the same change makes
  them wrong.
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
The existing policy-completeness check (raise `ValueError` naming the ply missing
from the policy) moves to full expansion, where it covers every legal ply in one
pass. Per the decisions above, make `PositionEvaluation.policy` required, drop the
engine's fallback, have `NullEvaluator` return a uniform policy over the legal
plies, and correct the two docstrings that claim UCT equivalence. Keep the
`is_fully_expanded` / `unexplored_plies` machinery in place for now if it
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
called exactly once per iteration (a counting evaluator); a terminal leaf is
re-scored from its outcome without reaching the evaluator; `NullEvaluator`
produces a uniform policy summing to one; and — the point of the story — a child
with a dominant prior is re-selected while a low-prior sibling stays at zero
visits, which the old code could not produce.

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
stays.

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
