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

### Outcome — no retuning warranted

Measured as blunder rate against a negamax solver over all 1090 distinct
non-terminal positions reachable in five plies. The engine is now deterministic
(the expansion shuffle is gone and every tie-break is a `max`), so these are
exact rates, not samples. The old exploration scale was reproduced under the new
engine by an evaluator returning all-`1.0` priors, which isolates the scale
change from the expansion change:

| iterations | 1/K priors (new) | 1.0 priors (old scale) | heuristic |
| --- | --- | --- | --- |
| 50 | 12.3% | 11.1% | 0.7% |
| 200 | 3.6% | 5.2% | 0.0% |
| 1000 | 0.3% | 0.8% | 0.0% |
| 5000+ | 0.0% | 0.0% | 0.0% |

Normalised priors are neither wrong nor in need of a compensating constant: they
are *better* from 200 iterations up, since narrowing exploration concentrates
visits where visit-count selection reads them. Below ~100 iterations the old
wider scale is marginally better, which is the expected crossover — with nine
legal plies and a budget that small, the risk is failing to sample a ply at all.
Neither example is anywhere near that regime: `main.py`'s null-evaluator engine
runs 200,000 iterations (0 blunders) and its heuristic engine 200 (0 blunders).
No change to `examples/tictactoe`, and no change to the default
`exploration_constant`.

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

### Outcome — targets sharpened, not degenerate; play-time budget corrected

Self-play targets are concentrated where the game is nearly decided, not
collapsed. Over 300 samples on an untrained network, every legal ply still
receives a non-zero visit share at every stage, and the opening stays near
uniform (3.074 bits against a 3.170 uniform baseline) while positions with five
or six legal plies sharpen to a mean top probability of ~0.65–0.71. All nine
opening plies still occur across 60 temperature-1.0 games, so self-play
diversity is intact. The mean entropy drop against the figure recorded in the
general-cleanup review (2.181 → 1.709 bits) is entirely this late-game
sharpening — the search resolving tactics it previously spread visits across.
A six-iteration training run falls steadily on both heads (total 2.735 → 2.412,
value 0.552 → 0.353, policy 2.183 → 2.059).

One example change: `main.py`'s neural play engine moves from 10 iterations to
200. The 10 dated from the era this story ends — when the root was never
evaluated, ten iterations bought one visit per root child and the policy head
was unused at play time (general-cleanup story review, finding #2, which
deferred the budget question to whenever the root-prior quirk was fixed).
Measured against the solver with the checked-in weights, 10 iterations blunders
in 36.3% of positions, 50 in 12.5%, and 200 in 3.7% at roughly 30 ms per ply —
so 200 is both defensible and imperceptible, and it matches the self-play and
heuristic budgets. `tournament.py`'s `--mcts-iterations` default of 100 is left
alone: it applies equally to every player in a comparison, so it trades absolute
strength for run time without biasing the result.

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

### Outcome — one edit, to the MCTS section

The required policy is the only part of this story a consumer of the package has
to know about, since it is the one changed contract they can break: an evaluator
that omits a legal ply now raises. The MCTS section gained a sentence describing
the expansion model (descend to a leaf, evaluate once, attach a child per legal
ply) and a short paragraph stating the policy requirement and pointing at
`NullEvaluator` as the pattern for evaluators without a policy head.

Everything else was checked and left alone: the "what's in the box" table entry
for `NullEvaluator` ("uniform prior, used as a baseline") is if anything more
accurate now, `PositionEvaluation (value + policy)` is unchanged, the retention
paragraph still describes what re-rooting carries forward, and the quick-start
snippet still runs as written. The search-internal changes — full expansion,
removed fields, evaluate-then-expand ordering — are implementation detail that
the README should not carry.
