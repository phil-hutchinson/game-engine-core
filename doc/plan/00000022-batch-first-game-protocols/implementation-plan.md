# Implementation Plan: Batch-first game protocols

Introduces the batch seams the fleet (#23, #24) will call. The sequence builds
the new artifact first with no consumers (Step 1), routes the engine through it
without changing behaviour (Step 2), then widens the evaluator surfaces —
protocol first (Step 3), then the network path that makes the batching real
(Step 4) — before migrating the collector (Step 5) and the docs (Step 6).

Steps 1–2 and 3–4 are independent chains; either could be done first. Step 5
needs both.

## Ownership

Claude implements every step in full — library production code, tests, examples
and fixtures. The design was settled in discussion before the plan was written,
so the developer's involvement is a review at the end of each step rather than a
split of the work within one. Each step stops for that review before the next
begins.

## Decisions

Points settled in discussion, recorded here so the steps are unambiguous.

- **The batch seam is a class, not a protocol pair.** `BatchPositionProcessor`
  is concrete and instantiable, with working defaults on every method. There is
  no separate protocol and no `SequentialBatchPositionProcessor` — declaring the
  surface twice would express nothing the base class does not, and selective
  override is the expected steady state rather than a fallback for toy games.
  The corollary for the evaluator seams is the inverse: `encode_positions` /
  `decode_policies` stay abstract, because no game-agnostic default exists.
- **Method names carry no `batch_` prefix.** The receiver already says it —
  `batch_ops.legal_plies(leaves)` — and the plural is the signal. Returns that
  are per-position collections nest (`legal_plies` returns a sequence of
  sequences); arguments that pair align by index rather than by tuple, matching
  `decode_policies`, which cannot zip because its rows arrive as one stacked
  tensor.
- **No scalar form survives on a widened surface.** `evaluate_position`,
  `encode_position` and `decode_policy` are removed, not deprecated — the
  package is alpha with one implementing repository, and leaving both forms in
  place is the two-touchpoint world this story exists to end. The cost is a
  three-line comprehension in each example evaluator, which the story treats as
  worth paying.
- **`GamePosition` keeps its scalars, and that is what makes delegation
  possible.** An implementer who has vectorised `legal_plies` on their processor
  can write the position's scalar property as a batch-of-one call into it. The
  hazard is delegating to a method the processor did *not* override, which
  recurses into the default that called it; the class docstring must state the
  rule, and Step 1 may add the introspection guard that makes it loud.
- **The engine takes the processor as a constructor argument with a default.**
  Not a back-reference from the position: existing call sites — including both
  examples and every test — construct `MCTSEngine` unchanged, and a game can
  play one position type under variant processors.

## Step 1 — `BatchPositionProcessor`

Add a concrete generic class (over the ply and position type parameters) in
`game_engine_core/game/`, with `outcomes`, `legal_plies` and `apply_plies`
implemented as loops over the corresponding `GamePosition` members. Returns are
index-aligned with the input and, for `legal_plies`, nested one level. The
docstring carries the alignment contract, the "override selectively" intent, and
the delegation rule from the decisions above. Optionally add the introspection
that reports which methods a subclass actually overrode, so a delegating
position can assert its assumption at construction rather than blowing the stack
mid-search. Nothing consumes the class in this step.

Depends on: nothing (entry point). Steps 2 and 5 consume it.

Tests (Claude): a new suite over the Nim fixture covering index alignment for
all three methods, the nested shape of `legal_plies`, empty-batch behaviour, and
`apply_plies` pairing positions with plies by index rather than cross-producting
them. Plus a subclass in the test file that overrides exactly one method,
asserting the override is used and the other two still loop — the selective
override the story is built around. If the introspection guard is implemented,
one test that it reports the overridden method and not the inherited ones.

Verification (automated): Run `pytest tests/core` and confirm the new suite is
green and nothing else moved. Run `pyright` to confirm the generic parameters
line up with `GamePosition`'s.

## Step 2 — Route `MCTSEngine` through the processor

Give `MCTSEngine` a `batch_ops` constructor argument defaulting to a base
`BatchPositionProcessor` instance, and replace every direct position operation
in the engine with a batch-of-one call through it: the terminal test in
`_mcts_iteration`, the legality and successor construction in
`_evaluate_and_expand_node`, and the two fallback paths that read
`root.position.legal_plies` (`_visit_distribution`'s zero-total branch,
`_select_best_ply` / `_select_best_ply_with_temperature`'s empty-children
branch). **No behaviour change** — this is a re-routing step; the wave that makes
the batches wider than one is #23.

Depends on: Step 1 (the class must exist).

Tests (Claude): the existing `tests/core/test_mcts_engine.py` suite must pass
untouched — that it needs no edits is the evidence that behaviour is unchanged.
Add a recording processor (a subclass counting calls per method) asserting the
engine reaches positions only through it: every call the engine makes appears in
the recorder, and a position whose scalar members raise still searches
successfully when a processor that does not consult them is supplied. That
second test is what pins the fallback paths, which are easy to miss.

Verification (automated): Run `pytest` and confirm the whole suite is green with
no test edits outside the new ones. Run `pyright`. Then confirm the engine still
plays: `python -m examples.tictactoe` against both the null-evaluator and
heuristic engines should behave exactly as before.

## Step 3 — `PositionEvaluator` → `evaluate_positions`

Replace `evaluate_position` with `evaluate_positions(positions)` returning one
`PositionEvaluation` per position, index-aligned. Update the three
implementations: `NullEvaluator` services the plural form itself (its uniform
policy construction moves inside a loop), `TicTacToeHeuristicEvaluator` wraps its
existing body in a comprehension, and `NeuralNetworkEvaluator` gets an interim
`evaluate_positions` that loops its current scalar path — enough to keep the
learning package green until Step 4 makes it genuinely batched. The engine's
call site in `_evaluate_and_expand_node` becomes a batch-of-one call.

Depends on: Step 2 only for sequencing — keeping the engine's re-routing diff
separate from its evaluator diff. Nothing here needs Step 1.

Tests (Claude): update `tests/core/test_null_evaluator.py` and
`examples/tictactoe/tests/test_heuristic_evaluator.py` to the plural surface,
adding an alignment assertion to each (N positions in, N evaluations out, in
order — not N copies of the first). The engine and integration suites should
otherwise pass unchanged.

Verification (automated): Run `pytest` for a green suite. Run `pyright` to
confirm no caller still reaches for `evaluate_position`.

## Step 4 — Make `NeuralNetworkEvaluator` genuinely batched

Replace the ABC's abstract surface with `encode_positions` and
`decode_policies`, and reimplement `evaluate_positions` as a single stacked
forward: stack the encodings into one `(N, *sample_shape)` tensor, run the model
once under `no_grad` in eval mode, split the value column into N floats, and hand
the policy rows to `decode_policies` alongside their positions. The
`unsqueeze(0)` / `squeeze(0)` batch-of-one dance and the interim loop from Step 3
both disappear. Update the two subclasses — `TicTacToeNNEvaluator` implements
`encode_positions` by genuinely stacking (the worked example of the pattern) and
`decode_policies` by masking per row; `NimNNEvaluator` stays a plain
comprehension for contrast. `SelfPlayCollector`'s `encode_position` call site
becomes batch-of-one here, since the scalar method is gone; the rest of the
collector's migration is Step 5.

Depends on: Step 3 (the plural evaluator surface it plugs into).

Tests (Claude): the equivalence test is the point of this step — a batched
evaluation of N positions must equal, elementwise, the N evaluations produced by
calling with one position at a time (values within float tolerance, policies key
for key). Extend `tests/learning/test_neural_network_evaluator.py` with it,
plus a decode alignment test that pairs each policy row with its own position
(a batch of positions with different legal sets, where a mispairing would surface
as a probability on an illegal ply). The existing eval-mode enforcement test must
survive: assert the model is put in eval mode on the batched path too. Update
`examples/tictactoe_learning/tests/test_nn_evaluator.py` to the plural surface.

Verification (manual): Run `pytest` for a green suite, then
`python -m examples.tictactoe_learning.selfplay` and confirm the reported policy
entropy and per-ply coverage match what Step 4 of #21 recorded (targets
sharpened but non-degenerate) — the batching must not change what self-play
produces. A short `python -m examples.tictactoe_learning.train` run should still
show loss decreasing on both heads.

## Step 5 — `SelfPlayCollector`

Make the collector's game touchpoints plural. `PolicyTransform` becomes
batch-shaped — N positions and N policies in, N re-keyed policies out, aligned by
index — and its docstring gains the same alignment contract the other seams
carry. The collector takes a `batch_ops` argument defaulting to a base processor
and uses it for the terminal test and ply application. The game loop stays
sequential and `collect` still plays one game at a time; turning the loop into a
fleet is #24. The per-step capture, the backward value assignment and the
returned `TrainingSample`s are unchanged in meaning.

Depends on: Step 1 (the processor) and Step 4 (`encode_positions`).

Tests (Claude): update `tests/learning/test_self_play_collector.py` for the new
transform signature, including a transform that asserts it receives N positions
and N policies paired by index and that returning them re-keyed lands on the
right samples. Keep the existing coverage of the identity default (no transform
supplied) and of the alternating value back-fill, which this step must not
disturb.

Verification (automated): Run `pytest tests/learning` and confirm green,
then `pytest` for the whole suite. Re-run
`python -m examples.tictactoe_learning.selfplay` and confirm the collected
sample count and value distribution are unchanged from Step 4.

## Step 6 — README and epic docs

Check `README.md` against the finished change. Three places are known stale: the
`game_engine_core.game` row in the "what's in the box" table (which now holds
`BatchPositionProcessor` as well as `StandardGame`), step 4 of "Implementing a
game" (`PositionEvaluator` is now a plural surface), and the
`game_engine_learning` paragraph, which names `encode_position` and
`decode_policy` directly. The batch seam and the "override selectively, and mind
the delegation direction" rule are the parts a consumer has to know about, and
warrant a short paragraph of their own. Then align the sibling story documents:
#20's wave phase list and #23/#24's bare `outcomes(...)` / `legal_plies(...)` /
`apply_plies(...)` references should name the `batch_ops` seam, and #24 should
record that `position_factory` stays scalar.

Depends on: Steps 1–5 (documents the settled surfaces).

Verification (manual): Run `/update-readme` (or review the branch diff against
`README.md`) and confirm the README describes the batch seam accurately, or is
verified as already accurate. Run `pytest` and `pyright` once more to confirm the
branch is green before closing the story.
