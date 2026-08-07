# Implementation Plan: Vectorised node representation (issue #26)

Turns the tree's child representation inside out: statistics move from per-child
`MCTSNode` objects onto parallel arrays held by the parent, PUCT selection
becomes a single array computation, and child objects are materialised only when
the search first descends into their slot.

## Shape of the change

Everything happens inside `game_engine_core/engines/mcts_engine.py`. The fleet
loop (`_mcts_iteration`), the batched protocols (`BatchPositionProcessor`), and
the public signatures of `select_ply` / `select_plies_for_training` are
untouched. Search *results* must not move: the plan is sequenced so that every
step up to the last is a representation change with identical behaviour, which
is what makes the existing suite a usable regression net throughout.

Two structural decisions the steps assume:

- **The parent's arrays are the single copy of a child's statistics.** A
  materialised child addresses them through its own (parent, slot) pair rather
  than holding duplicates. Mirroring the values on both the node and the parent
  array would be simpler to write and would drift the first time a code path
  updated only one of them.
- **Materialisation is always the last act of a descent.** A newly materialised
  child has no children of its own, so it is the leaf that descent was looking
  for. That means each tree needs at most one new successor per iteration, which
  is what makes the deferred, fleet-batched materialisation in Step 5 possible
  at all.

## Risks to watch

**Small branching factors may get slower.** numpy's per-call overhead dominates
below roughly 20–50 elements, and the `NimPosition` fixture branches at 2. A
vectorised score over two slots will lose to the Python loop it replaces. The
win is real only at realistic branching, which is why Step 1 establishes a
benchmark before anything changes. Step 6 measures and records what happened; it
deliberately does not act on the result. What to do about a low-branching
regression — accept it, threshold it, or vectorise across the fleet instead — is
left open for a decision on the numbers.

**The fleet is the better vectorisation axis, and this story does not take it.**
Scoring all N trees' current nodes in one array op would amortise numpy overhead
across the fleet the way the forward pass already is. It is out of scope here
because the trees sit at different depths with different branching, so the
batch is ragged and needs padding or a flattened layout — a larger change than
this story. Worth recording as a follow-up if Step 6 shows the per-node op is
still overhead-bound.

**Deferred materialisation costs `select_ply` (N = 1) something for nothing.**
Play carries the extra descent phase and gets no batching from it. The
alternative — materialising eagerly at width one during descent — would mean two
descent paths to keep in agreement, which is worse than the unused phase.

---

### Step 1 — Take numpy as a core dependency and establish the baseline

Add `numpy` to `game_engine_core`'s runtime dependencies in `pyproject.toml`
(the package has none today; numpy and torch are currently under the `learning`
extra only). Note in the dependency comment why core now needs it.

Add a search micro-benchmark under `doc/plan/00000026-vectorized-node-representation/`
or a scratch script — it is a measuring instrument for this story, not a
deliverable — that runs a fixed-seed search at a fixed iteration budget over
both a narrow position (the Nim fixture, branching 2) and a wide one (the
TicTacToe example, branching up to 9), for a fleet of 1 and a fleet of 64, and
reports wall-clock and iterations/sec for each.

Depends on: nothing. It comes first so every later step has a before-number to
compare against, and so the dependency is in place before any code imports it.

Verification (manual): Run the benchmark and confirm it prints stable numbers
across two consecutive runs (within a few percent). Record them in the plan or
the peer review as the baseline. Run `pytest`, `pyright`, and `ruff check .` and
confirm all three are clean — the dependency addition alone must change nothing.

---

### Step 2 — Move child statistics onto the parent as slot-indexed arrays

Restructure `MCTSNode` so an expanded node holds its legal plies in slot order
alongside parallel arrays of child priors, visit counts, and total values, and
so a materialised child knows the parent and slot that address its own
statistics. The per-child scalar `prior`, `visits`, and `total_value` fields go
away as independent storage; a node's own visit count and average value are read
through its slot in its parent (a root keeps its own scalars, having no parent).

Every reader of `node.children` and `child.prior` moves to slot addressing:
`_select_leaves`, `_backpropagate`, `_expand_leaves`, `_visit_distribution`,
`_select_best_ply`, `_select_best_ply_with_temperature`, and `observe_ply`. The
`str(ply)` keying survives in exactly two places — consuming
`PositionEvaluation.policy` at expansion (mapping policy keys to slots), and the
`dict[str, float]` that `select_plies_for_training` returns, which is a public
contract and does not change. `observe_ply` matches its ply against the root's
slot plies rather than scanning child objects.

Children are still materialised eagerly at expansion, and PUCT selection is still
a scalar loop — now over slot indices reading from the arrays rather than over
child objects. Nothing about search behaviour changes, including tie-breaks:
scanning slots in order and keeping the first maximum reproduces what `max()`
over `children` does today.

Depends on: Step 1 (numpy available for the array storage). Steps 3, 4 and 5 all
build on this layout; nothing else in the story is expressible until the
statistics live on the parent.

The existing tests that construct `MCTSNode` directly or assert on
`root.children` / `child.prior` (`test_mcts_engine.py` around the backpropagation,
prior-ranking, and re-rooting tests, plus the fleet test asserting slots get
distinct child objects) need rewriting against the new layout. Rewrite them to
assert the same *facts* through the new accessors — do not weaken an assertion to
make it pass.

Verification (automated): Run `pytest tests/core` and confirm the full MCTS suite
passes, including the sign-alternation, prior-ranking, zero-visit-distribution,
incomplete-policy, and re-rooting tests. Then run `pytest` for the example suites
and confirm the TicTacToe integration test still passes. `pyright` and
`ruff check .` clean.

---

### Step 3 — Vectorise PUCT selection

Replace the scalar slot loop in `_select_leaves` with a single array
computation over the parent's arrays: an exploitation vector from total values
and visits, an exploration vector from priors, the parent's visit count and the
exploration constant, then `argmax` over their sum. Guard the zero-visit
division so an unvisited slot scores as it does today (exploitation 0, full
exploration term) rather than producing a NaN that would silently poison the
argmax.

`argmax` returns the first maximal index, matching the first-wins tie-break that
Step 2 preserved, so the selected slot is identical for identical statistics.

Depends on: Step 2 (the arrays it operates on). Step 5's batching does not need
this, but the story's stated win does.

Verification (automated): Add a test that drives randomised prior/visit/value
combinations — including all-zero visits, a single visited slot, and exact ties —
through both the vectorised selection and a scalar reference computation in the
test, and asserts the chosen slot agrees on every case. Then run `pytest` and
confirm the existing suite is unchanged, particularly
`test_a_dominant_prior_is_reselected_while_its_sibling_stays_unvisited` and
`test_zero_visit_root_children_are_ranked_by_prior`, which pin the exact
selection behaviour.

Verification (manual): Re-run the Step 1 benchmark and record the numbers. Expect
the wide position to improve and the narrow one possibly to regress — that is the
small-array risk, and this is the measurement that decides whether it needs a
threshold.

---

### Step 4 — Materialise child nodes lazily

Stop building successor positions at expansion. `_expand_leaves` keeps its
`legal_plies` call and its policy-to-priors mapping, but no longer calls
`apply_plies`; a node becomes expanded when it has a priors array, whether or not
any child object exists. Descent materialises: when selection picks a slot whose
child object is absent, build that successor position, create the node, and stop
— it is the leaf.

`observe_ply` must now handle re-rooting onto a slot that was never descended
into. It has the parent position and the slot's ply, so it can materialise the
new root on demand rather than falling back to discarding the tree; keep the
existing discard path for a ply the root has no slot for.

Expansion stays all-or-nothing on a missing policy entry: priors still resolve
for the whole batch before any node is marked expanded.

Successor construction here is at width one, per tree — Step 5 batches it. This
step is about correctness, and separating it from the batching keeps the two
verifiable independently.

Depends on: Step 2 (slot addressing is what makes a slot addressable without a
child object) and Step 3 (selection must already return a slot rather than a
child). Step 5 batches what this step defers.

Verification (automated): Add tests asserting (a) the first iteration against a
root leaves the node expanded with priors for every legal ply but with no child
objects materialised, (b) a slot that PUCT never selects across a full budget
never has a successor position built — assert against a counting
`BatchPositionProcessor` that `apply_plies` was never called with that ply, and
(c) `observe_ply` onto a legal-but-unvisited ply re-roots rather than discarding
the tree. Run `pytest` and confirm search results are unchanged — the fleet
tests' evaluator-call-count assertions are the sharpest check that descent did
not change shape.

---

### Step 5 — Batch deferred materialisations across the fleet

Restructure `_select_leaves` so descents defer their materialisation instead of
performing it inline: each tree descends to either a leaf or a pending
(parent, slot) materialisation, the fleet's pending materialisations are issued
as one `apply_plies` call, and the resulting nodes are scattered back to their
trees as that iteration's leaves. Since materialisation is always the last act
of a descent, one call per wave is enough — no interleaving by depth is needed.

Slot ordering stays load-bearing exactly as it is today: the leaves must come
back in root order or `_mcts_iteration` will backpropagate into the wrong tree.

This is what returns `apply_plies` to a width bounded by the fleet size. The
comment in `_expand_leaves` naming this call as the one seam call running at
N x branching factor, and pointing at #26 as its fix, should be removed along
with the condition it describes.

Depends on: Step 4 (there is nothing to defer until materialisation happens
during descent).

Verification (automated): Add a fleet test using a counting
`BatchPositionProcessor` asserting that one wave issues at most one
`apply_plies` call, of width no greater than the number of live trees — and
none at all on a wave where every tree descends into an already-materialised
slot. Confirm the existing
`test_expansion_builds_every_successor_in_the_fleet_in_one_call` is rewritten
(not deleted) to assert the new location and width of that call. Run `pytest`.

Verification (manual): Re-run the benchmark at fleet 64 and confirm the wide
position improves against both the Step 1 baseline and the Step 3 numbers.

---

### Step 6 — Measure and record

Run the full benchmark and compare against the Step 1 baseline across all four
cells (narrow/wide x fleet 1/64). Record the numbers in a `benchmark.md`
alongside this plan, along with the machine they were taken on — they are the
justification for taking a runtime dependency on numpy in a package that had
none, and a future reader needs to be able to find them.

**This step does not act on the result.** Vectorised selection is expected to
lose at low branching factors, where numpy's fixed per-call overhead exceeds the
Python loop it replaces; the open question is where that crossover sits and
whether it sits anywhere that matters for a game we care about. Options if the
narrow cells regress — accept it as a known cost, add a slot-count threshold
below which selection stays scalar, or move to vectorising across the fleet
instead of within a node — differ enough in cost and permanence that the call
should be made on the numbers, and it is left open here deliberately.

Note that the Step 1 benchmark measures two branching factors (2 and up to 9),
which shows the *direction* at those widths but cannot locate a crossover. If
the number itself is wanted, the benchmark needs a synthetic position with a
configurable branching factor swept across a range — a change to Step 1, not
this one.

Also confirm the story's non-goal held: search results did not move. The suite
passing is most of that argument, but a direct check is worth it — run a
fixed-seed self-play game before and after the branch and confirm the ply
sequences are identical.

Depends on: Step 5 (the last change that affects performance).

Verification (manual): Present the before/after table and the fixed-seed ply
sequence comparison. A passing result is: the wide cells improved, the ply
sequences match exactly, and the narrow-cell result is recorded — whatever it
says — with the decision on what to do about it noted as open.

---

### Step 7 — README check

Verify `README.md` is still accurate. The likely-affected areas are the
installation instructions — `game_engine_core` now pulls a runtime dependency
where it previously had none — and anything describing `MCTSNode` or the tree
representation.

Depends on: Step 6 (the change is complete and measured).

Verification (manual): Run `/update-readme`, which reviews the branch diff and
updates `README.md` if warranted. Confirm either that it made a correct update or
that no update was needed.
