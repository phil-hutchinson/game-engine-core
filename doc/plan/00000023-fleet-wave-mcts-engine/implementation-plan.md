# Implementation Plan — Fleet / wave MCTS engine (issue #23)

Depends on #21 (true PUCT / full expansion) and #22 (batch-first protocols), both landed.

## Design decisions taken before this plan

These amend the story, which was written before the API split was settled. Step 7
records them in `story.md`.

1. **Mass play exists for training only.** `select_ply` is the *play* surface — the
   `GameEngine` protocol method, single-game, retaining its tree across plies
   exactly as it does today. `select_plies_for_training` is the *training* surface —
   the fleet, bare roots on every call.
2. **No plural play form and no singular training form.** The story's
   `select_plies` is dropped (no consumer — #24's collector needs only the
   training form), and `select_ply_with_policy` is deleted rather than retained as
   an `N = 1` wrapper.
3. **One class, not two.** The play/training difference is confined to root
   provenance and return shape; everything else — node type, iteration, descent,
   expansion, backpropagation, ply choice — is shared. Splitting would mean a base
   class holding almost all of the code plus two thin leaves, on the one axis
   (retention) that #30 leaves explicitly open.
4. **`_root_node` stays.** The story's "hold N trees instead of a single
   `_root_node`" is superseded: the retained root remains for play, and the fleet's
   N roots are call-scoped locals with no fleet state on the engine.

## Starting state

Step 1 is a retrofit — most of its engine work is already in the working tree.
Currently `ruff check .` passes, `pyright` reports 21 errors (15 in
`tests/core/test_mcts_engine.py`, 4 in `game_engine_learning/self_play_collector.py`,
2 in `tests/core/test_mcts_engine_batch_ops.py`), and `pytest` is at 19 failed /
197 passed. The `examples` suite is green at 102 passed and needs no API change,
since it only ever calls `select_ply`.

---

### Step 1 — Land the fleet iteration and restore the engine tests

Finish the engine-side retrofit in `MCTSEngine`: the per-iteration method takes N
roots and performs one iteration for every tree — select a leaf per tree, partition
the leaves on the batched outcome call, evaluate the non-terminal ones in a single
call, scatter their values back to their slots, and backpropagate every tree. Name
it in the singular (it performs *one* iteration across the fleet; the plural belongs
to the tree-growing loop) and name its parameter for the sequence it receives.
Bring the public docstring on the training method in line with what it now does:
one result per input position aligned by slot, N trees searched in lockstep so each
iteration is a single batched evaluation, bare roots per call with nothing retained,
and why (#30). Comment the leaf partition and value scatter, which are the subtlest
lines in a module that comments heavily everywhere else — a reader needs to know
which structure maps a crunched result back to its slot, and that every slot's value
is provably written. Fix the docstrings of the private methods this step leaves in
final shape, including the slot-order guarantee from leaf selection that everything
downstream depends on and nothing currently states; the methods Steps 4 and 5
restructure keep their docstrings until then.

Then repair the two engine test files without adding coverage: the call sites that
still pass a scalar position to the plural API and unpack a scalar result, and the
six tests that reach for the pre-retrofit private names.

Also point `SelfPlayCollector`'s per-ply search at the training method at width one,
leaving its sequential game loop otherwise intact. This was originally Step 2's
work, but deleting `select_ply_with_policy` leaves the collector unbuildable, so no
earlier step can reach a green suite without it. It stays a minimal adaptation rather
than the fleet driver: turning the loop into a fleet is #24's story, and doing it
here would pull that scope forward.

Depends on: nothing — #21 and #22 are landed.

Verification (automated): `pytest` green across the whole repo and `pyright` clean —
216 passed and 0 errors, from 19 failed / 197 passed and 21 errors. `pytest examples`
on its own still passes 102; it exercises the untouched `select_ply` path, so it is
the regression check that the play surface really was preserved.

---

### Step 2 — Verify the collector path end to end

Step 1 leaves the collector's unit tests green, which shows the seam type-checks and
returns the right shape but not that self-play still trains. Exercise the real path.

Depends on: Step 1 (which contains the collector change itself).

Verification (manual): run the tic-tac-toe learning self-play example end to end and
confirm it produces training samples and completes a training iteration as before.
The scripts under `examples/tictactoe_learning` are the only coverage of this path
outside the unit tests, so exercising them is what confirms the seam actually works
rather than merely satisfying its types.

---

### Step 3 — Fill the fleet test gaps

Add the coverage the retrofit created and Step 1 deliberately did not write. The
gaps, each of which is a claim the story makes and nothing currently checks:

- One evaluator call per iteration regardless of N — the story's central
  performance claim, and the reason the wave exists at all.
- Slot alignment: the result at index *i* belongs to the position at index *i*.
  Needs a fleet of genuinely different positions, so a misalignment cannot pass by
  coincidence.
- Terminal leaves leave the batch but still advance their game's iteration: the
  batch narrows while every tree in the fleet still gains a visit.
- A wave in which every selected leaf is terminal never reaches the evaluator.
- Slot independence: no tree leaks into another, including when two slots start
  from equal positions.
- `N = 1` through the training method agrees with the play path on the same
  position and budget, and the training method retains nothing between successive
  calls — the pair of behaviours that distinguish the two surfaces.
- An empty fleet returns an empty result.

Depends on: Steps 1 and 2 (the suite must be green first, or new failures cannot be
distinguished from unrepaired ones).

Verification (automated): `pytest tests/core` green with the new cases, and each new
case demonstrably fails when its behaviour is inverted — the call-count and
alignment tests in particular are worthless if they pass against a scalar
implementation, so check that they do not.

---

### Step 4 — Separate evaluation from expansion

Split the combined evaluate-and-expand method. Evaluation and the value scatter
belong to the iteration, which already owns the terminal partition and the empty-batch
guard; what remains is an expansion step handed leaves already paired with their
evaluations. This is the story's "the evaluation side effect becomes an explicit
scatter" made structural rather than a comment, and it leaves the expansion step
small enough for Step 5 to land in.

No behaviour change: the same evaluator call with the same batch, the same children
in the same order.

Depends on: Step 3 (the call-count and alignment tests are what make a
"no behaviour change" claim checkable; without them this is a refactor verified only
by the suite not noticing).

Verification (automated): `pytest` green with no test changes. Any test that needs
editing to accommodate this step is a signal the refactor changed behaviour — treat
it as a finding, not as an edit.

---

### Step 5 — Widen expansion's successor construction

Replace the per-child successor call with a single one spanning every expanding leaf
in the fleet. This is the last width-1 call on the fleet path and the one story scope
item still outstanding: it was #22's peer-review item 3, explicitly deferred to this
story.

It also strengthens an existing guarantee. Priors must all resolve before any
successor is built, which upgrades the incomplete-policy contract from "this node is
not left half-expanded" to "no node in the batch is expanded at all" — a contract
change worth its own test, and one that could not have been written before this step.

Depends on: Step 4 (lands in the extracted expansion step).

Verification (automated): a test counting successor-construction calls through the
existing spy processor in `tests/core/test_mcts_engine_batch_ops.py` shows one call
per iteration rather than one per child, and a test confirms an incomplete policy for
one slot leaves every tree in the fleet unexpanded. Full suite green.

---

### Step 6 — Reconcile the documentation

Amend `story.md` with the four decisions recorded at the top of this plan, and check
`README.md` against the story's changes — the engine gained a public method and the
self-play collector changed how it drives search, both of which the README may
describe. `/update-readme` reviews the branch diff and updates it if warranted.
Also confirm the private docstrings left stale by the retrofit now match their
methods, particularly the slot-order guarantee from leaf selection, which is
load-bearing for every step downstream of it and currently stated nowhere.

Depends on: Steps 1–5 (the documentation describes the finished shape).

Also record the decision on the remaining width-1 calls: the zero-visit fallbacks in
the visit-distribution and ply-selection helpers still request legal plies one
position at a time, so a fleet of N makes N width-1 calls at the end of a search.
They fire only when the budget cannot descend past a root, so leaving them is
defensible — but the reasoning belongs in a docstring rather than in this plan.

Verification (manual): read `story.md` and `README.md` against the branch diff and
confirm nothing describes the pre-split API or the single-tree iteration. `pytest`,
`ruff check .` and `pyright` all clean as a final gate.
