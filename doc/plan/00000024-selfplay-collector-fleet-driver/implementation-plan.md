# Implementation Plan — SelfPlayCollector as fleet driver (issue #24)

Depends on #22 (batch-first protocols) and #23 (fleet / wave MCTS engine), both landed.
The engine already exposes `select_plies_for_training(positions)` and the collector
already calls it — at width one, inside a still-sequential game loop that #23 left in
place deliberately. This story turns that loop into the fleet.

## Design decisions taken before this plan

These settle questions the story leaves open. Step 7 records them in `story.md`.

1. **One engine per `collect`, not per game.** The story's fleet hands *the* engine every
   live game's position at once, so a per-game engine has no meaning. `engine_factory`
   keeps its signature and is called once per `collect` instead of once per game. This is
   safe because the training surface retains nothing between calls and holds no fleet
   state (#23 decision 4: the fleet's N roots are call-scoped locals). It is also
   *required*: lockstep depends on every slot sharing one iteration budget, which a
   per-game engine could not guarantee. `observe_ply` and `reset` remain uncalled by the
   collector, as today.
2. **Samples are returned in slot order, not completion order.** Games retire at
   different turns, so the natural emission order is "shortest game first". Instead each
   slot accumulates its own samples and the buckets are concatenated in slot order when
   the fleet drains. This keeps `collect`'s observable output byte-for-byte identical to
   the sequential driver for a deterministic fleet, which is what makes Step 3's
   equivalence check possible at all. It refines the story's step 4 ("emit its
   `TrainingSample`s") to mean *emit into the slot's bucket*; the per-game back-fill
   still happens at retirement, while the outcome is in hand.
3. **One `outcomes` call per fleet-turn, at the top of the turn.** The sequential loop
   calls `outcomes` twice per game (the loop condition, then again for the terminal
   outcome). Retiring at the top of the turn from a single batched call serves both: the
   value that retires a slot is the value its back-fill needs. It also enforces the
   engine's precondition — `select_plies_for_training` raises on a terminal slot and
   forfeits the whole fleet's search (#23 peer review, finding 8) — and it makes a game
   whose *starting* position is already terminal fall out correctly, emitting nothing.
4. **The fleet is the live set; the batch is dense and compacts as games retire.** Game
   identity is fixed for the game's life — each slot owns its position, step records and
   sample bucket — but its *position in the batch* shrinks as earlier slots retire. The
   batch cannot carry holes: every seam on this path is typed `Sequence[TPosition]` with
   no optional, and a terminal slot makes the engine raise and forfeit the whole fleet's
   search (#23 peer review, finding 8). Compaction is an order-preserving filter, so the
   story's "positions go in in slot order" still holds — the live set stays ascending in
   original slot, merely with gaps. A retired slot never re-enters (no midstream refill;
   epic P4), which is what makes the long tail permanent.

## Contracts on caller-supplied collaborators

Two obligations this story places on externals, to be stated in docstrings rather than
left implied (Step 1 and Step 7 carry the edits):

- **The engine must retain no per-game state on the training path.** `MCTSEngine`
  satisfies this by construction (call-scoped roots; `_root_node` belongs to `select_ply`,
  which the collector never calls), but `engine_factory` is typed on the concrete class,
  so a subclass relying on per-game construction to clear state would now carry it across
  the fleet. Per-game engine *configuration* is likewise no longer possible — and is
  incompatible with lockstep anyway, which needs one shared iteration budget.
- **The policy transform receives one batch spanning several different games.** It has
  only ever seen width-one batches from a single game; correct index-aligned transforms
  are unaffected, but one that assumed a call's positions all came from the same game
  would break. The evaluator and `batch_ops` need no new guarantee: both are already
  shared instances, and #23's wave already hands the evaluator leaves from N games in one
  call.

## Starting state

`pytest`, `ruff check .` and `pyright` are all clean on the branch point (f032086).
`SelfPlayCollector.collect` is `for _ in range(n_games): self._play_game()`, and
`_play_game` drives `select_plies_for_training` at width one with a comment naming this
story. `tests/learning/test_self_play_collector.py` has 7 tests, two of which assert the
policy transform is called with a width-one batch — those assertions are pinning the
current driver, not the contract, and Step 3 rewrites them.

---

### Step 1 — Hoist the engine to fleet scope

Call `engine_factory` once per `collect` and hand the resulting engine to the per-game
play, instead of constructing one engine per game. Update the constructor docstring for
`engine_factory` to say it is called once per `collect` call and that every game in the
fleet shares that engine's iteration budget and temperature.

Nothing else changes: the loop stays sequential and each game is still played to
completion.

Depends on: nothing. It comes first because the fleet has exactly one engine, and doing
this while the loop is still sequential isolates the change from the restructure.

Verification (automated): `pytest tests/learning/test_self_play_collector.py` green with
no test edits, plus a new test that counts factory invocations and confirms
`collect(n_games=3)` calls it once. Invert it by restoring the per-game call to confirm
the test fails at 3.

---

### Step 2 — Restructure the game loop as a fleet turn loop, driven at N = 1

Rewrite the driver into the story's fleet loop — a live set of slots, each holding its
current position and its step records; per turn, one batched terminal test that retires
finished slots (back-filling and bucketing their samples), one `select_plies_for_training`
call over the live positions, one batched capture, one batched ply application — but have
`collect` still enter it once per game, with a fleet of one.

This is the step that carries the real risk (retirement mid-turn, slot alignment,
back-fill for a slot that leaves at an arbitrary turn), and running it at N = 1 means the
existing tests are a complete oracle for it: every observable must be unchanged.

Depends on: Step 1 (the loop is handed an engine rather than making one).

Verification (automated): the full `tests/learning` suite green **with no test edits**,
including the two transform tests that assert a width-one batch — at N = 1 the widths
really are one, so any edit needed here is a signal the restructure changed behaviour
rather than shape. `pyright` and `ruff check .` clean.

---

### Step 3 — Bootstrap the whole fleet

Change `collect` to build all N slots up front — N calls to the scalar position factory —
and run them through the turn loop as a single fleet, returning the slot-ordered
concatenation of the buckets. The per-turn calls widen from one to the live count with no
further change to the loop written in Step 2.

Two existing transform tests assert a width-one batch; rewrite them to assert the batch
width is the live-game count and that index *i* of the transform's input pairs with slot
*i*'s sample, which is the contract those tests were reaching for.

New coverage, each a claim this step makes and nothing yet checks:

- A fleet of N deterministic games returns exactly what N sequential `collect(1)` calls
  returned — same samples, same order (decision 2).
- Games of different lengths, via a position factory that varies its starting position
  per call: every slot's values and encodings are correct despite retiring at different
  turns, and the shorter games do not disturb the longer ones.
- Slot independence where two slots start from equal positions.
- Positions go into the engine in live-slot order and results come back aligned — needs
  slots that are genuinely distinguishable, so a misalignment cannot pass by coincidence.

Depends on: Step 2 (the loop exists and is proven at width one; this only widens the
entry point).

Verification (automated): `pytest tests/learning` green with the new cases. The
equivalence case is the load-bearing one — check it fails if the buckets are concatenated
in completion order rather than slot order, otherwise it is not testing what it claims.

---

### Step 4 — Pin the per-turn call widths

Add the coverage for the story's actual purpose: that a fleet-turn makes *one* call of
width N through each seam rather than N calls of width one. Using a recording evaluator,
transform and batch processor (the `_RecordingBatchProcessor` pattern in
`tests/core/test_mcts_engine_batch_ops.py` is the model), assert that per turn there is
exactly one `select_plies_for_training`, one `encode_positions`, one policy-transform,
one `outcomes` and one `apply_plies` call, each of width equal to the live-game count —
and that the width shrinks as games retire.

This is separate from Step 3 because Step 3's tests pass just as well against a driver
that loops the fleet one game at a time; correctness and batching are independent claims.

Depends on: Step 3 (the fleet must actually be N wide before its width can be asserted).

Verification (automated): `pytest tests/learning` green, and each new assertion
demonstrably fails against a width-one driver — confirm by temporarily bootstrapping the
fleet one game at a time.

---

### Step 5 — Fleet membership edges

Cover the boundaries decision 3 and decision 4 create: `collect(0)` returns an empty list
and never enters the turn loop; a game whose starting position is already terminal
retires on the first turn and contributes no samples, without reaching the engine — which
is what keeps the engine's non-terminal precondition satisfied; and a fleet mixing such a
game with live ones still returns the live games' samples in slot order.

Depends on: Step 4 (the recording seams it uses are in place by then).

Verification (automated): `pytest tests/learning` green with the new cases. The
terminal-at-bootstrap case must assert the engine was not called for that slot — if it
were, `select_plies_for_training` would raise and forfeit the whole fleet.

---

### Step 6 — Exercise the real self-play path

The unit tests use Nim with a null-ish evaluator; nothing so far shows the fleet produces
usable training data against a real network. Run the tic-tac-toe learning example end to
end: the self-play diagnostic, then a training run.

Also take a wall-clock reading against the branch point on the same game count and
iteration budget. The story's whole justification is the batched forward pass, and this is
the first point at which that is measurable end to end. A speedup is expected but not a
gate — the example is CPU-bound and small, and the epic's thesis is stated for a GPU
forward; a *regression* here is a finding.

Depends on: Steps 1–5 (the driver is complete and unit-verified).

Verification (manual): `python -m examples.tictactoe_learning.selfplay` produces the same
kind of summary as before — sample count matching the games played, a sane win/draw/loss
split, mean policy entropy below the uniform baseline — and `python -m
examples.tictactoe_learning.train` completes an iteration with a decreasing loss history.
`pytest examples` green. Record the before/after timing of the selfplay run in the peer
review or commit message.

**Result.** `selfplay` (20 games, 200 MCTS iterations, CPU): 142 samples, 45/63/34
win/draw/loss, mean policy entropy 1.760 bits against a 2.361-bit uniform baseline —
the same shape as the sequential driver's 154 samples and 44/72/38. `train --iterations 3
--games 8 --epochs 5 --mcts-iterations 100` completed every iteration with loss falling
within each. `pytest examples` 102 passed.

Wall-clock over three runs each of `selfplay`, fleet against the branch point f032086:
**3.31 / 3.43 / 3.50 s** versus **4.80 / 4.96 / 5.05 s** — about 1.45× faster. Worth
noting this is the *unfavourable* case for the story's thesis: a tiny MLP on CPU, where a
batched forward is nowhere near flat in cost. The win here comes mostly from amortising
per-call overhead rather than from GPU saturation, so it is a floor on what the change is
worth, not a ceiling.

---

### Step 7 — Reconcile the documentation

Update the `SelfPlayCollector` class docstring: the "game loop stays sequential ... called
batch-of-one throughout" paragraph is now false, and what replaces it is the fleet
contract — N games in lockstep, slot identity, the shrinking live set, and the long tail
the skateboard accepts. Give `collect` a docstring stating the return order (decision 2)
and that all N games are in flight at once. Remove the `#24` comment in the loop.

Amend `story.md` with the four decisions above. Then check `README.md` against the
story's changes — it describes `SelfPlayCollector` reaching positions through
`batch_ops`, which is still true but now at width N, and it may describe self-play as
game-at-a-time elsewhere. `/update-readme` reviews the branch diff and updates it if
warranted.

Depends on: Steps 1–6 (the documentation describes the finished shape).

Verification (manual): read `story.md`, `README.md` and the collector docstrings against
the branch diff and confirm nothing still describes a sequential, batch-of-one driver.
`pytest`, `ruff check .` and `pyright` all clean as a final gate.
