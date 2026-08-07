# Benchmark: vectorised node representation (issue #26)

Numbers from the measuring instruments beside this file. They are the
justification for taking a runtime dependency on numpy in a package that had
none, so they are recorded here rather than left in a terminal — and every table
below is reproducible from one of:

| | |
|---|---|
| `benchmark.py` | the four-cell search benchmark; the step-by-step tables |
| `selection_sweep.py` | the selection kernel isolated and swept across branching factors |
| `seam_call_counts.py` | how many calls the engine makes into consuming code, and how wide |
| `profile_cell.py` | where the time goes inside one cell |
| `ply_sequences.py` | fixed-seed self-play, for the non-goal check |

None of them is a deliverable and none is collected by pytest (`testpaths`
covers `tests` and `examples` only).

Read `benchmark.py`'s module docstring for what the columns mean. In short:
`wall` is the fastest repeat of a full `select_plies_for_training` call, and
`iters/s` is tree-iterations per second (`iterations x fleet / wall`) — the
figure to compare across steps, and the only one comparable across fleet sizes.
`signature` is a checksum of slot 0's search result; the story's non-goal is that
search results do not move, so it must stay identical for a given cell across
every step.

## Machine

All numbers below were taken on this machine. Timings from anywhere else are not
comparable — only ratios within a single machine are.

| | |
|---|---|
| CPU | 11th Gen Intel Core i7-11800H @ 2.30GHz, 16 threads |
| Memory | 16 GB |
| Platform | Linux 5.15.153.1-microsoft-standard-WSL2 (devcontainer on WSL2) |
| Python | 3.12.13 (GCC 12.2.0) |
| numpy | 2.5.1 |

The WSL2 devcontainer is a noisy host: individual repeats vary by tens of
percent. That is why the headline is the minimum rather than the mean — the
noise is one-sided, so the fastest repeat is the most stable estimate of the
work. The `spread` column is the check on that; a run whose spread is large in
*every* cell should be discarded rather than compared.

## Step 1 — baseline (pre-change)

Commit: `7f041fd` plus the uncommitted Step 1 dependency change. This is the
scalar, eagerly-expanded tree as it stands before any of this story's code
changes — numpy is a declared dependency at this point but nothing imports it.

`iterations=800 seed=20260730`

| cell | fleet | repeats | wall (s) | median | spread | iters/s | signature |
|---|---:|---:|---:|---:|---:|---:|---|
| narrow (nim) | 1 | 15 | 0.0243 | 0.0260 | 11.6% | 32,913 | `dc8e8031` |
| narrow (nim) | 64 | 3 | 1.6392 | 1.6798 | 7.6% | 31,235 | `efd7b510` |
| wide (tictactoe) | 1 | 15 | 0.0445 | 0.0468 | 8.5% | 17,989 | `509ce670` |
| wide (tictactoe) | 64 | 3 | 3.1930 | 3.3390 | 4.9% | 16,035 | `8f8f1ab7` |

Reproducibility check: a second consecutive run agreed to 4.1%, 1.7%, 0.7% and
3.3% respectively on `wall`, and matched all four signatures exactly.

Two things worth noting before the comparisons start:

- **Throughput barely falls from fleet 1 to fleet 64** (−5% narrow, −11% wide).
  The fleet costs almost nothing per tree, which is the batching from #22/#23
  doing its job — and it means any change this story makes to per-node cost
  should show up at both fleet sizes rather than being masked by fleet overhead.
- **The wide position is already ~1.8x more expensive per iteration than the
  narrow one** (18k vs 33k iters/s), at a branching factor of at most 9 against
  2. That gap is the eager-expansion and scalar-descent cost the story is aimed
  at, and it is the cell where a win is expected.

## Step 2 — statistics moved onto the parent as slot-indexed arrays

Still a scalar selection loop, now reading from the arrays by slot instead of
from child objects. Nothing about search behaviour changed and the signatures
confirm it.

| cell | fleet | wall (s) | iters/s | vs baseline | signature |
|---|---:|---:|---:|---:|---|
| narrow (nim) | 1 | 0.0376 | 21,292 | −35% | `dc8e8031` |
| narrow (nim) | 64 | 2.7342 | 18,726 | −40% | `efd7b510` |
| wide (tictactoe) | 1 | 0.0581 | 13,778 | −23% | `509ce670` |
| wide (tictactoe) | 64 | 4.3394 | 11,799 | −26% | `8f8f1ab7` |

**All four signatures are unchanged from the baseline.** This is the step that
moved every statistic to a new home, so it was the one most likely to move a
search result, and it did not.

The regression is expected and is the cost of the intermediate state, not of the
representation: a scalar loop now pays numpy's per-element indexing overhead on
every read where it previously did a Python attribute lookup, and gets none of
the vectorisation back until Step 3. A profile of the wide cell puts
`child_puct_value`, `child_average_value` and the `visits` property at 36% of
total runtime — precisely the code Step 3 replaces with a single array
computation. `record_visit` does not reach the top ten, so backpropagation's move
into the arrays is not a material cost.

One number worth carrying into Step 3: `visits` was called 98,424 times in that
profile, once per slot scored, to recompute a parent visit count that is constant
across the whole loop. Vectorising hoists it out by construction.

## Step 3 — vectorised PUCT selection

**Taken on a different machine from Steps 1–2.** None of the numbers below are
comparable to the ones above — only ratios within this section are meaningful.
Since the switch happened mid-story, Step 2 was re-run here first to give this
machine its own baseline; that re-run is what Step 3 is measured against, not
the Step 1/2 figures above.

### Machine

| | |
|---|---|
| CPU | AMD Ryzen 7 8700F, 8 cores / 16 threads |
| Memory | 8 GB |
| Platform | Linux 6.18.33.2-microsoft-standard-WSL2 (devcontainer on WSL2) |
| Python | 3.12.13 (GCC 12.2.0) |
| numpy | 2.5.1 |

### Step 2 baseline, re-measured on this machine

`iterations=800 seed=20260730`

| cell | fleet | repeats | wall (s) | median | spread | iters/s | signature |
|---|---:|---:|---:|---:|---:|---:|---|
| narrow (nim) | 1 | 15 | 0.0213 | 0.0219 | 12.1% | 37,634 | `dc8e8031` |
| narrow (nim) | 64 | 3 | 1.6547 | 1.6968 | 4.9% | 30,942 | `efd7b510` |
| wide (tictactoe) | 1 | 15 | 0.0327 | 0.0345 | 18.0% | 24,487 | `509ce670` |
| wide (tictactoe) | 64 | 3 | 2.6056 | 2.7428 | 6.1% | 19,650 | `8f8f1ab7` |

Reproducibility check: a second run agreed to within 3.1%, 2.7%, 3.0%, and 3.7%
on `iters/s` respectively, and matched all four signatures.

### Step 3

| cell | fleet | wall (s) | iters/s | vs baseline (this machine) | signature |
|---|---:|---:|---:|---:|---|
| narrow (nim) | 1 | 0.0585 | 13,672 | −64% | `dc8e8031` |
| narrow (nim) | 64 | 4.2962 | 11,917 | −61% | `efd7b510` |
| wide (tictactoe) | 1 | 0.0405 | 19,757 | −19% | `509ce670` |
| wide (tictactoe) | 64 | 3.1731 | 16,136 | −18% | `8f8f1ab7` |

Reproducibility check: a second run agreed to within 0.0%, 0.3%, 1.0%, and 4.0%
on `iters/s` respectively (the wide/fleet-64 cell ran noisier, 15% spread on
that repeat set, but the headline minimum still held). **All four signatures
are unchanged from the Step 2 baseline above and from the Steps 1–2 figures
taken on the old machine** — search results did not move.

### This does not match the plan's expectation

The plan's Step 3 section expected the wide cell to improve and left the narrow
cell's regression as the open small-array risk. What was measured instead is a
regression in **all four cells**, wide included — steeper on narrow (−61/−64%)
but still substantial on wide (−18/−19%).

A likely reason: `child_puct_values` is not one numpy call, it's about eight —
`zeros_like`, the guarded `divide`, the negation, and each step of building the
exploration term (`priors * constant`, `* sqrt`, `child_visits + 1`, the
division, the final addition) is its own array op, each paying numpy's per-call
dispatch cost. TicTacToe's branching factor (up to 9) sits well inside the
20–50-element range the plan's own risk note names as overhead-bound, so it is
plausible this cell was never going to clear that bar without fusing those ops
down — this measurement doesn't distinguish "vectorising per-node selection
doesn't pay off at these branching factors" from "this particular
implementation makes more numpy calls than it needs to."

Left for Step 6, per the plan: whether to accept this, add a slot-count
threshold, reduce the op count and re-measure, or reconsider per-node
vectorisation versus fleet-axis vectorisation. Not acted on here.

## Steps 4 and 5 — lazy materialisation, then deferred and batched across the fleet

Measured together on the Step 3 machine (AMD Ryzen 7 8700F). Step 4 stops
building successor positions at expansion and builds them during descent at
width one; Step 5 defers those materialisations to the end of the wave and
issues them as a single `apply_plies` call bounded by fleet size. Neither is
expected to move selection cost — they change what `apply_plies` is called with,
and Step 4 additionally stops building successors for slots the search never
visits.

`iterations=800 seed=20260730`

| cell | fleet | repeats | wall (s) | median | spread | iters/s | signature |
|---|---:|---:|---:|---:|---:|---:|---|
| narrow (nim) | 1 | 15 | 0.0575 | 0.0582 | 14.9% | 13,903 | `dc8e8031` |
| narrow (nim) | 64 | 3 | 4.2446 | 4.2827 | 1.9% | 12,062 | `efd7b510` |
| wide (tictactoe) | 1 | 15 | 0.0368 | 0.0380 | 19.3% | 21,724 | `509ce670` |
| wide (tictactoe) | 64 | 3 | 2.6332 | 2.6595 | 4.9% | 19,444 | `8f8f1ab7` |

Reproducibility check: a second consecutive run agreed to 0.8%, 1.6%, 1.9%, and
3.1% respectively on `iters/s`, and matched all four signatures exactly.

Against the Step 3 figures on this same machine, the wide cells recover and the
narrow cells do not:

| cell | Step 3 iters/s | Step 5 iters/s | vs Step 3 |
|---|---:|---:|---:|
| narrow (nim) | 13,672 | 13,903 | +2% |
| narrow (nim) fleet 64 | 11,917 | 12,062 | +1% |
| wide (tictactoe) | 19,757 | 21,724 | +10% |
| wide (tictactoe) fleet 64 | 16,136 | 19,444 | +21% |

That split is the shape the plan predicted for batching. Materialisation cost
scales with the branching factor, so the wide cell had more of it to save;
Nim's branching factor of 2 leaves almost nothing to recover, and its +1/+2% is
inside the noise.

What this does **not** show is a return to the Step 1 baseline — the Step 3
section's re-measured baseline is Step 2's, not Step 1's, and Step 2 had already
given up 27–43%. Step 6 measures Step 1 on this machine to close that gap.

## Step 6 — measure and record

Full before/after on one machine: the Step 1 commit (`5dcedf9`, the scalar
eagerly-expanded tree) against the tip of the branch (`51125b6`, Step 5), taken
back to back on the Ryzen 7 8700F described in the Step 3 section. The baseline
was run from a git worktree at `5dcedf9` with the editable install's import hook
disabled, so it exercises that commit's engine rather than the branch tip's.

### Step 1 baseline, on this machine

`iterations=800 seed=20260730`

| cell | fleet | repeats | wall (s) | median | spread | iters/s | signature |
|---|---:|---:|---:|---:|---:|---:|---|
| narrow (nim) | 1 | 15 | 0.0129 | 0.0134 | 26.5% | 61,959 | `dc8e8031` |
| narrow (nim) | 64 | 3 | 0.9363 | 0.9668 | 5.9% | 54,683 | `efd7b510` |
| wide (tictactoe) | 1 | 15 | 0.0229 | 0.0237 | 43.0% | 34,911 | `509ce670` |
| wide (tictactoe) | 64 | 3 | 1.8966 | 1.9071 | 4.6% | 26,996 | `8f8f1ab7` |

### Branch tip (Step 5), on this machine

| cell | fleet | repeats | wall (s) | median | spread | iters/s | signature |
|---|---:|---:|---:|---:|---:|---:|---|
| narrow (nim) | 1 | 15 | 0.0578 | 0.0616 | 19.9% | 13,844 | `dc8e8031` |
| narrow (nim) | 64 | 3 | 4.2162 | 4.2575 | 1.3% | 12,144 | `efd7b510` |
| wide (tictactoe) | 1 | 15 | 0.0362 | 0.0375 | 19.2% | 22,105 | `509ce670` |
| wide (tictactoe) | 64 | 3 | 2.6800 | 2.6909 | 2.0% | 19,105 | `8f8f1ab7` |

Reproducibility check: a second consecutive pair of runs agreed to within 4.9%
on every baseline cell and 3.5% on every branch-tip cell, and every cell's delta
below moved by at most three points. All eight signatures matched. The fleet-1
spreads are wide (26% and 43% on the baseline) because those repeats are
12–23 ms and a single scheduler event dominates one; the minimum is stable
across runs, which is what the spread column is there to let you check.

### The result

| cell | baseline iters/s | branch iters/s | change |
|---|---:|---:|---:|
| narrow (nim) fleet 1 | 61,959 | 13,844 | **−78%** |
| narrow (nim) fleet 64 | 54,683 | 12,144 | **−78%** |
| wide (tictactoe) fleet 1 | 34,911 | 22,105 | **−37%** |
| wide (tictactoe) fleet 64 | 26,996 | 19,105 | **−29%** |

**All four cells regress, wide included.** The plan's Step 3 expectation was
that the wide cells would improve and only the narrow ones were at risk; that is
not what happened, and Steps 4 and 5 recovered part of the wide regression
without closing it.

Per step, on this machine, so the cost can be attributed:

| cell | Step 1 | Step 2 | Step 3 | Step 5 |
|---|---:|---:|---:|---:|
| narrow (nim) fleet 1 | 61,959 | 37,634 (−39%) | 13,672 (−78%) | 13,844 (−78%) |
| narrow (nim) fleet 64 | 54,683 | 30,942 (−43%) | 11,917 (−78%) | 12,144 (−78%) |
| wide (tictactoe) fleet 1 | 34,911 | 24,487 (−30%) | 19,757 (−43%) | 22,105 (−37%) |
| wide (tictactoe) fleet 64 | 26,996 | 19,650 (−27%) | 16,136 (−40%) | 19,105 (−29%) |

Percentages are against Step 1. Step 4 was not measured on its own; Step 5's
column is the branch tip.

Two thirds of the loss is in place before a single array operation runs. Step 2
— scalar selection reading from numpy arrays by slot — costs 27–43% on its own,
and that is a floor the later steps inherit: every read of a statistic that used
to be a Python attribute lookup is now an element extraction from an ndarray,
which is several times more expensive at scalar width. Step 3 then doubles the
narrow loss and adds to the wide one. Steps 4 and 5 give back 3 points on narrow
and 11–14 on wide.

### Where the time goes now

`cProfile` on the fleet-64 cells at the branch tip, sorted by total time:

| | wide (tictactoe) | narrow (nim) |
|---|---:|---:|
| profiled wall | 4.33 s | 6.34 s |
| `child_puct_values` (cumulative) | 1.56 s — 36% | 3.53 s — 56% |
| `argmax` at the call site (cumulative) | 0.30 s — 7% | 0.66 s — 10% |
| **selection, total** | **43%** | **66%** |
| slots scored per tree-iteration | 4.35 | 9.97 |

Selection is the whole story, and the narrow cell is worse for two compounding
reasons rather than one. The obvious one is width: scoring 2 elements cannot
amortise numpy's per-call dispatch. The one the plan did not anticipate is
*depth* — Nim at pile 21 with takes of 1–2 builds a far deeper tree than
TicTacToe, so a single iteration calls `child_puct_values` 10 times against the
wide cell's 4.3. The narrow cell pays the fixed overhead more than twice as
often per iteration *and* has the least to gain from each payment.

`child_puct_values` is around nine array operations (`zeros_like`, the `!= 0`
mask, the guarded `divide`, the negation, and four more building the exploration
term), plus `argmax` at the call site, on an array of 2 to 9 elements.
`zeros_like` alone is 0.30 s of the wide cell and 0.70 s of the narrow one —
more than 10% of narrow's total runtime spent allocating two-element arrays.
This is consistent with the Step 3 section's guess and gives it a number: the
measurement still cannot separate "per-node vectorisation does not pay at these
branching factors" from "this implementation issues too many numpy calls", but
it does say the fixed per-call cost, not the arithmetic, is what is being paid.

### Where the crossover actually sits

The four cells measure branching factors of 2 and up to 9, which shows the
direction but cannot locate the crossover — the plan notes this and defers it as
a change to Step 1. It can be answered more cheaply by isolating the selection
kernel instead of building a synthetic position: no tree, no descent, no seam,
just "score every slot of one node and pick the best", which is the entire
difference between Steps 1, 2 and 3.

Five variants. A, B and C are transcribed from the commit they belong to; D and
E do not exist anywhere and were written to test what the first three imply.
All five are asserted to select the same slot for the same statistics before
being timed.

| | storage | algorithm | provenance |
|---|---|---|---|
| A | child objects, Python scalars | scalar loop | `5dcedf9` (Step 1) |
| B | ndarray, slot-indexed | scalar loop | `a89ee11` (Step 2) |
| C | ndarray, slot-indexed | vectorised | `328be36` (Step 3), byte-identical at the tip |
| D | Python list, slot-indexed | scalar loop | hypothetical |
| E | ndarray, slot-indexed | vectorised, fused | hypothetical |

Nanoseconds per node scored, per-cell minimum across four runs:

| width | A objects | B nparrays | C vectorised | D lists | E fused | D vs A | E vs C |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 641 | 1,112 | 5,590 | 648 | 3,659 | 0.99x | 1.53x |
| 4 | 1,028 | 1,913 | 5,600 | 1,051 | 3,688 | 0.98x | 1.52x |
| 9 | 1,934 | 3,870 | 5,657 | 1,992 | 3,704 | 0.97x | 1.53x |
| 16 | 3,235 | 6,753 | 5,737 | 3,358 | 3,711 | 0.96x | 1.55x |
| 25 | 4,918 | 10,276 | 5,716 | 5,044 | 3,762 | 0.98x | 1.52x |
| 50 | 9,356 | 20,174 | 5,858 | 9,842 | 3,844 | 0.95x | 1.52x |
| 100 | 18,790 | 39,264 | 6,008 | 19,198 | 3,887 | 0.98x | 1.55x |
| 200 | 36,413 | 77,282 | 6,697 | 37,512 | 4,213 | 0.97x | 1.59x |
| 400 | 72,684 | 151,114 | 7,693 | 75,623 | 4,663 | 0.96x | 1.65x |

The kernel is measuring what the search pays: C's 5.7 us here against the 7.0 us
per call the profile above attributes to `child_puct_values`, the gap being
profiler overhead. Multiplying by descent depth also reproduces the profiled
shares — 5,657 ns x 4.35 nodes = 24.6 us against a wide-cell iteration whose
selection profiles at 22.5 us, and 5,590 ns x 9.97 = 55.7 us against the narrow
cell's 54.3 us. That agreement is what licenses projecting these onto
throughput.

**C is flat, and that is the whole finding.** 5.59 us at width 2 and 7.69 us at
width 400 — 200x the work for 38% more time. Fitting the columns gives
A ~ 279 ns + 181 ns/slot against C ~ 5,590 ns + 5.3 ns/slot: essentially all of
C is fixed dispatch cost for its ten-ish array operations, and the arithmetic is
free. The crossover is that floor divided by A's slope, putting it at a
**branching factor of about 30**. So the vectorisation is not wrong; it has only
ever been measured in the regime where it cannot pay.

**B never crosses.** Scalar-over-arrays is 1.8-2.2x slower than
scalar-over-objects at every width tested, including 400, because an ndarray
element read is dearer than a Python attribute lookup and width does not change
that. Step 2 has no standalone value at any branching factor; it is worth only
what Step 3 makes of it.

**D says slot addressing is free.** D holds at 0.95-0.99x of A everywhere. It
keeps exactly the layout Steps 4 and 5 are written against — statistics on the
parent, addressed by slot — and differs from B only in the container. So the 2x
Step 2 paid was entirely ndarray element access, including the `int()`/`float()`
conversions needed to get Python numbers back out, which D does not perform.
Slot addressing and numpy are separable, and only the second one costs anything.

**E says the floor is reducible, but not by much.** Folding the scalar
coefficient into one multiply, replacing the `zeros_like` plus masked `divide`
with a clamped denominator, and accumulating in place gives a consistent 1.5x
and drops the floor from 5.59 us to 3.66 us — moving the crossover against A
from ~30 to **~19**. E's substitution is exact rather than approximate only
because `record_visit` increments a slot's visits and total value together, so
zero visits implies a total value of exactly 0.0; that invariant is currently
implicit and would need stating if E were adopted.

Caveats. One machine and one numpy build, and since the crossover is a
dispatch-to-arithmetic ratio it will move on other hardware. The harness models
the scored node as a root, so B, C and E are spared an array read for the parent
visit count that they pay at a real non-root — if anything it flatters them.
And this is the kernel alone: it is silent on the ~34-57% of runtime outside
selection, which includes `record_visit` and the `visits` property, both of
which also moved onto ndarrays in Step 2. The sweep therefore *understates* how
far apart A and C are in whole-search terms.

### What the story bought

The timings are the cost. The benefit is on the seam, and it does not show up in
a wall clock here because the fixtures' positions are nearly free to construct.
Every call the engine makes into consuming code, counted at both commits on the
wide cell at fleet 64:

| seam method | baseline calls | total width | max | tip calls | total width | max |
|---|---:|---:|---:|---:|---:|---:|
| `evaluate_positions` | 800 | 45,643 | 64 | 800 | 45,643 | 64 |
| `outcomes` | 800 | 51,200 | 64 | 800 | 51,200 | 64 |
| `legal_plies` | 800 | 45,643 | 64 | 800 | 45,643 | 64 |
| `apply_plies` | 800 | **220,572** | **576** | 799 | **48,611** | **64** |

Three of the four are identical — in particular the evaluator seam, the
expensive one behind a real network, is untouched by this story. `apply_plies`
moved in two independent ways:

**Its width fell from N x branching factor to N.** 576 is 64 trees x 9 legal
plies at an empty TicTacToe root; the tip's 64 is one successor per tree. This
is the bound #26 existed to establish, and it is what makes the call shape
predictable for an implementation that batches it into a single engine or
device call.

**Its total volume fell 4.5x**, from 220,572 positions constructed to 48,611.
That part is Step 4's laziness rather than Step 5's batching: the baseline built
every legal successor of every leaf whether or not the search descended into it,
and at 800 iterations against branching 9 most were never visited. On the narrow
cell the same figure is 88,718 to 48,646 — only 1.8x, since branching 2 leaves
little waste to eliminate. Both effects grow with the branching factor.

Step 5's own contribution is not visible as a reduction against the baseline,
because both are one call per wave. Its job was to preserve that: Step 4 alone
materialises inline during descent, which would mean up to N calls of width one
per wave. The three commits in sequence:

| | calls per wave | width |
|---|---|---|
| baseline | 1 | up to N x B |
| Step 4 | up to N | 1 |
| Step 5 (tip) | 1 | up to N |

The tip's 799 rather than 800 is the wave on which every tree descended into an
already-materialised slot and no call was needed at all.

### The non-goal held

All four benchmark signatures are identical between the baseline commit and the
branch tip, and identical to the Step 1 figures taken on the original machine.

The benchmark only exercises `select_plies_for_training`, which builds bare
roots and retains nothing, so it cannot catch a re-rooting regression — and
Step 4 changed `observe_ply`. A fixed-seed self-play game was therefore run
through the play path (`select_ply` + `observe_ply`, temperature 0, 200
iterations, one engine playing both sides) at both commits:

| game | plies | outcome |
|---|---|---|
| tictactoe | `8 6 7 9 3 5 1 4` | −1, three in a row |
| nim (pile 21) | `1 1 1 1 1 1 2 1 1 2 2 1 1 2 1 2` | −1, last token taken |

Both sequences are byte-identical before and after the branch. `pytest` (241
passed), `pyright` and `ruff check .` are clean at the tip.

### The decision

The plan specifies that Step 6 does not act, and it does not. The branch ships
as measured.

That is a decision on the numbers rather than a deferral by default. The options
the plan listed resolve as follows:

- **A slot-count threshold** is dead, and the sweep is what kills it. The scalar
  path a threshold would fall back to is B, roughly twice as slow as the code
  this branch replaced — below the threshold the engine would be worse than
  doing nothing. Getting A's numbers back under a threshold means scalar copies
  of the statistics alongside the arrays, the duplication the plan's first
  structural decision rejected on drift grounds.
- **Reverting Steps 2 and 3 while keeping 4 and 5** is possible, and D is how.
  Not as a revert — Steps 4 and 5 are written against slot addressing
  throughout, and Step 3 alone reverts to B — but the dependency is on holding a
  slot's statistics before its position exists, not on those statistics being
  ndarrays. D preserves slot addressing exactly, so Steps 4 and 5 are untouched
  and the change is confined to `MCTSNode`.
- **Fusing the operation** (E) is worth a measured 1.5x and moves the crossover
  to ~19. Real, but not a factor-of-N result, and it does not reach either
  example game's width.
- **Vectorising across the fleet** remains the strongest option structurally,
  and the sweep says why: the cost is a fixed per-call floor, so scoring the
  fleet's current nodes together pays it once per wave instead of once per node
  scored. That puts every game above the crossover regardless of its own
  branching factor. The ragged-batch problem the plan names as the reason it is
  out of scope is unchanged.

**Shipping as-is is chosen because the shipped cost is the bounded one.** C is
flat at ~5.6-7.7 us per node scored across a 200x range of widths; D is linear
and unbounded, reaching 75 us per node at width 400. Neither game in this repo
is evidence about a consuming game's branching factor — both exist to
demonstrate library usage — so the width that matters is unknown, and the option
whose worst case is capped is the one to hold while it stays unknown. The cost
of waiting is bounded by the same argument: at most ~3 us per node scored over
the best alternative.

Picking between D and E needs the branching factor *and depth* of a real
consuming game. Implementing one is what will supply them. The analysis, the
recommendation (E, on the bounding argument) and the trigger conditions are
written up in
[`doc/proposed_stories/reduce-puct-selection-cost.md`](../../proposed_stories/reduce-puct-selection-cost.md)
so this does not have to be rederived.

One thing the benchmark structurally cannot decide: whether the trade is worth
taking at all. The seam reduction above is worth whatever `apply_plies` costs
the consuming game, and here that is a board copy — near zero, so the fixtures
show all of the cost and none of the benefit. A game whose position transition
is expensive, or batched onto a device, pays the timings above out of a much
larger total and collects a 4.5x cut in transitions for it. The regression in
this document is measured against the most hostile possible downstream and
should be read as an upper bound on the harm.

## Sign-off

**Ship as measured.** Taken deliberately, with the regression in the four cells
above understood and accepted, on the reasoning already set out in this
document: the shipped cost is the bounded one, and both games measured here are
illustrative examples of library usage rather than evidence about a consuming
game's branching factor.

No whole-search measurement above the ~30 crossover backs this up, and none is
being manufactured to. `selection_sweep.py` isolates the *kernel*, so it is
silent on the 34–57% of runtime outside selection; producing a credible
whole-search number at a real width would mean building the synthetic
configurable-branching-and-depth position that
[`doc/proposed_stories/reduce-puct-selection-cost.md`](../../proposed_stories/reduce-puct-selection-cost.md)
lists as not-yet-existing. That is a fixture that would only ever approximate the
thing it stands in for, and the first real consuming game supplies the same
answer directly and for free.

**Reassess from a real game.** The trigger is the first game that consumes this
library at a known branching factor and depth — profile what share of search
time selection actually takes there, and read the result against the trigger
conditions in the proposed story (below ~19 favours D, above ~30 favours the
shipped kernel plus E, between the two favours E). Until such a game exists the
question is not answerable with anything better than the bounding argument, and
the cost of waiting is capped at roughly 3 µs per node scored.
