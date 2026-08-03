# Benchmark: vectorised node representation (issue #26)

Numbers from `benchmark.py`, which lives beside this file. They are the
justification for taking a runtime dependency on numpy in a package that had
none, so they are recorded here rather than left in a terminal.

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
