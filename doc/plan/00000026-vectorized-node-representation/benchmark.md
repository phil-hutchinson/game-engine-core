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
