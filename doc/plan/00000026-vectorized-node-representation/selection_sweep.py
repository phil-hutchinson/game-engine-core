"""PUCT selection kernel, swept across branching factors (issue #26, Step 6).

``benchmark.py`` measures whole searches at the two branching factors this repo
has games for (2 and up to 9). Those games exist to demonstrate library usage,
not to stand in for a consuming game, so their widths show the direction of a
change and nothing more. This script removes them from the picture: it isolates
the selection kernel — score every slot of one node, return the best — and
parameterises it on width, so the question "what does selection cost at
branching factor N" can be answered for any N.

Nothing else is timed. No tree, no descent, no seam, no expansion, no
backpropagation, no evaluation, no position construction. That is the point, but
it also bounds what the numbers mean: selection is 43% of a wide fleet-64
iteration and 66% of a narrow one (see ``profile_cell.py``), so this speaks to
roughly half to two-thirds of search time and is silent on the rest — including
``record_visit`` and the ``visits`` property, which also moved onto ndarrays in
Step 2.

Five variants. A, B and C are transcribed from the commit they belong to; D and
E exist nowhere and are here to test what the first three imply.

    A  scalar loop over child objects     5dcedf9 (Step 1, pre-story)
    B  scalar loop over ndarrays by slot  a89ee11 (Step 2)
    C  vectorised over ndarrays           328be36 (Step 3), byte-identical at tip
    D  scalar loop over lists by slot     hypothetical: slots without numpy
    E  vectorised, fused                  hypothetical: C with the op count cut

The variants differ on two axes, which is what lets each be attributed:

    A vs D   objects vs slot-indexed lists, both scalar  -> is slot addressing free?
    B vs D   ndarray vs list, both scalar by slot        -> what does numpy cost?
    B vs C   scalar vs vectorised, both ndarray          -> what does vectorising buy?
    C vs E   op count, both vectorised ndarray           -> is the floor reducible?

All five must select the same slot for the same statistics. That is asserted
before anything is timed, so a variant that got faster by computing something
else fails rather than posting a number.

Two things to know before reading the output. The node is modelled as a root, so
the parent visit count is a plain int attribute in every variant rather than an
array read — at a real non-root, B pays an ndarray read for it once per slot, so
this flatters B, C and E. And the crossover between a flat variant and a linear
one is a ratio of numpy's dispatch cost to interpreter overhead, so it will move
on other hardware and other numpy builds.

Run it from anywhere:

    python doc/plan/00000026-vectorized-node-representation/selection_sweep.py
"""

from __future__ import annotations

import math
import random
import timeit
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

EXPLORATION_CONSTANT = 1.41
WIDTHS = (2, 4, 9, 16, 25, 50, 100, 200, 400)
PARENT_VISITS = 400
SEED = 20260730

# Repeats per variant per width. The machine this was developed on is a noisy
# WSL2 devcontainer, and the noise is one-sided, so the headline is the minimum
# — same argument as benchmark.py's `wall` column.
REPEATS = 5


class ChildObject:
    """Step 1's child: its own scalars, its own parent pointer."""

    def __init__(self, parent: RootObject, prior: float, visits: int, total_value: float):
        self.parent = parent
        self.prior = prior
        self.visits = visits
        self.total_value = total_value

    @property
    def average_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def puct_value(self, exploration_constant: float = EXPLORATION_CONSTANT) -> float:
        exploitation = -self.average_value
        exploration = exploration_constant * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration


class RootObject:
    def __init__(self, visits: int):
        self.visits = visits
        self.children: list[ChildObject] = []


class ArrayNode:
    """Steps 2 and 3's node: its children's statistics as parallel ndarrays."""

    def __init__(self, visits: int, priors: list[float], child_visits: list[int], child_values: list[float]):
        self.visits = visits
        self.child_priors = np.array(priors, dtype=np.float64)
        self.child_visits = np.array(child_visits, dtype=np.int64)
        self.child_total_values = np.array(child_values, dtype=np.float64)
        self.child_count = len(priors)

    def child_average_value(self, slot: int) -> float:
        """Step 2. The int()/float() calls are the price of getting Python numbers back."""
        visits = int(self.child_visits[slot])
        if visits == 0:
            return 0.0
        return float(self.child_total_values[slot]) / visits

    def child_puct_value(self, slot: int, exploration_constant: float = EXPLORATION_CONSTANT) -> float:
        """Step 2: one slot at a time, read out of the arrays."""
        visits = int(self.child_visits[slot])
        exploitation = -self.child_average_value(slot)
        exploration = exploration_constant * float(self.child_priors[slot]) * math.sqrt(self.visits) / (1 + visits)
        return exploitation + exploration

    def child_puct_values(self, exploration_constant: float = EXPLORATION_CONSTANT) -> NDArray[np.float64]:
        """Step 3, verbatim from the shipped engine. Around ten array operations."""
        average_values = np.zeros_like(self.child_total_values)
        np.divide(
            self.child_total_values,
            self.child_visits,
            where=(self.child_visits != 0),
            out=average_values,
        )
        exploitation = -average_values

        all_visits_sqrt = math.sqrt(self.visits)
        exploration = exploration_constant * self.child_priors * all_visits_sqrt / (self.child_visits + 1)

        return exploitation + exploration


class ListNode:
    """Slot addressing without numpy: ArrayNode's layout, in lists.

    Note the absence of int()/float() conversions. B needs them to turn numpy
    scalars back into Python numbers, and that is part of what it pays for.
    """

    def __init__(self, visits: int, priors: list[float], child_visits: list[int], child_values: list[float]):
        self.visits = visits
        self.child_priors = list(priors)
        self.child_visits = list(child_visits)
        self.child_total_values = list(child_values)
        self.child_count = len(priors)

    def child_average_value(self, slot: int) -> float:
        visits = self.child_visits[slot]
        if visits == 0:
            return 0.0
        return self.child_total_values[slot] / visits

    def child_puct_value(self, slot: int, exploration_constant: float = EXPLORATION_CONSTANT) -> float:
        visits = self.child_visits[slot]
        exploitation = -self.child_average_value(slot)
        exploration = exploration_constant * self.child_priors[slot] * math.sqrt(self.visits) / (1 + visits)
        return exploitation + exploration


def fused_puct_values(node: ArrayNode, exploration_constant: float = EXPLORATION_CONSTANT) -> NDArray[np.float64]:
    """E: the same scores in roughly half the array operations.

    The scalar coefficient folds into one multiply, the zero-visit guard becomes
    a clamped denominator rather than an allocation plus a masked divide, and the
    exploration term accumulates in place.

    The clamp is exact rather than approximate, but only because ``record_visit``
    increments a slot's visit count and its total value together: zero visits
    implies a total value of exactly 0.0, so the quotient is 0.0 either way. That
    invariant is implicit in the engine today. Adopting E would mean stating it
    where both ``expand`` and ``record_visit`` can see it — a future path that
    updated one array without the other would silently change selection rather
    than fail.
    """
    scores = node.child_priors * (exploration_constant * math.sqrt(node.visits))
    scores /= node.child_visits + 1
    scores -= node.child_total_values / np.maximum(node.child_visits, 1)
    return scores


def select_a(root: RootObject) -> int:
    best_child = max(root.children, key=lambda child: child.puct_value())
    return root.children.index(best_child)


def select_b(node: ArrayNode) -> int:
    return max(range(node.child_count), key=node.child_puct_value)


def select_c(node: ArrayNode) -> int:
    return int(np.argmax(node.child_puct_values()))


def select_d(node: ListNode) -> int:
    return max(range(node.child_count), key=node.child_puct_value)


def select_e(node: ArrayNode) -> int:
    return int(np.argmax(fused_puct_values(node)))


def make_statistics(width: int, rng: random.Random) -> tuple[list[float], list[int], list[float]]:
    """A plausible mid-search node: non-uniform priors, most slots unvisited.

    PUCT concentrates visits, so at a realistic budget a wide node has a handful
    of well-visited slots and a long tail of untouched ones. A uniformly-visited
    node would understate the zero-visit guard every variant has to carry.
    """
    weights = [rng.random() for _ in range(width)]
    total = sum(weights)
    priors = [weight / total for weight in weights]

    visits: list[int] = []
    values: list[float] = []
    remaining = PARENT_VISITS
    for prior in priors:
        slot_visits = min(remaining, int(prior * PARENT_VISITS))
        remaining -= slot_visits
        visits.append(slot_visits)
        # Zero visits must mean exactly zero total value: that is the engine's
        # invariant, and E's clamped denominator depends on it.
        values.append(rng.uniform(-1.0, 1.0) * slot_visits)
    return priors, visits, values


def time_call(callable_under_test: Callable[[Any], int], argument: Any) -> float:
    """Nanoseconds per call, best of REPEATS timed batches."""
    timer = timeit.Timer(lambda: callable_under_test(argument))
    number, _ = timer.autorange()
    runs = timer.repeat(repeat=REPEATS, number=number)
    return min(runs) / number * 1e9


def main() -> None:
    rng = random.Random(SEED)

    print(f"\nPUCT selection kernel, nanoseconds per node scored (parent visits={PARENT_VISITS})")
    print(f"{'width':>6} {'A objects':>11} {'B nparrays':>11} {'C vector':>10} "
          f"{'D lists':>10} {'E fused':>10}   {'D vs A':>7} {'E vs C':>7}")
    print("-" * 88)

    for width in WIDTHS:
        priors, visits, values = make_statistics(width, rng)

        root = RootObject(PARENT_VISITS)
        root.children = [
            ChildObject(root, prior, slot_visits, value)
            for prior, slot_visits, value in zip(priors, visits, values, strict=True)
        ]
        array_node = ArrayNode(PARENT_VISITS, priors, visits, values)
        list_node = ListNode(PARENT_VISITS, priors, visits, values)

        chosen = (
            select_a(root), select_b(array_node), select_c(array_node),
            select_d(list_node), select_e(array_node),
        )
        assert len(set(chosen)) == 1, f"width {width}: variants disagree, {chosen}"

        nanos_a = time_call(select_a, root)
        nanos_b = time_call(select_b, array_node)
        nanos_c = time_call(select_c, array_node)
        nanos_d = time_call(select_d, list_node)
        nanos_e = time_call(select_e, array_node)

        print(f"{width:>6} {nanos_a:>11,.0f} {nanos_b:>11,.0f} {nanos_c:>10,.0f} "
              f"{nanos_d:>10,.0f} {nanos_e:>10,.0f}   {nanos_a / nanos_d:>6.2f}x "
              f"{nanos_c / nanos_e:>6.2f}x")

    print("\nD vs A above 1.00x means slot addressing in lists beats Step 1's child objects.")
    print("E vs C above 1.00x means the fused kernel beats the shipped one.")
    print("A single run on a busy machine will be noisy; the recorded table is the")
    print("per-cell minimum across four runs.")


if __name__ == "__main__":
    main()
