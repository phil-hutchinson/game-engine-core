from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..game.batch_position_processor import BatchPositionProcessor
from ..models.position_evaluation import PositionEvaluation
from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition
from ..protocols.position_evaluator import PositionEvaluator

# Every unexpanded node shares these rather than allocating three zero-length
# arrays it will immediately throw away at expansion. Read-only so a stray write
# through one node cannot reach another; expansion replaces them outright.
_NO_PRIORS: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
_NO_VISITS: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
_NO_VALUES: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
for _empty in (_NO_PRIORS, _NO_VISITS, _NO_VALUES):
    _empty.setflags(write=False)


# eq=False: the generated __eq__ would compare every field as a tuple, including
# three ndarrays, which raises rather than returning a bool as soon as the game's
# position and ply types have value equality. Identity is what every caller here
# already relies on, and it keeps the node hashable.
@dataclass(eq=False)
class MCTSNode[TPosition: GamePosition[Any], TPly: GamePly]:
    """A node in the MCTS tree, holding its children's statistics as arrays.

    A child is not an object carrying its own scalars: it is a *slot*, an index
    into this node's parallel ``child_*`` arrays, ordered by the legal plies in
    ``child_plies``. The priors array is the policy itself — dense and
    positional, rather than keyed by ``str(ply)``.

    Those arrays are the single copy of a child's statistics. A materialised
    child addresses them through its own ``(parent, slot)`` pair rather than
    holding duplicates: mirroring the values on both the node and the parent
    array would be simpler to write and would drift the first time a code path
    updated only one of them.

    A root is the exception, having no parent to read through, and keeps its own
    scalars. ``slot`` is None exactly when ``parent`` is None, which is what
    makes each accessor below a clean either/or.

    A node with no slots is a leaf, whether because it has not been expanded or
    because it is terminal and never will be.
    """

    position: TPosition
    parent: MCTSNode[TPosition, TPly] | None
    ply_from_parent: TPly | None  # ply that led to this position
    slot: int | None = None  # index of this node in its parent's arrays

    # The children, in slot order. Empty until this node is expanded.
    child_plies: list[TPly] = field(default_factory=lambda: [])
    child_priors: NDArray[np.float64] = field(default_factory=lambda: _NO_PRIORS)
    child_visits: NDArray[np.int64] = field(default_factory=lambda: _NO_VISITS)
    child_total_values: NDArray[np.float64] = field(default_factory=lambda: _NO_VALUES)
    children: dict[int, MCTSNode[TPosition, TPly]] = field(default_factory=lambda: {})

    # A root's own statistics, read only while ``parent`` is None. Everything
    # else reads its counterparts out of its parent's arrays.
    root_visits: int = 0
    root_total_value: float = 0.0

    @property
    def child_count(self) -> int:
        """Number of slots, which is the number of legal plies once expanded."""
        return len(self.child_plies)

    @property
    def visits(self) -> int:
        """Visit count for this node, read through its slot in its parent."""
        if self.parent is None:
            return self.root_visits
        assert self.slot is not None
        return int(self.parent.child_visits[self.slot])

    @property
    def total_value(self) -> float:
        """Accumulated value for this node, read through its slot in its parent."""
        if self.parent is None:
            return self.root_total_value
        assert self.slot is not None
        return float(self.parent.child_total_values[self.slot])

    @property
    def average_value(self) -> float:
        """Average value from this node's perspective."""
        visits = self.visits
        if visits == 0:
            return 0.0
        return self.total_value / visits

    def record_visit(self, value: float) -> None:
        """Add one visit and ``value`` wherever this node's statistics live."""
        if self.parent is None:
            self.root_visits += 1
            self.root_total_value += value
            return
        assert self.slot is not None
        self.parent.child_visits[self.slot] += 1
        self.parent.child_total_values[self.slot] += value

    def child_average_value(self, slot: int) -> float:
        """Average value of a child slot, from that child's perspective."""
        visits = int(self.child_visits[slot])
        if visits == 0:
            return 0.0
        return float(self.child_total_values[slot]) / visits

    def child_puct_values(self, exploration_constant: float = 1.41) -> NDArray[np.float64]:
        """PUCT selection scores for every one of this node's child slots, as an array.

        Scored from the parent rather than from the child, since the parent holds
        every term: the child's statistics, its prior, and the parent visit count
        the exploration term scales by. It also means a slot can be scored
        without a child object existing for it.

        Note this is not UCT with priors: the exploration term is finite at zero
        visits, so an unvisited sibling can stay unvisited indefinitely while a
        high-prior ply is re-selected. Uniform priors do not recover UCB1, whose
        exploration term is unbounded as visits approach zero.
        """
        # exploitation
        average_values = np.zeros_like(self.child_total_values)
        np.divide(
            self.child_total_values,
            self.child_visits,
            where=(self.child_visits != 0),
            out=average_values,
        )
        exploitation = -average_values

        # exploration
        all_visits_sqrt = math.sqrt(self.visits)
        exploration = exploration_constant * self.child_priors * all_visits_sqrt / (self.child_visits + 1)

        return exploitation + exploration

    def expand(self, plies: list[TPly], priors: Sequence[float]) -> None:
        """Give this node one slot per legal ply, priced by the policy."""
        self.child_plies = plies
        self.child_priors = np.array(priors, dtype=np.float64)
        self.child_visits = np.zeros(len(plies), dtype=np.int64)
        self.child_total_values = np.zeros(len(plies), dtype=np.float64)

    def detach_as_root(self) -> None:
        """Promote this node to a root, bringing its statistics with it.

        Its parent holds the only copy of them and is about to be dropped, so
        they have to be read across before the link is broken — a root carries
        its own scalars precisely because it has no slot to read through.
        """
        if self.parent is None:
            return
        self.root_visits = self.visits
        self.root_total_value = self.total_value
        self.parent = None
        self.slot = None
        self.ply_from_parent = None


class MCTSEngine[TPly: GamePly, TPosition: GamePosition[Any], TEvaluator: PositionEvaluator[Any, Any]]:
    """Monte Carlo Tree Search engine."""

    _root_node: MCTSNode[TPosition, TPly] | None

    def __init__(
        self,
        evaluator: TEvaluator,
        iterations: int = 1_000,
        verbose: bool = False,
        temperature: float = 0.0,
        batch_ops: BatchPositionProcessor[TPly, TPosition] | None = None,
    ):
        self.evaluator = evaluator
        self.iterations = iterations
        self.verbose = verbose
        self._temperature = temperature
        self._batch_ops = batch_ops if batch_ops is not None else BatchPositionProcessor()
        self._root_node = None

    def select_ply(self, game_position: TPosition) -> TPly:
        """Select the best ply using MCTS."""
        if self._root_node is None:
            self._root_node = self._create_root(game_position)

        # Play is the fleet at N = 1: the retained root is wrapped into a
        # one-slot fleet and searched by the same machinery as training.
        root_nodes = [self._root_node]

        self._grow_trees(root_nodes)

        return self._choose_plies(root_nodes)[0]

    def observe_ply(self, position: TPosition, ply: TPly, new_position: TPosition) -> None:
        """Update tree based on ply applied in-game"""
        if self._root_node is None:
            return

        # Matched against the root's slot plies rather than against child objects:
        # the slots are the tree's record of what is legal here, and the child
        # objects are only their materialisation.
        new_slot = next(
            (slot for slot, slot_ply in enumerate(self._root_node.child_plies) if str(slot_ply) == str(ply)),
            None,
        )
        if new_slot is None:
            # Unreachable through legal play against a searched root: full
            # expansion gives every legal ply a slot. Only a ply the tree never
            # saw — an illegal one, or any ply if the root was never searched —
            # lands here, and the tree is discarded rather than mis-rooted.
            self._root_node = None
            return

        if new_slot in self._root_node.children:
            new_root = self._root_node.children[new_slot]
            new_root.detach_as_root()
            self._root_node = new_root
        else:
            # A legal ply the search never descended into: it has a slot (and
            # thus a prior and statistics) but no materialised child. Materialise
            # it now using the position the caller has already computed, rather
            # than discarding the tree.
            old_root = self._root_node
            ply_from_old_root = self._root_node.child_plies[new_slot]
            new_root: MCTSNode[TPosition, TPly] = MCTSNode(
                position=new_position, parent=old_root, ply_from_parent=ply_from_old_root, slot=new_slot
            )
            # Deliberately not written into old_root.children: detach_as_root reads
            # the statistics through (parent, slot), not through that dict, and
            # old_root is dropped on the next line.
            new_root.detach_as_root()
            self._root_node = new_root

    def reset(self) -> None:
        """A new game has started, clear state"""
        self._root_node = None

    def select_plies_for_training(self, positions: Sequence[TPosition]) -> Sequence[tuple[TPly, dict[str, float]]]:
        """Search a fleet of independent games in lockstep, one ply per game.

        Index-aligned with ``positions``: the position at index *i* is game *i*'s, and
        the result at index *i* is game *i*'s. Each game gets its own tree, addressed by
        that fleet position; the trees never interact. A single game is the fleet at
        N = 1.

        The point of the plural form is the evaluator call. All N trees advance one
        iteration together, so each iteration gathers one leaf per game into a single
        ``evaluate_positions`` call rather than making N calls of width one. With equal
        iteration budgets the games stay synchronised with no explicit coordination.
        Leaves that are terminal are scored from their outcome and leave the batch,
        which narrows it without breaking lockstep — those games still advance.

        Roots are built bare from ``positions`` on every call and nothing is retained
        between calls, unlike ``select_ply``, which keeps its tree across plies. Whether
        retaining visit statistics across plies helps in training is open (issue #30);
        until it is settled this path does not.

        Every position must be non-terminal and have at least one legal ply. A terminal
        root never gains slots, so ply choice finds no plies to fall back on and
        raises — forfeiting the completed searches of every other fleet position, not
        just its own. Deciding which games are in flight is the caller's job (issue #24).

        Each returned policy is the normalised visit count for every legal ply at that
        game's root, used as the policy training target during self-play collection.
        Expansion gives every legal ply a slot, so plies the search never descended
        into are present with 0 visits and thus 0 probability.

        Which plies land at exactly 0 is a function of the iteration budget against the
        branching factor: PUCT is free to leave a low-prior ply unvisited at any budget,
        where round-robin expansion once guaranteed every child a visit. Callers collecting
        training data should size iterations well above the number of legal plies, since a
        0 here is a hard zero in a cross-entropy policy target.

        Returns:
            One (selected_ply, policy) pair per input position, in fleet order, where
            policy maps str(ply) to probability for every legal ply in that position.
        """
        roots = self._create_roots(positions)
        self._grow_trees(roots)
        return list(zip(self._choose_plies(roots), self._visit_distributions(roots), strict=True))

    def _create_roots(self, game_positions: Sequence[TPosition]) -> Sequence[MCTSNode[TPosition, TPly]]:
        """Create one bare root per position, in fleet order."""
        return [self._create_root(game_position) for game_position in game_positions]

    def _create_root(self, game_position: TPosition) -> MCTSNode[TPosition, TPly]:
        """Create a bare root node for the given position."""
        root: MCTSNode[TPosition, TPly] = MCTSNode(
            position=game_position, parent=None, ply_from_parent=None
        )
        return root

    def _grow_trees(self, roots: Sequence[MCTSNode[TPosition, TPly]]) -> None:
        """Run the full iteration budget against every tree in the fleet.

        One pass of the loop is one iteration for every tree, so equal budgets keep
        the games synchronised without tracking progress per fleet position.
        """
        # An empty fleet is reachable — #24 shrinks the fleet as its games finish — and
        # every phase of an empty iteration is a width-zero seam call. Harmless today,
        # but the budget is typically in the thousands and a vectorised processor need
        # not treat an empty batch as free, so skip the loop rather than the work.
        if not roots:
            return

        for _ in range(self.iterations):
            self._mcts_iteration(roots)

    def _choose_plies(self, roots: Sequence[MCTSNode[TPosition, TPly]]) -> Sequence[TPly]:
        """Select one ply per root from that root's children, in fleet order.

        Which rule applies is the temperature setting's business, and it is the same
        rule for every fleet position.
        """
        if self._temperature == 0.0:
            return [self._select_best_ply(root) for root in roots]
        return [self._select_best_ply_with_temperature(root, self._temperature) for root in roots]

    def _visit_distributions(self, roots: Sequence[MCTSNode[TPosition, TPly]]) -> Sequence[dict[str, float]]:
        """Return one visit distribution per root, in fleet order."""
        return [self._visit_distribution(root) for root in roots]

    def _visit_distribution(self, root: MCTSNode[TPosition, TPly]) -> dict[str, float]:
        """Return a normalised visit-count distribution over all legal plies at the root.

        The zero-total fallback below is deliberately left at width one, so a fleet of
        N makes N legality calls rather than one of width N. It is one of the three
        width-one seam calls left on the fleet path, alongside the no-children fallbacks
        in _select_best_ply and _select_best_ply_with_temperature. All three fire only
        when the budget cannot descend past a root — a budget that small is not a
        configuration worth optimising for, and widening this one alone would not help,
        since reaching the other two would mean restructuring ply choice into a plural
        form as well.
        """
        # Back to str(ply) keying here: the returned distribution is a public
        # contract and stays a dict, so the slot ordering is only internal.
        child_visits: dict[str, int] = {
            str(slot_ply): int(visits)
            for slot_ply, visits in zip(root.child_plies, root.child_visits, strict=True)
        }
        total = sum(child_visits.values())
        if total == 0:
            # No counts to normalise: either the root was never expanded, or it
            # was expanded but no iteration descended past it. Only this branch
            # needs the legal plies, since an unexpanded root has no slots.
            legal_plies = list(self._batch_ops.legal_plies([root.position])[0])
            return {str(ply): 1.0 / len(legal_plies) for ply in legal_plies}
        return {k: v / total for k, v in child_visits.items()}

    def _mcts_iteration(self, roots: Sequence[MCTSNode[TPosition, TPly]]) -> None:
        """Run one MCTS iteration for every tree: Select, Evaluate, Expand, Backpropagate.

        Each phase sweeps the whole fleet before the next begins, which is what
        collapses N evaluator calls of width one into a single batched call. Every
        tree advances exactly one iteration, so the fleet stays in lockstep.

        Still exactly one evaluation per tree per iteration. The value
        backpropagated is the leaf's own — expansion only makes the leaf
        descendable next time, and a terminal leaf is re-scored from its outcome
        without reaching the evaluator.
        """
        selected_nodes = self._select_leaves(roots)

        outcomes = self._batch_ops.outcomes([selected_node.position for selected_node in selected_nodes])

        # Partition the leaves by outcome. Terminal leaves are scored from that
        # outcome and leave the batch; the rest are evaluated together and their
        # values scattered back, with pending as the lane map — pending[i] is the
        # fleet position that evaluation i belongs to. Every fleet position in values
        # is written exactly once, here or in the scatter below, so the 0.0 never
        # survives.
        values: list[float] = [0.0] * len(selected_nodes)
        pending: list[int] = []
        for fleet_position, outcome in enumerate(outcomes):
            if outcome is None:
                pending.append(fleet_position)
            else:
                values[fleet_position] = float(outcome)

        # Guarded because an all-terminal iteration has nothing to evaluate — routine
        # late in a game — and there is no reason to hand the evaluator an empty batch.
        if pending:
            leaves = [selected_nodes[fleet_position] for fleet_position in pending]
            evaluations = self.evaluator.evaluate_positions([leaf.position for leaf in leaves])
            for fleet_position, evaluation in zip(pending, evaluations, strict=True):
                values[fleet_position] = evaluation.value
            self._expand_leaves(leaves, evaluations)

        for node, value in zip(selected_nodes, values, strict=True):
            self._backpropagate(node, value)

    def _select_leaves(self, roots: Sequence[MCTSNode[TPosition, TPly]]) -> Sequence[MCTSNode[TPosition, TPly]]:
        """Descend by PUCT to one leaf per tree, returned in the order the roots came in.

        A leaf is a node with no slots: either a node not yet evaluated, or a
        terminal one — which never gains slots and so is reached as a leaf on
        every iteration it wins.

        Descent also materialises, but the materialisation itself is deferred
        rather than performed inline. An expanded node's slots exist as priors
        and statistics before any child object does; the first descent to pick
        a given slot stops there and records (parent, slot, fleet_position) as
        pending, instead of building the successor position on the spot. A slot
        PUCT never picks stays unmaterialised indefinitely, which is what keeps
        expansion cheap: the fleet builds at most one successor per tree per
        iteration rather than one per legal ply.

        Materialisation is always the last act of a descent — a freshly
        materialised node has no children of its own to keep descending into —
        so once every tree has either reached a leaf or gone pending, the
        fleet's pending slots are exactly this wave's remaining leaves. That is
        what lets the whole wave's materialisation happen as a single batched
        ``apply_plies`` call below, at a width bounded by how many trees went
        pending (never by branching factor): no interleaving by depth is
        needed, since a tree never has more than one pending slot per wave.

        The fleet ordering is load-bearing. Everything downstream pairs these leaves
        with their outcomes, evaluations and values by index, so a result that did
        not come back in root order would be backpropagated into the wrong tree.
        Each pending leaf is scattered back into its own fleet position once
        materialised, so the returned order matches ``roots`` regardless of which
        trees went pending.
        """
        leaves: list[MCTSNode[TPosition, TPly]] = []
        # (parent, slot, fleet_position): the slot is an index into `parent`'s
        # arrays, the fleet position an index into `roots`/`leaves`. Both are dense
        # ints, so they are named apart deliberately — see the Vocabulary section of
        # CLAUDE.md. fleet_position is where this pending materialisation's eventual
        # node belongs in `leaves`, once the batch call below resolves it into a
        # real node.
        pending: list[tuple[MCTSNode[TPosition, TPly], int, int]] = []

        for fleet_position, root in enumerate(roots):
            current = root

            while current.child_count:
                # argmax returns the first maximal index, so an exact tie keeps the
                # earliest legal ply rather than breaking randomly.
                best_slot = int(np.argmax(current.child_puct_values()))
                if best_slot in current.children:
                    current = current.children[best_slot]
                else:
                    # best_slot is unvisited: no entry in children means no node has
                    # been materialised for it yet, even though its priors and
                    # statistics already exist in current's arrays. Defer rather
                    # than materialise here: record it as pending and stop
                    # descending. `current` (the parent) is only a placeholder in
                    # `leaves` until the batch below replaces it with this tree's
                    # real leaf.
                    pending.append((current, best_slot, fleet_position))
                    break  # Stop descending; can't proceed without materializing the child first.

            leaves.append(current)

        # One batched call for the whole wave, not one per tree: every pending
        # slot's parent position and chosen ply are gathered first, so the
        # width is the number of trees that went pending this wave.
        if pending:
            positions = [parent.position for parent, _, _ in pending]
            plies = [parent.child_plies[slot] for parent, slot, _ in pending]
            successors = self._batch_ops.apply_plies(positions, plies)

            for (parent, slot, fleet_position), successor_position in zip(pending, successors, strict=True):
                # slot=slot is required, not optional: it is how this node's
                # statistics resolve through its parent's arrays (see
                # visits/total_value/record_visit), and backpropagation starts
                # from this exact node.
                new_node = MCTSNode(
                    position=successor_position,
                    parent=parent,
                    ply_from_parent=parent.child_plies[slot],
                    slot=slot,
                )
                parent.children[slot] = new_node
                leaves[fleet_position] = new_node

        return leaves

    def _expand_leaves(
        self,
        leaves: Sequence[MCTSNode[TPosition, TPly]],
        evaluations: Sequence[PositionEvaluation],
    ) -> None:
        """Give every leaf a slot per legal ply, priced by its evaluation's policy.

        This is priors only — no successor position is built here. A leaf becomes
        expanded the moment it has a priors array; whether any of its slots also
        has a materialised child object is a separate, later fact, decided by
        descent rather than by expansion. ``apply_plies`` is not called by this
        method at all: ``_select_leaves`` defers materialisation for every slot
        a descent actually selects until the whole fleet has descended, then
        resolves them all in one batched call (``observe_ply``, for re-rooting,
        materialises a single slot from the position the caller already supplies,
        so it costs no ``apply_plies`` call at all).

        Pairs ``leaves`` with ``evaluations`` by index. Both arrive already narrowed to
        the non-terminal leaves of one iteration, so this method works entirely in that
        narrowed space and never sees a fleet position — mapping results back to games
        is the caller's business.

        This is the one place ``PositionEvaluation.policy``'s ``str(ply)`` keying is
        consumed: the policy is read into the leaf's priors array, positionally by
        slot, and the string keys end here. Evaluators must supply an entry covering
        every legal ply (see PositionEvaluation.policy) — the engine has no uniform
        default. An incomplete policy leaves *every* leaf in the batch unexpanded,
        not just the offending one: priors all resolve before any node is expanded.

        The caller is responsible for only passing non-terminal leaves, which it
        establishes from the outcomes it already has. Re-checking would mean a second
        trip through batch_ops for a fact one frame up already knows — free when
        outcome was a property read, not free now that it is a seam call a game may
        vectorise.
        """
        batch_legal_plies: Sequence[Sequence[TPly]] = self._batch_ops.legal_plies(
            [leaf.position for leaf in leaves]
        )

        # Resolving every leaf's priors before expanding any of them is what makes
        # expansion all-or-nothing across the batch — a missing policy entry raises
        # before a single leaf is marked expanded. Nothing here touches successor
        # positions; a slot is priors and statistics only until descent visits it.

        nodes_to_process = list(zip(leaves, evaluations, batch_legal_plies, strict=True))
        for index, (leaf, evaluation, legal_plies) in enumerate(nodes_to_process):
            for legal_ply in legal_plies:
                ply_key = str(legal_ply)
                if ply_key not in evaluation.policy:
                    # index is into this batch, not into the fleet — this method has no
                    # slot indices — but it is enough to identify the offending
                    # evaluation among N, which the ply key alone is not.
                    raise ValueError(
                        f"Policy missing entry for ply '{ply_key}' "
                        f"(batch leaf {index}, position {leaf.position})"
                    )

        # Every policy entry is known to be present at this point, so this pass
        # only reads priors into each leaf's slots — no child objects are created.
        for leaf, evaluation, legal_plies in nodes_to_process:
            priors: list[float] = []
            for legal_ply in legal_plies:
                priors.append(evaluation.policy[str(legal_ply)])
            leaf.expand(list(legal_plies), priors)

    def _backpropagate(self, node: MCTSNode[TPosition, TPly], value: float) -> None:
        """Update statistics for this node and all ancestors."""
        current: MCTSNode[TPosition, TPly] | None = node

        while current is not None:
            current.record_visit(value)
            current = current.parent
            value = -value

    def _select_best_ply(self, root: MCTSNode[TPosition, TPly]) -> TPly:
        """Select ply with highest visit count, breaking ties on prior.

        The tie-break only bites when no child has been visited — an expanded
        root with a budget too small to descend past it — where it returns the
        highest-prior ply instead of the first one in legal order.
        """
        if not root.child_count:
            plies = list(self._batch_ops.legal_plies([root.position])[0])
            if not plies:
                raise RuntimeError("No available plies - position should have been treated as terminal.")
            return random.choice(plies)

        best_slot = max(
            range(root.child_count),
            key=lambda slot: (int(root.child_visits[slot]), float(root.child_priors[slot])),
        )
        return root.child_plies[best_slot]

    def _select_best_ply_with_temperature(self, root: MCTSNode[TPosition, TPly], temperature: float) -> TPly:
        """Select ply proportionally to visit counts, scaled by temperature."""
        if not root.child_count:
            if self.verbose:
                print('No children. Choosing randomly.')
            plies = list(self._batch_ops.legal_plies([root.position])[0])
            if not plies:
                raise RuntimeError("No available plies - position should have been treated as terminal.")
            return random.choice(plies)

        visit_counts = [int(visits) for visits in root.child_visits]
        total_visits = sum(visit_counts)

        if total_visits == 0:
            if self.verbose:
                print('No visits. Choosing randomly.')
            return random.choice(root.child_plies)

        probabilities = [(v / total_visits) ** (1.0 / temperature) for v in visit_counts]
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        if self.verbose:
            scores = [root.child_average_value(slot) for slot in range(root.child_count)]
            prob_percentages = [f"{p*100:.3f}%" for p in probabilities]
            combined = sorted(
                zip(root.child_plies, visit_counts, scores, probabilities, prob_percentages, strict=True),
                key=lambda x: x[3], reverse=True
            )
            parts = [f"({ply}, {v}, {s}, {pct})" for ply, v, s, _, pct in combined]
            print(f"Ply analysis (ply, visits, score, probability): [{', '.join(parts)}]")

        rand_val = random.random()
        cumulative = 0.0
        for slot, prob in enumerate(probabilities):
            cumulative += prob
            if rand_val <= cumulative:
                return root.child_plies[slot]

        if self.verbose:
            print("Fallback to random.")
        return random.choice(root.child_plies)
