from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..game.batch_position_processor import BatchPositionProcessor
from ..models.position_evaluation import PositionEvaluation
from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition
from ..protocols.position_evaluator import PositionEvaluator


@dataclass
class MCTSNode[TPosition: GamePosition[Any], TPly: GamePly]:
    """A node in the MCTS tree."""

    position: TPosition
    parent: MCTSNode[TPosition, TPly] | None
    ply_from_parent: TPly | None  # ply that led to this position
    children: list[MCTSNode[TPosition, TPly]] = field(default_factory=lambda: [])

    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0

    # Share of the parent's policy mass for the ply leading here, set when the
    # parent was expanded. Drives the PUCT exploration term. A root's own prior
    # is never read, since selection only ever scores children.
    prior: float = 1.0

    @property
    def average_value(self) -> float:
        """Average value from this node's perspective."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def puct_value(self, exploration_constant: float = 1.41) -> float:
        """PUCT selection score.

        Note this is not UCT with priors: the exploration term is finite at zero
        visits, so an unvisited sibling can stay unvisited indefinitely while a
        high-prior ply is re-selected. Uniform priors do not recover UCB1, whose
        exploration term is unbounded as visits approach zero.
        """
        assert self.parent is not None
        exploitation = -self.average_value
        exploration = exploration_constant * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration


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

        new_root = next((node for node in self._root_node.children if str(node.ply_from_parent) == str(ply)), None)
        if new_root is None:
            # Unreachable through legal play against a searched root: full
            # expansion gives every legal ply a child. Only a ply the tree never
            # saw — an illegal one, or any ply if the root was never searched —
            # lands here, and the tree is discarded rather than mis-rooted.
            self._root_node = None
            return

        new_root.parent = None
        new_root.ply_from_parent = None
        self._root_node = new_root

    def reset(self) -> None:
        """A new game has started, clear state"""
        self._root_node = None

    def select_plies_for_training(self, positions: Sequence[TPosition]) -> Sequence[tuple[TPly, dict[str, float]]]:
        """Search a fleet of independent games in lockstep, one ply per game.

        Index-aligned with ``positions``: the position at index *i* is game *i*'s, and
        the result at index *i* is game *i*'s. Each game gets its own tree, addressed by
        that slot; the trees never interact. A single game is the fleet at N = 1.

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

        Each returned policy is the normalised visit count for every legal ply at that
        game's root, used as the policy training target during self-play collection.
        Expansion attaches a child per legal ply, so plies the search never descended
        into are present with 0 visits and thus 0 probability.

        Which plies land at exactly 0 is a function of the iteration budget against the
        branching factor: PUCT is free to leave a low-prior ply unvisited at any budget,
        where round-robin expansion once guaranteed every child a visit. Callers collecting
        training data should size iterations well above the number of legal plies, since a
        0 here is a hard zero in a cross-entropy policy target.

        Returns:
            One (selected_ply, policy) pair per input position, in slot order, where
            policy maps str(ply) to probability for every legal ply in that position.
        """
        roots = self._create_roots(positions)
        self._grow_trees(roots)
        return list(zip(self._choose_plies(roots), self._visit_distributions(roots), strict=True))

    def _create_roots(self, game_positions: Sequence[TPosition]) -> Sequence[MCTSNode[TPosition, TPly]]:
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
        the games synchronised without tracking progress per slot.
        """
        for _ in range(self.iterations):
            self._mcts_iteration(roots)

    def _choose_plies(self, roots: Sequence[MCTSNode[TPosition, TPly]]) -> Sequence[TPly]:
        """Select a ply from the root's children according to the temperature setting."""
        if self._temperature == 0.0:
            return [self._select_best_ply(root) for root in roots]
        return [self._select_best_ply_with_temperature(root, self._temperature) for root in roots]

    def _visit_distributions(self, roots: Sequence[MCTSNode[TPosition, TPly]]) -> Sequence[dict[str, float]]:
        return [self._visit_distribution(root) for root in roots]

    def _visit_distribution(self, root: MCTSNode[TPosition, TPly]) -> dict[str, float]:
        """Return a normalised visit-count distribution over all legal plies at the root.

        The zero-total fallback below is deliberately left at width one, so a fleet of
        N makes N legality calls rather than one of width N. It is the only width-one
        seam call left on the fleet path. It fires only when the budget cannot descend
        past a root — a budget that small is not a configuration worth optimising for,
        and widening it would mean restructuring ply choice into a plural form to reach
        the same fallback in _select_best_ply and _select_best_ply_with_temperature.
        Pinned by test_the_zero_visit_fallback_still_asks_for_legality_one_slot_at_a_time.
        """
        child_visits: dict[str, int] = {
            str(child.ply_from_parent): child.visits
            for child in root.children
            if child.ply_from_parent is not None
        }
        total = sum(child_visits.values())
        if total == 0:
            # No counts to normalise: either the root was never expanded, or it
            # was expanded but no iteration descended past it. Only this branch
            # needs the legal plies, since an unexpanded root has no children.
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
        # slot that evaluation i belongs to. Every slot in values is written
        # exactly once, here or in the scatter below, so the 0.0 never survives.
        values: list[float] = [0.0] * len(selected_nodes)
        pending: list[int] = []
        for slot, outcome in enumerate(outcomes):
            if outcome is None:
                pending.append(slot)
            else:
                values[slot] = float(outcome)

        # Guarded because an all-terminal iteration has nothing to evaluate — routine
        # late in a game — and there is no reason to hand the evaluator an empty batch.
        if pending:
            leaves = [selected_nodes[slot] for slot in pending]
            evaluations = self.evaluator.evaluate_positions([leaf.position for leaf in leaves])
            for slot, evaluation in zip(pending, evaluations, strict=True):
                values[slot] = evaluation.value
            self._expand_leaves(leaves, evaluations)

        for node, value in zip(selected_nodes, values, strict=True):
            self._backpropagate(node, value)

    def _select_leaves(self, roots: Sequence[MCTSNode[TPosition, TPly]]) -> Sequence[MCTSNode[TPosition, TPly]]:
        """Descend by PUCT to one leaf per tree, returned in the order the roots came in.

        A leaf is a node with no children: either a node not yet evaluated, or a
        terminal one — which never gains children and so is reached as a leaf on
        every iteration it wins.

        The slot ordering is load-bearing. Everything downstream pairs these leaves
        with their outcomes, evaluations and values by index, so a result that did
        not come back in root order would be backpropagated into the wrong tree.
        """

        return_value: list[MCTSNode[TPosition, TPly]] = []

        for root in roots:
            current = root

            while current.children:
                best_child = max(current.children, key=lambda child: child.puct_value())
                current = best_child

            return_value.append(current)

        return return_value

    def _expand_leaves(
        self,
        leaves: Sequence[MCTSNode[TPosition, TPly]],
        evaluations: Sequence[PositionEvaluation],
    ) -> None:
        """Attach a child per legal ply to every leaf, priced by its evaluation's policy.

        Pairs ``leaves`` with ``evaluations`` by index. Both arrive already narrowed to
        the non-terminal leaves of one iteration, so this method works entirely in that
        narrowed space and never sees a slot index — mapping results back to games is
        the caller's business.

        Each policy is consumed here and not retained: a prior is only ever read at
        child construction. Evaluators must supply one covering every legal ply (see
        PositionEvaluation.policy) — the engine has no uniform default. An incomplete
        policy leaves *every* leaf in the batch unexpanded, not just the offending one:
        priors all resolve before any successor is built.

        The caller is responsible for only passing non-terminal leaves, which it
        establishes from the outcomes it already has. Re-checking would mean a second
        trip through batch_ops for a fact one frame up already knows — free when
        outcome was a property read, not free now that it is a seam call a game may
        vectorise.
        """
        batch_legal_plies: Sequence[Sequence[TPly]] = self._batch_ops.legal_plies(
            [leaf.position for leaf in leaves]
        )

        # Flatten the fleet's expansions into one batch: every expanding leaf's
        # position repeated against each of its legal plies. Resolving all priors
        # first is what makes expansion all-or-nothing across the batch — a missing
        # policy entry raises before a single successor exists.
        flat_positions: list[TPosition] = []
        flat_plies: list[TPly] = []
        flat_priors: list[float] = []
        for leaf, evaluation, legal_plies in zip(leaves, evaluations, batch_legal_plies, strict=True):
            for legal_ply in legal_plies:
                ply_key = str(legal_ply)
                try:
                    flat_priors.append(evaluation.policy[ply_key])
                except KeyError:
                    raise ValueError(f"Policy missing entry for ply '{ply_key}'") from None
                flat_positions.append(leaf.position)
                flat_plies.append(legal_ply)

        # One call spanning every expanding leaf in the fleet, index-paired rather
        # than a cross product. A leaf's own children were already a batch before the
        # fleet existed; the fleet collapses those batches into this single call.
        successors = self._batch_ops.apply_plies(flat_positions, flat_plies)

        # Walk the flat results back out to their leaves. offset is the lane map at
        # this level, the way pending is one frame up.
        offset = 0
        for leaf, legal_plies in zip(leaves, batch_legal_plies, strict=True):
            end = offset + len(legal_plies)
            leaf.children.extend(
                MCTSNode(position=position, parent=leaf, ply_from_parent=legal_ply, prior=prior)
                for position, legal_ply, prior in zip(
                    successors[offset:end], legal_plies, flat_priors[offset:end], strict=True
                )
            )
            offset = end

    def _backpropagate(self, node: MCTSNode[TPosition, TPly], value: float) -> None:
        """Update statistics for this node and all ancestors."""
        current: MCTSNode[TPosition, TPly] | None = node

        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent
            value = -value

    def _select_best_ply(self, root: MCTSNode[TPosition, TPly]) -> TPly:
        """Select ply with highest visit count, breaking ties on prior.

        The tie-break only bites when no child has been visited — an expanded
        root with a budget too small to descend past it — where it returns the
        highest-prior ply instead of the first one in legal order.
        """
        if not root.children:
            plies = list(self._batch_ops.legal_plies([root.position])[0])
            if not plies:
                raise RuntimeError("No available plies - position should have been treated as terminal.")
            return random.choice(plies)

        best_child = max(root.children, key=lambda child: (child.visits, child.prior))
        assert best_child.ply_from_parent is not None
        return best_child.ply_from_parent

    def _select_best_ply_with_temperature(self, root: MCTSNode[TPosition, TPly], temperature: float) -> TPly:
        """Select ply proportionally to visit counts, scaled by temperature."""
        if not root.children:
            if self.verbose:
                print('No children. Choosing randomly.')
            plies = list(self._batch_ops.legal_plies([root.position])[0])
            if not plies:
                raise RuntimeError("No available plies - position should have been treated as terminal.")
            return random.choice(plies)

        visit_counts = [child.visits for child in root.children]
        total_visits = sum(visit_counts)

        if total_visits == 0:
            if self.verbose:
                print('No visits. Choosing randomly.')
            plies = [child.ply_from_parent for child in root.children]
            assert all(p is not None for p in plies)
            return random.choice(plies)  # type: ignore[return-value]

        probabilities = [(v / total_visits) ** (1.0 / temperature) for v in visit_counts]
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        if self.verbose:
            plies = [child.ply_from_parent for child in root.children]
            scores = [child.average_value for child in root.children]
            prob_percentages = [f"{p*100:.3f}%" for p in probabilities]
            combined = sorted(
                zip(plies, visit_counts, scores, probabilities, prob_percentages, strict=True),
                key=lambda x: x[3], reverse=True
            )
            parts = [f"({ply}, {v}, {s}, {pct})" for ply, v, s, _, pct in combined]
            print(f"Ply analysis (ply, visits, score, probability): [{', '.join(parts)}]")

        rand_val = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand_val <= cumulative:
                assert root.children[i].ply_from_parent is not None
                return root.children[i].ply_from_parent  # type: ignore[return-value]

        if self.verbose:
            print("Fallback to random.")
        result = random.choice(root.children).ply_from_parent
        assert result is not None
        return result
