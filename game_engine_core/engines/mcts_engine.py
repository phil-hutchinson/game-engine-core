from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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

    def __init__(self, evaluator: TEvaluator, iterations: int = 1_000, verbose: bool = False, temperature: float = 0.0):
        self.evaluator = evaluator
        self.iterations = iterations
        self.verbose = verbose
        self._temperature = temperature
        self._root_node = None

    def select_ply(self, game_position: TPosition) -> TPly:
        """Select the best ply using MCTS."""
        if self._root_node is None:
            self._root_node = self._create_root(game_position)

        self._grow_tree(self._root_node)

        return self._choose_ply(self._root_node)

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

    def select_ply_with_policy(self, game_position: TPosition) -> tuple[TPly, dict[str, float]]:
        """Select the best ply and return the MCTS visit distribution over all legal plies.

        The visit distribution is the normalised visit count for each legal ply at the root.
        Expansion attaches a child per legal ply, so plies the search never descended into
        are present with 0 visits and thus 0 probability. It is used as the policy training
        target during self-play data collection.

        Which plies land at exactly 0 is a function of the iteration budget against the
        branching factor: PUCT is free to leave a low-prior ply unvisited at any budget,
        where round-robin expansion once guaranteed every child a visit. Callers collecting
        training data should size iterations well above the number of legal plies, since a
        0 here is a hard zero in a cross-entropy policy target.

        Returns:
            A tuple of (selected_ply, policy) where policy maps str(ply) to probability
            for every legal ply in the position.
        """
        root = self._create_root(game_position)
        self._grow_tree(root)
        return self._choose_ply(root), self._visit_distribution(root)

    def _create_root(self, game_position: TPosition) -> MCTSNode[TPosition, TPly]:
        """Create a bare root node for the given position."""
        root: MCTSNode[TPosition, TPly] = MCTSNode(
            position=game_position, parent=None, ply_from_parent=None
        )
        return root

    def _grow_tree(self, root: MCTSNode[TPosition, TPly]) -> None:
        """Run all MCTS iterations on the tree provided by root."""
        for _ in range(self.iterations):
            self._mcts_iteration(root)

    def _choose_ply(self, root: MCTSNode[TPosition, TPly]) -> TPly:
        """Select a ply from the root's children according to the temperature setting."""
        if self._temperature == 0.0:
            return self._select_best_ply(root)
        return self._select_best_ply_with_temperature(root, self._temperature)

    def _visit_distribution(self, root: MCTSNode[TPosition, TPly]) -> dict[str, float]:
        """Return a normalised visit-count distribution over all legal plies at the root."""
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
            legal_plies = list(root.position.legal_plies)
            return {str(ply): 1.0 / len(legal_plies) for ply in legal_plies}
        return {k: v / total for k, v in child_visits.items()}

    def _mcts_iteration(self, root: MCTSNode[TPosition, TPly]) -> None:
        """Run one MCTS iteration: Select, Evaluate, Expand, Backpropagate.

        Exactly one evaluation per iteration. The value backpropagated is the
        leaf's own — expansion only makes the leaf descendable next time, and a
        terminal leaf is re-scored from its outcome without reaching the
        evaluator.
        """
        selected_node = self._select_leaf(root)

        outcome = selected_node.position.outcome
        if outcome is not None:
            value = float(outcome)
        else:
            value = self._evaluate_and_expand_node(selected_node)

        self._backpropagate(selected_node, value)

    def _select_leaf(self, root: MCTSNode[TPosition, TPly]) -> MCTSNode[TPosition, TPly]:
        """Descend by PUCT to a leaf: a node with no children.

        That is either a node not yet evaluated, or a terminal one — which never
        gains children and so is reached as a leaf on every iteration it wins.
        """
        current = root

        while current.children:
            best_child = max(current.children, key=lambda child: child.puct_value())
            current = best_child

        return current

    def _evaluate_and_expand_node(self, node: MCTSNode[TPosition, TPly]) -> float:
        """Evaluate a non-terminal leaf and attach a child for every legal ply.

        The policy is consumed here and not retained: a prior is only ever read
        at child construction. Evaluators must supply one covering every legal
        ply (see PositionEvaluation.policy) — the engine has no uniform default.
        """
        assert node.position.outcome is None

        evaluation = self.evaluator.evaluate_position(node.position)
        policy = evaluation.policy
        legal_plies: Sequence[TPly] = node.position.legal_plies

        # Build the children before attaching any, so an incomplete policy leaves
        # the node an unexpanded leaf rather than a half-expanded one.
        children: list[MCTSNode[TPosition, TPly]] = []
        for legal_ply in legal_plies:
            ply_key = str(legal_ply)
            try:
                prior = policy[ply_key]
            except KeyError:
                raise ValueError(f"Policy missing entry for ply '{ply_key}'") from None
            new_position = node.position.apply_ply(legal_ply)
            children.append(MCTSNode(
                position=new_position,
                parent=node,
                ply_from_parent=legal_ply,
                prior=prior,
            ))

        node.children.extend(children)
        return evaluation.value

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
            plies = list(root.position.legal_plies)
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
            plies = list(root.position.legal_plies)
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
