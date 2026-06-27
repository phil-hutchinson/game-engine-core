from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from typing import Any

from ..protocols.game_position import GamePosition
from ..protocols.game_ply import GamePly
from ..protocols.position_evaluator import PositionEvaluator


@dataclass
class MCTSNode[TPosition: GamePosition[Any], TPly: GamePly]:
    """A node in the MCTS tree."""

    position: TPosition
    parent: MCTSNode[TPosition, TPly] | None
    ply_from_parent: TPly | None  # ply that led to this position
    children: list[MCTSNode[TPosition, TPly]] = field(default_factory=lambda: [])
    unexplored_plies: list[TPly] | None = None

    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0

    # Policy head output stored at evaluation time; distributed to children as priors on expansion.
    prior: float = 1.0
    policy: dict[str, float] | None = None

    @property
    def average_value(self) -> float:
        """Average value from this node's perspective."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def is_fully_expanded(self) -> bool:
        """True if all possible moves have been tried."""
        return (self.unexplored_plies is not None) and len(self.unexplored_plies) == 0

    def puct_value(self, exploration_constant: float = 1.41) -> float:
        """PUCT selection score.

        Reduces to UCT when all priors are uniform (policy=None on the evaluator).
        """
        assert self.parent is not None
        exploitation = -self.average_value
        exploration = exploration_constant * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration


class MCTSEngine[TPly: GamePly, TPosition: GamePosition[Any], TEvaluator: PositionEvaluator[Any, Any]]:
    """Monte Carlo Tree Search engine."""

    def __init__(self, evaluator: TEvaluator, iterations: int = 200000, verbose: bool = False, temperature: float = 0.0):
        self.evaluator = evaluator
        self.iterations = iterations
        self.verbose = verbose
        self._temperature = temperature

    def select_ply(self, game_position: TPosition) -> TPly:
        """Select the best ply using MCTS."""
        root = self._build_tree(game_position)
        return self._choose_ply(root)

    def select_ply_with_policy(self, game_position: TPosition) -> tuple[TPly, dict[str, float]]:
        """Select the best ply and return the MCTS visit distribution over all legal plies.

        The visit distribution is the normalised visit count for each legal ply at the root,
        including unexplored plies (which receive 0 visits and thus 0 probability). It is
        used as the policy training target during self-play data collection.

        Returns:
            A tuple of (selected_ply, policy) where policy maps str(ply) to probability
            for every legal ply in the position.
        """
        root = self._build_tree(game_position)
        return self._choose_ply(root), self._visit_distribution(root)

    def _build_tree(self, game_position: TPosition) -> MCTSNode[TPosition, TPly]:
        """Run all MCTS iterations from the given position and return the root node."""
        root: MCTSNode[TPosition, TPly] = MCTSNode(
            position=game_position, parent=None, ply_from_parent=None
        )
        for _ in range(self.iterations):
            self._mcts_iteration(root)
        return root

    def _choose_ply(self, root: MCTSNode[TPosition, TPly]) -> TPly:
        """Select a ply from the root's children according to the temperature setting."""
        if self._temperature == 0.0:
            return self._select_best_ply(root)
        return self._select_best_ply_with_temperature(root, self._temperature)

    def _visit_distribution(self, root: MCTSNode[TPosition, TPly]) -> dict[str, float]:
        """Return a normalised visit-count distribution over all legal plies at the root."""
        legal_plies = list(root.position.legal_plies)
        child_visits: dict[str, int] = {
            str(child.ply_from_parent): child.visits
            for child in root.children
            if child.ply_from_parent is not None
        }
        # Include unexplored legal plies with 0 visits so the dict covers all legal plies.
        counts = {str(ply): child_visits.get(str(ply), 0) for ply in legal_plies}
        total = sum(counts.values())
        if total == 0:
            n = len(legal_plies)
            return {k: 1.0 / n for k in counts}
        return {k: v / total for k, v in counts.items()}

    def _mcts_iteration(self, root: MCTSNode[TPosition, TPly]) -> None:
        """Run one MCTS iteration: Select, Expand, Evaluate, Backpropagate."""
        node = self._select_leaf(root)
        expanded_node = self._expand_node(node)
        value = self._evaluate_node(expanded_node)
        self._backpropagate(expanded_node, value)

    def _select_leaf(self, root: MCTSNode[TPosition, TPly]) -> MCTSNode[TPosition, TPly]:
        """Select path down tree using PUCT until reaching an unexpanded node."""
        current = root

        while current.is_fully_expanded and current.children:
            best_child = max(current.children, key=lambda child: child.puct_value())
            current = best_child

        return current

    def _expand_node(self, node: MCTSNode[TPosition, TPly]) -> MCTSNode[TPosition, TPly]:
        """Expand node by adding one new child, or return node if terminal."""
        if node.position.outcome is not None:
            return node

        if node.unexplored_plies is None:
            node.unexplored_plies = list(node.position.legal_plies)

        if not node.unexplored_plies:
            return node

        ply = node.unexplored_plies.pop()
        new_position = node.position.apply_ply(ply)

        prior = 1.0
        if node.policy is not None:
            ply_key = str(ply)
            if ply_key not in node.policy:
                raise ValueError(f"Policy missing entry for move '{ply_key}'")
            prior = node.policy[ply_key]

        child: MCTSNode[TPosition, TPly] = MCTSNode(
            position=new_position,
            parent=node,
            ply_from_parent=ply,
            prior=prior,
        )
        node.children.append(child)
        return child

    def _evaluate_node(self, node: MCTSNode[TPosition, TPly]) -> float:
        """Evaluate the node using the position evaluator."""
        outcome = node.position.outcome
        if outcome is not None:
            return float(outcome)

        result = self.evaluator.evaluate_position(node.position)
        if result.policy is not None:
            node.policy = dict(result.policy)
        return result.value

    def _backpropagate(self, node: MCTSNode[TPosition, TPly], value: float) -> None:
        """Update statistics for this node and all ancestors."""
        current: MCTSNode[TPosition, TPly] | None = node

        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent
            value = -value

    def _select_best_ply(self, root: MCTSNode[TPosition, TPly]) -> TPly:
        """Select move with highest visit count."""
        if not root.children:
            plies = list(root.position.legal_plies)
            if not plies:
                raise RuntimeError("No available moves - position should have been treated as terminal.")
            return random.choice(plies)

        best_child = max(root.children, key=lambda child: child.visits)
        assert best_child.ply_from_parent is not None
        return best_child.ply_from_parent

    def _select_best_ply_with_temperature(self, root: MCTSNode[TPosition, TPly], temperature: float) -> TPly:
        """Select move proportionally to visit counts, scaled by temperature."""
        if not root.children:
            if self.verbose:
                print('No children. Choosing randomly.')
            plies = list(root.position.legal_plies)
            if not plies:
                raise RuntimeError("No available moves - position should have been treated as terminal.")
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
                zip(plies, visit_counts, scores, probabilities, prob_percentages),
                key=lambda x: x[3], reverse=True
            )
            parts = [f"({ply}, {v}, {s}, {pct})" for ply, v, s, _, pct in combined]
            print(f"Move analysis (ply, visits, score, probability): [{', '.join(parts)}]")

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
