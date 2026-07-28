from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from ..protocols.game_ply import GamePly
from ..protocols.game_position import GamePosition


class BatchPositionProcessor[TPly: GamePly, TPosition: GamePosition[Any]]:
    """Batch-shaped access to a position's scalar members.

    Every method is index-aligned with its input: position ``i`` in produces
    result ``i`` out. ``legal_plies`` nests one level further, since each
    position contributes its own sequence of plies rather than a single value.
    ``apply_plies`` pairs ``positions`` and ``plies`` by index rather than
    taking a cross product — position ``i`` receives ``plies[i]``.

    The implementations here simply loop the corresponding ``GamePosition``
    member, so the seam is usable with zero setup. Override selectively: a
    game whose positions can be scored several at once (vectorised legality, a
    single batched engine call, ...) should override just that method and
    inherit the loop for the rest — this is the expected steady state, not a
    fallback reserved for toy games.

    Delegation hazard: ``GamePosition`` keeps its own scalar properties, and an
    implementation may write one of them as a batch-of-one call into a
    processor it holds a reference to. That is only safe if the processor
    actually overrides the method being delegated to — delegating into the
    inherited loop above calls straight back into the same scalar property,
    recursing until the stack overflows. Use ``overridden_methods`` to assert
    the assumption at construction rather than discovering it mid-search.
    """

    _BATCH_METHOD_NAMES: ClassVar[tuple[str, ...]] = ("outcomes", "legal_plies", "apply_plies")

    def outcomes(self, positions: Sequence[TPosition]) -> Sequence[Literal[1, 0, -1] | None]:
        """Outcome of each position, aligned by index. See ``GamePosition.outcome``."""
        return [position.outcome for position in positions]

    def legal_plies(self, positions: Sequence[TPosition]) -> Sequence[Sequence[TPly]]:
        """Legal plies of each position, aligned by index and nested one level."""
        return [position.legal_plies for position in positions]

    def apply_plies(self, positions: Sequence[TPosition], plies: Sequence[TPly]) -> Sequence[TPosition]:
        """Apply ``plies[i]`` to ``positions[i]`` for every index; not a cross product."""
        return [position.apply_ply(ply) for position, ply in zip(positions, plies, strict=True)]

    def overridden_methods(self) -> frozenset[str]:
        """Names of the batch methods this instance's class actually overrides.

        Lets a delegating ``GamePosition`` assert at construction that the
        processor it was handed overrides the method it plans to delegate to,
        rather than finding out through infinite recursion mid-search.
        """
        return frozenset(
            name
            for name in self._BATCH_METHOD_NAMES
            if getattr(type(self), name) is not getattr(BatchPositionProcessor, name)
        )
