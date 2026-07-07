"""Tournament reporting: standings/cross-table markdown plus per-game JSON logs.

The caller supplies the output folder (typically a fresh timestamped one);
this module never invents paths. Ply annotations are already strings
(str(ply)), so game logs serialise to JSON directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from .game_record import GameRecord
from .tournament_result import TournamentResult


def write_tournament_report(result: TournamentResult, output_dir: Path) -> None:
    """Write standings.md and games/*.json under output_dir, echoing the summary to stdout."""
    summary = render_summary(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "standings.md").write_text(summary + "\n")

    games_dir = output_dir / "games"
    games_dir.mkdir(exist_ok=True)
    # Pad to the record count's width so filenames sort in playing order.
    width = max(3, len(str(len(result.records))))
    for game_number, record in enumerate(result.records, start=1):
        game_path = games_dir / f"game-{game_number:0{width}d}.json"
        game_path.write_text(json.dumps(_game_json(record), indent=2) + "\n")

    print(summary)
    print(f"\nFull report written to {output_dir}")


def render_summary(result: TournamentResult) -> str:
    """Standings and cross-table as markdown (also readable as console text)."""
    lines = ["# Tournament results", "", "## Standings", ""]
    lines.append("| Rank | Player | W | D | L | Points |")
    lines.append("| ---- | ------ | - | - | - | ------ |")
    for rank, entry in enumerate(result.standings, start=1):
        lines.append(
            f"| {rank} | {entry.player_name} | {entry.wins} | {entry.draws} "
            f"| {entry.losses} | {_points(entry.points)} |"
        )

    lines += ["", "## Cross-table", ""]
    names = list(result.player_names)
    lines.append("| |" + "".join(f" {name} |" for name in names))
    lines.append("| - |" + " - |" * len(names))
    for row in names:
        cells = "".join(
            " — |" if col == row else f" {_points(result.cross_table[row][col])} |"
            for col in names
        )
        lines.append(f"| **{row}** |" + cells)

    return "\n".join(lines)


def _points(points: float) -> str:
    # 2.0 -> "2", 2.5 -> "2.5": half points only ever come from draws.
    return f"{points:g}"


def _game_json(record: GameRecord) -> dict[str, object]:
    return {
        "side_one": record.players[1],
        "side_two": record.players[-1],
        "outcome": record.result.outcome,
        "opening_board": record.result.opening_board,
        "plies": [
            {"ply": ply, "board": board} for ply, board in record.result.game_log
        ],
    }
