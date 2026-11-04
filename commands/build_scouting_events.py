"""Prepara i conteggi Opta delle due finali per il confronto integrato."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scouting_data_access import load_verified_bundle
from src.scouting_events.data import build_bundle


def main():
    _, profiles, _, _ = load_verified_bundle(ROOT / "outputs/scouting_tracking", ROOT)
    report = build_bundle(ROOT, profiles, ROOT / "outputs/scouting_events")
    print(f"Eventi preparati: {report['player_matches']} giocatori-partita, {len(report['matches'])} finali.")


if __name__ == "__main__":
    main()
