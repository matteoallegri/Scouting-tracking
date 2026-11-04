#!/usr/bin/env python3
"""Costruisce i profili per la dashboard locale di scouting da tracking."""

from pathlib import Path as _Path
import sys as _sys

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

from src.scouting_tracking.config import ScoutingConfig
from src.scouting_tracking.pipeline import build_all

ROOT = _PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scouting_tracking.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/scouting_tracking")
    args = parser.parse_args()
    report = build_all(ROOT, args.output_dir, ScoutingConfig.from_json(args.config))
    print(f"Completato: {report['profiles']} profili, {report['reference_cohorts']} campioni di riferimento; {args.output_dir}")


if __name__ == "__main__":
    main()
