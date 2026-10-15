"""Single entry point: python -m src.run_all --stage {smoke,probe,main,analyze,replay}."""

from __future__ import annotations

import argparse

from src import analyze, replay, run
from src.candidates import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=["smoke", "probe", "main", "analyze", "replay"],
    )
    args = ap.parse_args()
    if args.stage in ("smoke", "probe", "main"):
        limit = {"smoke": 1, "probe": 5, "main": load_config()["n_main"]}[args.stage]
        run.run_stage(
            args.stage,
            limit=limit,
            estimate_usd=0.0,
            raw_name="ifeval_main" if args.stage == "main" else None,
        )
    elif args.stage == "analyze":
        analyze.main(analyze.ROOT / "results" / "raw" / "ifeval_main.json")
    else:
        replay.replay_all()


if __name__ == "__main__":
    main()
