"""Record the runtime environment into results/environment.json (no model calls, no spend)."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

import screamingface as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    sf.configure(engine_url=cfg["engine_url"], scoreboard_url=cfg["scoreboard_url"])
    boards = {b.id: b for b in sf.benchmarks.list()}
    env = {
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version,
        "packages": {
            name: md.version(name)
            for name in ("screamingface", "screamingface-engine", "url4", "aigateway")
            if _installed(name)
        },
        "engine_url": cfg["engine_url"],
        "scoreboard_url": cfg["scoreboard_url"],
        "benchmarks": {
            bid: {"revision": b.revision, "case_count": b.case_count}
            for bid, b in sorted(boards.items())
        },
    }
    out = ROOT / "results" / "environment.json"
    out.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _installed(name: str) -> bool:
    try:
        md.version(name)
    except md.PackageNotFoundError:
        return False
    return True


if __name__ == "__main__":
    main()
