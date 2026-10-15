"""Replay every saved url4 via `sf.evaluate(url4)` and compare with the main run.

url4 is already complete (benchmark + case limit embedded), so no `benchmark=`/`limit=` is passed.
Sampling is unseeded and the request cache is off, so a replay re-samples: scores are compared for
statistical consistency, not equality. Candidates are
replayed in parallel as separate processes, mirroring the main run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import screamingface as sf

from src import budget, winpatch  # noqa: F401
from src.candidates import ALL_IDS, load_config
from src.load import load_report, per_case_table

ROOT = Path(__file__).resolve().parents[1]


def replay_one(name: str) -> Path:
    cfg = load_config()
    budget.check_before(cfg, 0.0, confirmed=False)
    sf.configure(engine_url=cfg["engine_url"], scoreboard_url=cfg["scoreboard_url"])
    url4 = (
        (ROOT / "results" / "url4" / f"{name}.txt").read_text(encoding="utf-8").strip()
    )
    report = sf.evaluate(url4)
    out = ROOT / "results" / "raw" / f"replay_{name}.json"
    report.export(out)
    cost = budget.record(report, "replay", out, frozenset(cfg["free_tier_routes"]))
    print(f"{name}: replay cost ${cost:.4f}")
    return out


def replay_all() -> dict:
    procs = {
        name: subprocess.Popen(
            [sys.executable, "-W", "ignore", "-m", "src.replay", "--one", name],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for name in ALL_IDS
    }
    for name, proc in procs.items():
        out, _ = proc.communicate()
        print(out.strip())
        if proc.returncode:
            raise RuntimeError(f"replay of {name} failed")
    main = per_case_table(load_report(ROOT / "results" / "raw" / "ifeval_main.json"))
    result = {}
    for name in ALL_IDS:
        rep = per_case_table(
            load_report(ROOT / "results" / "raw" / f"replay_{name}.json")
        )
        a = main[main.candidate == name].set_index("case_id").score
        b = rep.set_index("case_id").score
        common = a.index.intersection(b.index)
        same = float((a[common] == b[common]).mean())
        result[name] = {
            "main_score": round(float(a[common].mean()), 6),
            "replay_score": round(float(b[common].mean()), 6),
            "difference": round(float(b[common].mean() - a[common].mean()), 6),
            "per_case_agreement": round(same, 6),
            "cases": len(common),
            "same_case_ids": bool(set(a.index) == set(b.index)),
        }
    result["_max_abs_difference"] = round(
        float(
            max(
                abs(v["difference"]) for k, v in result.items() if not k.startswith("_")
            )
        ),
        6,
    )
    (ROOT / "results" / "replay_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        np.round(result["_max_abs_difference"], 4),
        "= max |replay - main| score difference",
    )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    args = ap.parse_args()
    replay_one(args.one) if args.one else replay_all()
