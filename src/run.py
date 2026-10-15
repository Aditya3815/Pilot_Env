"""Smoke / probe / main runs with a budget guard. Spends real money; see src/budget.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import screamingface as sf

from src import budget, winpatch  # noqa: F401  (winpatch: Windows fsync workaround)
from src.candidates import ALL_IDS, build, load_config

ROOT = Path(__file__).resolve().parents[1]


def run_stage(
    stage: str,
    *,
    limit: int,
    estimate_usd: float,
    candidate_ids: tuple[str, ...] = ALL_IDS,
    benchmark: str | None = None,
    answer_seed: int | None = None,
    confirmed: bool = False,
    raw_name: str | None = None,
) -> Path:
    cfg = load_config()
    benchmark = benchmark or cfg["benchmark"]
    budget.check_before(cfg, estimate_usd, confirmed=confirmed)
    sf.configure(engine_url=cfg["engine_url"], scoreboard_url=cfg["scoreboard_url"])
    recipes = build(cfg)
    selected = [recipes[i] for i in candidate_ids]
    out = ROOT / "results" / "raw" / f"{raw_name or stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = sf.evaluate(
            selected, benchmark=benchmark, limit=limit, answer_seed=answer_seed
        )
    except sf.ExecutionError as error:
        if error.code != "candidates_failed":
            raise
        print(f"FAILED candidates: {error.details.get('failed')}", file=sys.stderr)
        if error.partial_report is not None:
            error.partial_report.export(out.with_name(out.stem + "_partial.json"))
            budget.record(
                error.partial_report,
                stage,
                out.with_name(out.stem + "_partial.json"),
                frozenset(cfg["free_tier_routes"]),
            )
        raise
    report.export(out)
    if stage == "main":
        _save_url4(report)
    spent = budget.record(report, stage, out, frozenset(cfg["free_tier_routes"]))
    print(
        f"run cost ${spent:.4f}; cumulative ${budget.spent():.4f} / ${cfg['budget']['hard_cap_usd']}"
    )
    for cand in report.candidates:
        print(f"  {cand.name:18s} score={cand.score} coverage={cand.coverage}")
    if any(c.coverage < 1.0 for c in report.candidates):
        raise RuntimeError(
            f"incomplete grading in {out}; inspect failures before continuing"
        )
    return out


def _save_url4(report) -> None:
    """Save each candidate's url4 string and its Url4.to_python() source (replay recipes)."""
    folder = ROOT / "results" / "url4"
    folder.mkdir(parents=True, exist_ok=True)
    for cand in report.candidates:
        (folder / f"{cand.name}.txt").write_text(
            str(cand.url4) + "\n", encoding="utf-8"
        )
        (folder / f"{cand.name}.py").write_text(cand.url4.to_python(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "probe", "main"])
    ap.add_argument("--confirmed", action="store_true")
    args = ap.parse_args()
    limit, estimate = {
        "smoke": (1, 0.0),
        "probe": (5, 0.0),
        "main": (load_config()["n_main"], 0.0),
    }[args.stage]
    path = run_stage(
        args.stage,
        raw_name="ifeval_main" if args.stage == "main" else None,
        limit=limit,
        estimate_usd=estimate,
        confirmed=args.confirmed,
    )
    print(f"exported {path}")


if __name__ == "__main__":
    main()
