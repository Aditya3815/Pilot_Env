"""Raw report JSON -> tidy tables. Reads only the exported report; no SDK, no network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PER_CASE_COLUMNS = [
    "candidate",
    "case_id",
    "score",
    "cost_usd",
    "provider_latency_ms",
    "tokens_in",
    "tokens_out",
    "abstained",
]


def _sum_known(values: list) -> float:
    """Sum of non-null values; NaN when none is known (unknown is never treated as zero)."""
    known = [v for v in values if v is not None]
    return float(sum(known)) if known else float("nan")


def load_report(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def per_case_table(report: dict) -> pd.DataFrame:
    """One row per (candidate, case).

    Fields used: `candidate.name`; per case `case_id`, `status`, `grade.score`
    (IFEval prompt-level all-strict, 1.0/0.0), `grade.metrics.strict_checks_passed`,
    `grade.metrics.follow_all_loose`, `grade.checks` (one per instruction), and per-operation
    `accounting` (`usage.input_tokens/output_tokens/cost_usd`, `provider_latency_ms`).
    The free-tier gateway attributes no per-call usage, so the accounting columns are NaN here
    (unknown, not zero); run-level totals live in `candidate_table`.
    """
    rows = []
    for cand in report["candidates"]:
        for case in cand["cases"]:
            ops = [op.get("accounting") or {} for op in case.get("operations") or []]
            usage = [a.get("usage") or {} for a in ops]
            grade = case.get("grade") or {}
            rows.append(
                {
                    "candidate": cand["name"],
                    "case_id": case["case_id"],
                    "score": grade.get("score")
                    if case["status"] == "scored"
                    else np.nan,
                    "cost_usd": _sum_known([u.get("cost_usd") for u in usage]),
                    "provider_latency_ms": _sum_known(
                        [a.get("provider_latency_ms") for a in ops]
                    ),
                    "tokens_in": _sum_known([u.get("input_tokens") for u in usage]),
                    "tokens_out": _sum_known([u.get("output_tokens") for u in usage]),
                    "abstained": case["status"] == "abstained",
                    "status": case["status"],
                }
            )
    return pd.DataFrame(rows)


def instruction_table(report: dict) -> pd.DataFrame:
    """One row per (candidate, case, instruction) with strict and loose pass flags."""
    rows = []
    for cand in report["candidates"]:
        for case in cand["cases"]:
            if case["status"] != "scored":
                continue
            for check in (case.get("grade") or {}).get("checks") or []:
                modes = {
                    (ev.get("metadata") or {}).get("mode"): ev.get("outcome")
                    for ev in check.get("evidence") or []
                }
                rows.append(
                    {
                        "candidate": cand["name"],
                        "case_id": case["case_id"],
                        "instruction_id": check["id"],
                        "strict_pass": modes.get("strict") == "PASS",
                        "loose_pass": modes.get("loose") == "PASS",
                    }
                )
    return pd.DataFrame(rows)


def candidate_table(report: dict) -> pd.DataFrame:
    """Run-level totals per candidate (the only token/latency data the free tier exposes)."""
    rows = []
    for cand in report["candidates"]:
        u = cand["usage"]
        rows.append(
            {
                "candidate": cand["name"],
                "run_cases": len(cand["cases"]),
                "coverage": cand["coverage"],
                "engine_score": cand["score"],
                "tokens_in_total": u["input_tokens"],
                "tokens_out_total": u["output_tokens"],
                "cost_usd_total": u["cost_usd"],
                "wall_ms_total": cand["duration_ms"],
            }
        )
    return pd.DataFrame(rows)
