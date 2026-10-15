"""Cost ledger + hard-cap enforcement.

Cost per candidate is `candidate.usage.cost_usd` from the exported report (the Engine-measured run
total, including grading). A priced run with `cost_usd=None` is never counted as zero silently: it
is written with status `unpriced` and the guard refuses to continue until it is resolved.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "cost_ledger.csv"
COLUMNS = [
    "timestamp_utc",
    "stage",
    "candidate",
    "cases",
    "cost_usd",
    "tokens_in",
    "tokens_out",
    "cost_status",
    "cumulative_usd",
    "report_file",
]


class BudgetError(RuntimeError):
    pass


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    with LEDGER.open(encoding="utf-8", newline="") as f:
        return sum(float(r["cost_usd"]) for r in csv.DictReader(f))


def check_before(cfg: dict, estimate_usd: float, *, confirmed: bool) -> None:
    b = cfg["budget"]
    if _has_unpriced():
        raise BudgetError("ledger has unpriced rows; resolve them before spending more")
    cap, so_far = float(b["hard_cap_usd"]), spent()
    if so_far + estimate_usd > cap:
        raise BudgetError(
            f"hard cap: spent {so_far:.4f} + est {estimate_usd:.4f} > cap {cap}"
        )
    needs_confirm = (
        estimate_usd > float(b["confirm_above_run_usd"])
        or so_far + estimate_usd > float(b["confirm_above_cap_fraction"]) * cap
    )
    if needs_confirm and not confirmed:
        raise BudgetError(
            f"estimate {estimate_usd:.2f} needs user confirmation (--confirmed)"
        )


def _has_unpriced() -> bool:
    if not LEDGER.exists():
        return False
    with LEDGER.open(encoding="utf-8", newline="") as f:
        return any(r["cost_status"] == "unpriced" for r in csv.DictReader(f))


def record(
    report, stage: str, report_file: Path, free_routes: frozenset[str] = frozenset()
) -> float:
    """Append one ledger row per candidate; return the total cost of this report."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    running = spent()
    total = 0.0
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    try:
        shown = report_file.relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(report_file)
    with LEDGER.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for cand in report.candidates:
            u = cand.usage
            tokens = (u.input_tokens or 0) + (u.output_tokens or 0)
            cost = u.cost_usd
            if cost is not None:
                status = "priced"
            elif tokens and cand.models and all(m in free_routes for m in cand.models):
                status = "free_tier"
            else:
                status = "unpriced" if tokens else "no_usage"
            cost = float(cost or 0.0)
            running += cost
            total += cost
            cases = len(cand.cases) if hasattr(cand, "cases") else ""
            w.writerow(
                [
                    now,
                    stage,
                    cand.name,
                    cases,
                    f"{cost:.6f}",
                    u.input_tokens,
                    u.output_tokens,
                    status,
                    f"{running:.6f}",
                    shown,
                ]
            )
    return total
