"""Deterministic analysis. Reads only the saved raw report JSON; writes tables, figures, summary.

Metrics and definitions were fixed before the main run (see README).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.candidates import ALL_IDS, SAMPLE_SOLO_IDS, load_config
from src.load import candidate_table, instruction_table, load_report, per_case_table

ROOT = Path(__file__).resolve().parents[1]
SYNTH_IDS = ("synth_default", "coverage")
# Okabe-Ito (colourblind-safe)
PALETTE = {
    "solo_default": "#000000",
    "solo_s0": "#56B4E9",
    "solo_s1": "#0072B2",
    "solo_s2": "#009E73",
    "judge_pick": "#E69F00",
    "synth_default": "#D55E00",
    "coverage": "#CC79A7",
}


def _r(x: float, nd: int = 6) -> float | None:
    return (
        None
        if x is None or (isinstance(x, float) and np.isnan(x))
        else round(float(x), nd)
    )


def boot_ci(idx: np.ndarray, values: np.ndarray) -> list[float]:
    """Percentile 95% CI of the mean over case resamples `idx` (shape B x n)."""
    means = values[idx].mean(axis=1)
    return [_r(np.percentile(means, 2.5)), _r(np.percentile(means, 97.5))]


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar on binarised outcomes (pass = score >= 1)."""
    pa, pb = a >= 1.0, b >= 1.0
    only_a, only_b = int((pa & ~pb).sum()), int((~pa & pb).sum())
    disc = only_a + only_b
    p = 1.0 if disc == 0 else float(binomtest(only_a, disc, 0.5).pvalue)
    return {"only_first": only_a, "only_second": only_b, "p_value": _r(p)}


def pareto_front(
    points: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Non-dominated points (lower x better, higher y better), sorted by x."""
    front, best = [], -np.inf
    for x, y, name in sorted(points, key=lambda p: (p[0], -p[1])):
        if y > best:
            front.append((x, y, name))
            best = y
    return front


def main(raw: Path, out_root: Path = ROOT) -> dict:
    cfg = load_config()
    price = cfg["notional_price_usd_per_mtok"]
    report = load_report(raw)
    pc, instr, cand = (
        per_case_table(report),
        instruction_table(report),
        candidate_table(report),
    )
    tables = out_root / "results" / "tables"
    figs = out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    out_cols = [
        "candidate",
        "case_id",
        "score",
        "cost_usd",
        "provider_latency_ms",
        "tokens_in",
        "tokens_out",
        "abstained",
    ]
    pc[out_cols].to_csv(tables / "per_case.csv", index=False)

    # --- analysis case set: fully graded for ALL candidates --------------------------------
    names = [c for c in ALL_IDS if c in set(pc.candidate)]
    wide = pc.pivot(index="case_id", columns="candidate", values="score")[names]
    all_cases = list(wide.index)
    keep = wide.dropna().index
    excluded = [int(c) for c in all_cases if c not in keep]
    S = wide.loc[keep]
    n = len(S)
    case_ids = [int(c) for c in S.index]
    binary = bool(np.isin(S.to_numpy(), [0.0, 1.0]).all())

    rng = np.random.default_rng(cfg["bootstrap"]["rng_seed"])
    idx = rng.integers(0, n, size=(cfg["bootstrap"]["resamples"], n))

    # --- 1. per candidate -----------------------------------------------------------------
    cand = cand.set_index("candidate")
    rows, per_cand = [], {}
    for name in names:
        s = S[name].to_numpy()
        c = cand.loc[name]
        tin, tout = c.tokens_in_total / c.run_cases, c.tokens_out_total / c.run_cases
        notional = (tin * price["input"] + tout * price["output"]) / 1e6
        wall = c.wall_ms_total / c.run_cases / 1000
        per_cand[name] = {
            "score": _r(s.mean()),
            "score_ci95": boot_ci(idx, s),
            "tokens_in_per_case": _r(tin, 1),
            "tokens_out_per_case": _r(tout, 1),
            "notional_usd_per_case": _r(notional, 8),
            "notional_usd_total": _r(notional * c.run_cases, 6),
            "wall_s_per_case_mean": _r(wall, 3),
            "engine_score": _r(c.engine_score),
            "coverage": _r(c.coverage),
            "run_cases": int(c.run_cases),
        }
        rows.append({"candidate": name, **per_cand[name]})
    pd.DataFrame(rows).to_csv(tables / "candidates.csv", index=False)

    # --- 2/3. decomposition over the three solo draws ------------------------------------------
    panel = [c for c in SAMPLE_SOLO_IDS if c in S]
    P = S[panel].to_numpy()
    means = P.mean(axis=0)
    best_name = panel[int(np.argmax(means))]  # ties -> first (deterministic)
    best = float(means.max())
    oracle_case = P.max(axis=1)
    oracle = float(oracle_case.mean())
    fail_all = (P < 1.0).all(axis=1)
    k_fail = int(fail_all.sum())
    cp = binomtest(k_fail, n).proportion_ci(confidence_level=0.95, method="exact")
    beta = k_fail / n
    decomposition = {
        "panel": panel,
        "best_single": best_name,
        "best_single_score": _r(best),
        "oracle": _r(oracle),
        "oracle_ci95": boot_ci(idx, oracle_case),
        "oracle_gain": _r(oracle - best),
        "oracle_gain_ci95": boot_ci(idx, oracle_case - S[best_name].to_numpy()),
        "fail_definition": "score < 1.0 for all three draws",
        "beta": _r(beta),
        "beta_count": k_fail,
        "beta_cp95": [_r(cp.low), _r(cp.high)],
        "selection_ceiling": _r(1 - beta),
        "selection_ceiling_cp95": [_r(1 - cp.high), _r(1 - cp.low)],
        "zero_dollar_certificate_max_gain_over_best": _r(max((1 - cp.low) - best, 0.0)),
        "scores_binary": binary,
    }
    if not binary:
        decomposition["beta_lt_0_5"] = _r(float((P < 0.5).all(axis=1).mean()))

    # --- 4/5. synthesis surplus, rescue / damage -----------------------------------------------
    surplus, rd_rows = {}, []
    any_ok = (P >= 1.0).any(axis=1)
    for name in [*SYNTH_IDS, "judge_pick"]:
        if name not in S:
            continue
        s = S[name].to_numpy()
        diff = s - oracle_case
        surplus[name] = {
            "score": _r(s.mean()),
            "surplus_vs_oracle": _r(diff.mean()),
            "surplus_ci95": boot_ci(idx, diff),
        }
        rescue = fail_all & (s >= 1.0)
        damage = any_ok & (s < 1.0)
        surplus[name]["rescue"] = int(rescue.sum())
        surplus[name]["damage"] = int(damage.sum())
        for kind, mask in (("rescue", rescue), ("damage", damage)):
            for cid in np.array(case_ids)[mask]:
                rd_rows.append({"candidate": name, "kind": kind, "case_id": int(cid)})
    pd.DataFrame(rd_rows, columns=["candidate", "kind", "case_id"]).to_csv(
        tables / "rescue_damage.csv", index=False
    )

    # --- 6. instruction level and union oracle -------------------------------------------------
    instr = instr[instr.case_id.isin(keep)]
    acc = instr.groupby("candidate").agg(
        strict=("strict_pass", "mean"), loose=("loose_pass", "mean")
    )
    instr_level = {
        name: {
            "instruction_strict": _r(acc.loc[name, "strict"]),
            "instruction_loose": _r(acc.loc[name, "loose"]),
            "prompt_loose": _r(
                instr[instr.candidate == name]
                .groupby("case_id")
                .loose_pass.all()
                .mean()
            ),
        }
        for name in names
    }
    piv = (
        instr[instr.candidate.isin(panel)]
        .pivot_table(
            index=["case_id", "instruction_id"],
            columns="candidate",
            values="strict_pass",
            aggfunc="max",
        )
        .astype(bool)
    )
    union = float(piv.any(axis=1).mean())
    best_instr = float(max(acc.loc[p, "strict"] for p in panel))
    union_block = {
        "union_oracle_instruction_strict": _r(union),
        "best_single_instruction_strict": _r(best_instr),
    }
    for name in SYNTH_IDS:
        if name in acc.index:
            denom = union - best_instr
            union_block[f"efficiency_{name}"] = (
                _r((acc.loc[name, "strict"] - best_instr) / denom)
                if denom > 0
                else "not available (union - best <= 0)"
            )

    # --- 7. paired comparisons ----------------------------------------------------------------
    best_arr = S[best_name].to_numpy()
    pairs = [
        ("synth_default", "solo_default", S.get("solo_default")),
        ("synth_default", f"best_single({best_name})", best_arr),
        ("coverage", "synth_default", S.get("synth_default")),
        ("judge_pick", "oracle", oracle_case),
    ]
    comparisons = {}
    for a_name, b_name, b_vals in pairs:
        if a_name not in S or b_vals is None:
            continue
        a = S[a_name].to_numpy()
        b = np.asarray(b_vals, dtype=float)
        d = a - b
        entry = {"mean_diff": _r(d.mean()), "ci95": boot_ci(idx, d)}
        if binary and b_name != "oracle":
            entry["mcnemar_exact"] = mcnemar_exact(a, b)
        if b_name == "oracle":
            entry["note"] = "oracle is a per-case max, McNemar not applicable"
        comparisons[f"{a_name} vs {b_name}"] = entry

    # --- 8. figures ------------------------------------------------------------------------------
    _figures(figs, names, per_cand, oracle, best, decomposition, surplus)

    summary = {
        "raw_report": raw.name,
        "raw_report_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "benchmark": report["benchmark"],
        "n_cases_analysed": n,
        "case_ids": case_ids,
        "n_excluded": len(excluded),
        "excluded_case_ids": excluded,
        "model": cfg["model"],
        "notional_price_usd_per_mtok": price,
        "candidates": per_cand,
        "decomposition": decomposition,
        "synthesis": surplus,
        "instruction_level": {**instr_level, **union_block},
        "comparisons": comparisons,
        "latency": "per-case provider latency and critical-path latency not available (free-tier "
        "gateway attributes no call latency); wall_s_per_case_mean = run duration / cases, "
        "all candidates concurrent, includes contention",
        "bootstrap": cfg["bootstrap"],
    }
    (out_root / "results" / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


_LABEL_OFFSET = {"solo_s2": (8, -13), "solo_s0": (8, 7)}


def _scatter(ax, names, per_cand, xkey, oracle, logx, xlabel) -> None:
    pts = []
    for name in names:
        c = per_cand[name]
        x, y = c[xkey], c["score"]
        lo, hi = c["score_ci95"]
        ax.errorbar(
            x,
            y,
            yerr=[[y - lo], [hi - y]],
            fmt="o",
            color=PALETTE[name],
            capsize=3,
            ms=7,
        )
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=_LABEL_OFFSET.get(name, (8, 4)),
            fontsize=8,
        )
        pts.append((x, y, name))
    if logx:
        ax.set_xscale("log")
    xmax = ax.get_xlim()[1]
    front = pareto_front(pts)
    ax.step(
        [p[0] for p in front] + [xmax],
        [p[1] for p in front] + [front[-1][1]],
        where="post",
        color="#555555",
        lw=1.2,
        label="Pareto frontier",
    )
    ax.axhline(
        oracle, ls="--", color="#555555", lw=1, label=f"selection oracle ({oracle:.2f})"
    )
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("IFEval prompt-level strict score (95% bootstrap CI)")
    _style(ax)


def _figures(figs, names, per_cand, oracle, best, dec, surplus) -> None:
    plt.rcParams["svg.hashsalt"] = "pilot"
    for stem, key, logx, label in (
        (
            "pareto_cost",
            "notional_usd_per_case",
            True,
            "notional $ per case (list price; free tier, not billed)",
        ),
        (
            "pareto_latency",
            "wall_s_per_case_mean",
            False,
            "mean wall-clock seconds per case (concurrent, includes contention)",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        _scatter(ax, names, per_cand, key, oracle, logx, label)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(
                figs / f"{stem}.{ext}",
                dpi=160,
                metadata={"Date": None} if ext == "svg" else None,
            )
        plt.close(fig)

    steps = [
        (
            "best single\n(in-sample)",
            best,
            per_cand[dec["best_single"]]["score_ci95"],
            "#0072B2",
        ),
        (
            "judge_pick",
            per_cand["judge_pick"]["score"],
            per_cand["judge_pick"]["score_ci95"],
            PALETTE["judge_pick"],
        ),
        ("oracle", oracle, dec["oracle_ci95"], "#555555"),
    ]
    for name in SYNTH_IDS:
        steps.append(
            (name, per_cand[name]["score"], per_cand[name]["score_ci95"], PALETTE[name])
        )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [s[0] for s in steps],
        [s[1] for s in steps],
        yerr=[[s[1] - s[2][0] for s in steps], [s[2][1] - s[1] for s in steps]],
        color=[s[3] for s in steps],
        capsize=4,
        error_kw={"ecolor": "#333333", "lw": 1},
    )
    for i, s in enumerate(steps):
        ax.text(
            i, 0.805, f"{s[1]:.2f}", ha="center", va="bottom", fontsize=9, color="white"
        )
    ax.set_ylim(0.8, 1.02)
    ax.axhline(dec["selection_ceiling"], ls=":", color="#555555", lw=1)
    ax.text(
        len(steps) - 0.5,
        dec["selection_ceiling"] - 0.004,
        "selection ceiling 1 - beta",
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    ax.set_ylabel("score (y-axis starts at 0.80; error bars: 95% CI)")
    _style(ax)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            figs / f"decomposition.{ext}",
            dpi=160,
            metadata={"Date": None} if ext == "svg" else None,
        )
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "ifeval_main.json"))
    main(Path(ap.parse_args().raw))
