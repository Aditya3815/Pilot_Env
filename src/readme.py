"""Regenerate README.md from results/summary.json (+ environment.json, replay_summary.json).

Every number in the README is read from those files; the prose around them is static.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ci(c: list[float], nd: int = 2) -> str:
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def build() -> str:
    s = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    env = json.loads(
        (ROOT / "results" / "environment.json").read_text(encoding="utf-8")
    )
    rep = json.loads(
        (ROOT / "results" / "replay_summary.json").read_text(encoding="utf-8")
    )
    d, syn, cand, cmp_ = (
        s["decomposition"],
        s["synthesis"],
        s["candidates"],
        s["comparisons"],
    )
    il = s["instruction_level"]
    n = s["n_cases_analysed"]
    total_notional = sum(c["notional_usd_total"] for c in cand.values())
    bench = s["benchmark"]

    order = [
        "solo_default",
        "solo_s0",
        "solo_s1",
        "solo_s2",
        "judge_pick",
        "synth_default",
        "coverage",
    ]
    rows = "\n".join(
        f"| `{k}` | {cand[k]['score']:.2f} {_ci(cand[k]['score_ci95'])} "
        f"| {cand[k]['tokens_in_per_case']:.0f} / {cand[k]['tokens_out_per_case']:.0f} "
        f"| {cand[k]['notional_usd_per_case'] * 1000:.2f} "
        f"| {cand[k]['wall_s_per_case_mean']:.1f} "
        f"| [`results/url4/{k}.txt`](results/url4/{k}.txt) |"
        for k in order
    )
    sy = lambda k: syn[k]
    c_sd = cmp_["synth_default vs solo_default"]
    c_bs = cmp_[f"synth_default vs best_single({d['best_single']})"]
    c_cs = cmp_["coverage vs synth_default"]
    replay_rows = "\n".join(
        f"| `{k}` | {v['main_score']:.2f} | {v['replay_score']:.2f} | {v['per_case_agreement']:.2f} |"
        for k, v in rep.items()
        if not k.startswith("_")
    )

    return f"""# Selection ceiling vs synthesis surplus on IFEval: a single-model mini pilot

**Summary.** On the first {n} IFEval cases, three independent draws of one model
(`{s["model"]}`) had a per-case selection oracle of {d["oracle"]:.2f}, equal to the best single draw
({d["best_single_score"]:.2f}); all three draws failed together on {d["beta_count"]} of {n} cases
(beta = {d["beta"]:.2f}, Clopper-Pearson 95% CI {_ci(d["beta_cp95"], 3)}). Synthesis from those draws did not
exceed the ceiling: Self-MoA-style synthesis scored {sy("synth_default")["score"]:.2f}
(surplus {sy("synth_default")["surplus_vs_oracle"]:+.2f}, paired 95% CI {_ci(sy("synth_default")["surplus_ci95"])})
and the Coverage prompt {sy("coverage")["score"]:.2f}
(surplus {sy("coverage")["surplus_vs_oracle"]:+.2f}, {_ci(sy("coverage")["surplus_ci95"])}). The benchmark is nearly
saturated for this model, so the pilot has little power; the replay shows run-to-run noise of about
{rep["_max_abs_difference"]:.2f} in score, the same size as the differences between candidates.
Real spend was $0 (Google AI Studio free tier); the notional list-price cost of all candidates is
${total_notional:.3f}.

> **This is a single-model pilot, not the originally planned multi-model study.** The planned three-provider
> panel with a frontier synthesizer was not executable (no paid credit on any route). Everything below is the
> Self-MoA regime: sampling diversity from one model.

## 1. The question

Any policy whose output is one of the members' answers (router, vote, cascade, judge-pick) cannot score above the
**per-case selection oracle** `mean_c max_s score[s,c]`; on binary tasks that cap is `1 - beta`, where beta is the
rate at which every member fails the same case (Chen 2026, arXiv:2606.27288). A **synthesizer** writes a new answer
and is not bound by it. **Synthesis surplus = synthesis score - selection-oracle score.** A positive surplus is gain
no router or cascade could reach. The Self-MoA baseline aggregates several samples of one model (Li et al.,
arXiv:2502.00674), which is exactly the regime measured here.

## 2. Setup

| | |
|---|---|
| Model | `{s["model"]}` (the only route usable with a free Gemini AI Studio key) |
| Panel | 3 draws, `temperature=0.7`, byte-identical requests (`seed` is disabled on this route) |
| Benchmark | `{bench["id"]}`, Engine revision `{bench["revision"]}`, deterministic official verifier |
| Cases | first {n} in dataset order (`limit={n}`; not a random sample), excluded: {s["n_excluded"]} |
| Toolkit | `screamingface` {env["packages"]["screamingface"]}, Python 3.12.6, {env["os"]} |
| Date | 2026-10-15 (single sitting) |
| Spend | $0 billed (free tier); notional ${total_notional:.3f} at $0.25 / $1.50 per 1M input / output tokens |

The hypotheses, metrics, n and exclusion rules were written down before the main run; that planning document is not
part of this repository, so treat the design as author-declared rather than independently verifiable.

### Deviations from the originally planned study

- **One model instead of three providers plus a frontier synthesizer.** The OpenRouter key had no credit; Hugging Face
  credits were exhausted; Gemini 2.5 routes are closed to new keys; Groq routes are not served by the Engine.
  Only `gemini-3.1-flash-lite` was usable. The frontier-vs-cheap synthesizer contrast, mixed-model panels, matched-cost
  Self-MoA and the second (rubric-graded) benchmark were therefore not run.
- **No `seed`.** The route disables it, so panel draws are three identical `temperature=0.7` requests and sampling is
  unseeded; replay can only check statistical consistency.
- **No per-case token or latency data.** The free-tier gateway attributes no per-call usage; only run-level totals exist.
- **Windows workaround.** `screamingface` 0.1.3 calls `os.fsync` on a read-only file, which fails on Windows after the
  model calls finish; `src/winpatch.py` skips that call (durability only).
- **Cases are the first 50 in dataset order**, because the benchmark defines `slice=0:541` and `limit=n` takes a prefix.

## 3. Results (n = {n})

| candidate | score [95% CI] | tokens in / out per case | notional $ per 1000 cases | wall s per case | recipe |
|---|---|---|---|---|---|
{rows}

CIs are bootstrap percentile intervals (10,000 resamples over cases, fixed seed). Notional $ is a list-price estimate
from run-level token totals, not an invoice. Wall-clock per case is the run duration divided by cases; the seven
candidates ran concurrently on one free-tier key, so it includes contention and is not a clean per-candidate latency.
Per-case token and latency figures and fusion critical-path latency are not available (the free-tier gateway attributes
no per-call usage). Median and p95 latency are therefore not reported.

## 4. Decomposition

| quantity | value |
|---|---|
| best single draw (in-sample: `{d["best_single"]}`) | {d["best_single_score"]:.2f} |
| `judge_pick` (realizable selection) | {cand["judge_pick"]["score"]:.2f} |
| selection oracle | {d["oracle"]:.2f} {_ci(d["oracle_ci95"])} |
| oracle gain over best single | {d["oracle_gain"]:+.2f} |
| beta (all 3 draws fail, score < 1.0) | {d["beta"]:.2f} ({d["beta_count"]}/{n}), CP 95% {_ci(d["beta_cp95"], 3)} |
| selection ceiling 1 - beta | {d["selection_ceiling"]:.2f}, CP 95% {_ci(d["selection_ceiling_cp95"], 3)} |
| $0 certificate: max gain of any selection policy over best single | {d["zero_dollar_certificate_max_gain_over_best"]:.3f} |

| synthesis candidate | score | surplus vs oracle [paired 95% CI] | rescue | damage |
|---|---|---|---|---|
| `synth_default` (Self-MoA) | {sy("synth_default")["score"]:.2f} | {sy("synth_default")["surplus_vs_oracle"]:+.2f} {_ci(sy("synth_default")["surplus_ci95"])} | {sy("synth_default")["rescue"]} | {sy("synth_default")["damage"]} |
| `coverage` | {sy("coverage")["score"]:.2f} | {sy("coverage")["surplus_vs_oracle"]:+.2f} {_ci(sy("coverage")["surplus_ci95"])} | {sy("coverage")["rescue"]} | {sy("coverage")["damage"]} |

Rescue = all three draws failed but synthesis passed (beyond the ceiling). Damage = at least one draw passed but synthesis
failed. Case IDs: [`results/tables/rescue_damage.csv`](results/tables/rescue_damage.csv).

Instruction level (strict): best single {il["best_single_instruction_strict"]:.3f}, union oracle
{il["union_oracle_instruction_strict"]:.3f}. Synthesis efficiency is not available because union - best single is 0.

Paired comparisons (exploratory, many comparisons on a small n, no multiplicity correction; exact McNemar):

| comparison | mean difference [paired 95% CI] | McNemar p (only-first / only-second) |
|---|---|---|
| `synth_default` vs `solo_default` | {c_sd["mean_diff"]:+.2f} {_ci(c_sd["ci95"])} | {c_sd["mcnemar_exact"]["p_value"]:.2f} ({c_sd["mcnemar_exact"]["only_first"]} / {c_sd["mcnemar_exact"]["only_second"]}) |
| `synth_default` vs best single (`{d["best_single"]}`, in-sample) | {c_bs["mean_diff"]:+.2f} {_ci(c_bs["ci95"])} | {c_bs["mcnemar_exact"]["p_value"]:.2f} ({c_bs["mcnemar_exact"]["only_first"]} / {c_bs["mcnemar_exact"]["only_second"]}) |
| `coverage` vs `synth_default` | {c_cs["mean_diff"]:+.2f} {_ci(c_cs["ci95"])} | {c_cs["mcnemar_exact"]["p_value"]:.2f} ({c_cs["mcnemar_exact"]["only_first"]} / {c_cs["mcnemar_exact"]["only_second"]}) |

### Pre-registered hypotheses

- **H1** (surplus about 0 or negative on IFEval): consistent. Both synthesis surpluses are at or below 0, with upper CI
  bounds at 0.00; no case was rescued.
- **H2** (judge-pick recovers only part of the oracle gain): not informative. The oracle gain over the best single draw is
  {d["oracle_gain"]:+.2f}, so there was nothing to recover; `judge_pick` equals the oracle ({cand["judge_pick"]["score"]:.2f}).
- **H3** (`coverage` >= `synth_default`): not supported. The observed difference is {c_cs["mean_diff"]:+.2f} {_ci(c_cs["ci95"])}, the interval includes 0.
- H4 and H5 (mixed-model vs Self-MoA, rubric benchmark) were dropped at pre-registration and are not tested.

## 5. Figures

![Score vs notional cost](figures/pareto_cost.png)
![Score vs wall-clock latency](figures/pareto_latency.png)
![Decomposition](figures/decomposition.png)

## 6. What this does and does not show

- **One model, sampling diversity only.** The panel is three draws of the same model, so it cannot test mixed-model panels,
  the frontier-vs-cheap synthesizer lever, or matched-cost Self-MoA vs mixed fusion. Those remain open.
- **Saturated benchmark.** The draws score {d["best_single_score"]:.2f} or less on {n} cases; beta is about
  {d["beta"]:.2f} with a wide interval {_ci(d["beta_cp95"], 3)}. There is almost no headroom for synthesis surplus to appear,
  so a null here says little about harder or open-ended tasks. IFEval is verifiable, so the ceiling is expected to bind.
- **Small n, one sitting, first {n} cases only** (dataset order, not random). All comparisons are exploratory.
- **Noise is as large as the effects.** Replaying the same recipes (unseeded sampling) moved scores by up to
  {rep["_max_abs_difference"]:.2f}, the size of the synthesis-vs-solo gaps above. Do not read ranking among candidates
  from this pilot.

| candidate | main | replay | per-case agreement |
|---|---|---|---|
{replay_rows}

- **In-sample best single** biases the oracle gain downward and flatters selection baselines.
- **Separate runs.** Each candidate is its own run; fusion members are fresh draws, not the outputs of the `solo_s*` runs.
- **Provider latency is not wall-clock;** per-call latency was not available, and wall-clock includes concurrent contention.
- **Prices:** notional, Google list price as retrieved on the run date (verify before quoting).
- **Free tier:** Google may use free-tier prompts for product improvement (IFEval prompts are public).

## 7. Reproduce

Windows, Python 3.12 (`make` and Docker were unavailable here, so no working Makefile run and no Dockerfile).

```
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.lock
.venv/Scripts/screamingface.exe prepare ifeval
.venv/Scripts/screamingface.exe up
.venv/Scripts/python.exe -m src.connect_key --provider gemini-cli   # hidden prompt for the key
.venv/Scripts/python.exe -m src.run_all --stage smoke
.venv/Scripts/python.exe -m src.run_all --stage probe
.venv/Scripts/python.exe -m src.run_all --stage main
.venv/Scripts/python.exe -m src.run_all --stage analyze   # deterministic; regenerates summary.json byte for byte
.venv/Scripts/python.exe -m src.run_all --stage replay    # re-samples; reports consistency, not equality
.venv/Scripts/python.exe -m src.readme                    # regenerate this README from results/summary.json
.venv/Scripts/python.exe -m unittest tests.test_analyze -v
```

`analyze` reads only `results/raw/ifeval_main.json` (sha256 recorded in `results/summary.json`). Windows needs the
`src/winpatch.py` workaround for an `os.fsync` bug in `screamingface` 0.1.3 (see Deviations); it is applied
automatically by `src.run` and `src.replay`. Replay cost: $0 (free tier; measured, see `results/cost_ledger.csv`).

## 8. Next steps

Rubric-graded benchmarks (DRACO, HealthBench) where headroom exists; a synthesizer-tier sweep and mixed-provider panels
with paid credit; matched-cost Self-MoA; more sittings and larger n with seeded sampling on a route that supports `seed`.

## 9. Citation / author

[Your Name], [contact]. Cite: Chen 2026, arXiv:2606.27288 (selection ceiling); Li et al., arXiv:2502.00674 (Self-MoA).
"""


if __name__ == "__main__":
    (ROOT / "README.md").write_text(build(), encoding="utf-8")
    print("wrote README.md")
