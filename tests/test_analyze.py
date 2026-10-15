"""Hand-checked test of src/analyze.py on a synthetic report (10 cases, known answers)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src import analyze

# per-case strict scores, cases 1..10
DRAWS = {
    "solo_s0": [1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
    "solo_s1": [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
    "solo_s2": [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
}
SCORES = {
    "solo_default": [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],  # 0.7
    **DRAWS,
    "judge_pick": [1, 1, 0, 0, 0, 1, 1, 0, 1, 1],  # 0.6
    "synth_default": [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # 0.8, rescues 4, damages 2
    "coverage": [1, 0, 0, 1, 0, 1, 1, 1, 1, 1],  # 0.7
}


def _candidate(name: str, scores: list[int]) -> dict:
    cases = []
    for i, s in enumerate(scores, start=1):
        cases.append(
            {
                "case_id": 1000 + i,
                "status": "scored",
                "grade": {
                    "score": float(s),
                    "metrics": {},
                    "checks": [
                        {
                            "id": "instruction-1",
                            "evidence": [
                                {
                                    "metadata": {"mode": "strict"},
                                    "outcome": "PASS" if s else "FAIL",
                                },
                                {
                                    "metadata": {"mode": "loose"},
                                    "outcome": "PASS" if s else "FAIL",
                                },
                            ],
                        }
                    ],
                },
                "operations": [],
            }
        )
    return {
        "name": name,
        "cases": cases,
        "coverage": 1.0,
        "score": sum(scores) / len(scores),
        "duration_ms": 10_000,
        "usage": {"input_tokens": 1000, "output_tokens": 2000, "cost_usd": None},
    }


class AnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        raw = root / "synthetic.json"
        raw.write_text(
            json.dumps(
                {
                    "benchmark": {"id": "ifeval"},
                    "candidates": [_candidate(n, s) for n, s in SCORES.items()],
                }
            ),
            encoding="utf-8",
        )
        cls.s = analyze.main(raw, root)
        cls.root, cls.raw = root, raw

    def test_decomposition(self) -> None:
        d = self.s["decomposition"]
        self.assertEqual(d["best_single"], "solo_s0")  # three-way tie at 0.6 -> first
        self.assertAlmostEqual(d["best_single_score"], 0.6)
        self.assertAlmostEqual(d["oracle"], 0.8)
        self.assertAlmostEqual(d["oracle_gain"], 0.2)
        self.assertEqual(d["beta_count"], 2)
        self.assertAlmostEqual(d["beta"], 0.2)
        self.assertAlmostEqual(d["selection_ceiling"], 0.8)
        # exact Clopper-Pearson 95% CI for 2/10
        self.assertAlmostEqual(d["beta_cp95"][0], 0.025211, places=4)
        self.assertAlmostEqual(d["beta_cp95"][1], 0.556095, places=4)
        # certificate: (1 - 0.0252) - 0.6
        self.assertAlmostEqual(
            d["zero_dollar_certificate_max_gain_over_best"], 0.374789, places=4
        )

    def test_surplus_rescue_damage(self) -> None:
        sy = self.s["synthesis"]
        self.assertAlmostEqual(sy["synth_default"]["surplus_vs_oracle"], 0.0)
        self.assertEqual(sy["synth_default"]["rescue"], 1)
        self.assertEqual(sy["synth_default"]["damage"], 1)
        self.assertAlmostEqual(sy["coverage"]["surplus_vs_oracle"], -0.1)
        self.assertEqual(sy["coverage"]["rescue"], 1)
        self.assertEqual(sy["coverage"]["damage"], 2)
        self.assertAlmostEqual(sy["judge_pick"]["surplus_vs_oracle"], -0.2)
        rd = (self.root / "results" / "tables" / "rescue_damage.csv").read_text()
        self.assertIn("synth_default,rescue,1004", rd)
        self.assertIn("synth_default,damage,1002", rd)

    def test_comparisons_and_instruction_level(self) -> None:
        c = self.s["comparisons"]["synth_default vs solo_default"]
        self.assertAlmostEqual(c["mean_diff"], 0.1)
        self.assertEqual(c["mcnemar_exact"]["only_first"], 2)
        self.assertEqual(c["mcnemar_exact"]["only_second"], 1)
        self.assertAlmostEqual(c["mcnemar_exact"]["p_value"], 1.0)
        il = self.s["instruction_level"]
        self.assertAlmostEqual(il["union_oracle_instruction_strict"], 0.8)
        self.assertAlmostEqual(il["efficiency_synth_default"], 1.0)

    def test_cost_fields(self) -> None:
        c = self.s["candidates"]["solo_default"]
        self.assertAlmostEqual(c["tokens_in_per_case"], 100.0)
        # (100 * 0.25 + 200 * 1.50) / 1e6
        self.assertAlmostEqual(c["notional_usd_per_case"], 325e-6, places=9)
        self.assertAlmostEqual(c["wall_s_per_case_mean"], 1.0)

    def test_deterministic_byte_for_byte(self) -> None:
        summary_path = self.root / "results" / "summary.json"
        first = summary_path.read_bytes()
        analyze.main(self.raw, self.root)
        self.assertEqual(first, summary_path.read_bytes())

    def test_excludes_ungraded_cases(self) -> None:
        data = json.loads(self.raw.read_text(encoding="utf-8"))
        data["candidates"][4]["cases"][0]["status"] = "failed"
        data["candidates"][4]["cases"][0]["grade"] = None
        raw = self.root / "with_failure.json"
        raw.write_text(json.dumps(data), encoding="utf-8")
        s = analyze.main(raw, self.root / "x")
        self.assertEqual(s["n_cases_analysed"], 9)
        self.assertEqual(s["excluded_case_ids"], [1001])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
