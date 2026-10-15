"""All candidate recipes in one place (mini pilot: one model, three seeded samples)."""

from __future__ import annotations

from pathlib import Path

import screamingface as sf
import yaml

from src.prompts import COVERAGE_PROMPT, JUDGE_RUBRIC

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SOLO_IDS = ("solo_s0", "solo_s1", "solo_s2")
SOLO_IDS = ("solo_default", *SAMPLE_SOLO_IDS)
FUSION_IDS = ("judge_pick", "synth_default", "coverage")
ALL_IDS = (*SOLO_IDS, *FUSION_IDS)


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _draws(route: str, k: int, temp: float) -> list[sf.Model]:
    """k identical temperature-only requests. `seed` is not enabled on this route (the route disables it),
    so independence comes from fresh sampling at temperature > 0 with the request cache off."""
    return [sf.Model(route, params={"temperature": temp}) for _ in range(k)]


def build(cfg: dict) -> dict[str, sf.Model | sf.Fusion]:
    """Return {candidate id: Recipe}.

    The panel is three draws of the same route at the same temperature. The three `solo_s*`
    candidates send the same request as a panel member, so the selection oracle is computed over
    draws from the same distribution the fusions aggregate (but not the very same outputs: each
    candidate is its own run; see README limitations).
    """
    route = cfg["model"]
    k, temp = cfg["n_samples"], cfg["sample_temperature"]
    solos = {
        f"solo_s{i}": sf.Model(route, name=f"solo_s{i}", params={"temperature": temp})
        for i in range(k)
    }
    return {
        "solo_default": sf.Model(route, name="solo_default"),
        **solos,
        "judge_pick": sf.Fusion(
            _draws(route, k, temp),
            name="judge_pick",
            synthesizer=sf.Judge(route, returns="label", rubric=JUDGE_RUBRIC),
        ),
        "synth_default": sf.Fusion(
            _draws(route, k, temp), name="synth_default", synthesizer=route
        ),
        "coverage": sf.Fusion(
            _draws(route, k, temp),
            name="coverage",
            synthesizer=sf.Model(route, prompt=COVERAGE_PROMPT),
        ),
    }
