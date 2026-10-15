"""Connect a provider key through the toolkit's non-interactive mechanism.

Run this yourself in a terminal; the key is read with getpass, so it is never echoed,
logged, written to disk by this project, or passed on a command line.
"""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

import screamingface as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini-cli")
    provider = ap.parse_args().provider
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    sf.configure(engine_url=cfg["engine_url"], scoreboard_url=cfg["scoreboard_url"])
    key = getpass.getpass(f"{provider} API key (input hidden): ").strip()
    sf.connect(provider, api_key=key, method="api_key")
    conn = sf.connections.get(provider)
    print(f"{provider} status: {conn.status} (method={conn.auth_method})")


if __name__ == "__main__":
    main()
