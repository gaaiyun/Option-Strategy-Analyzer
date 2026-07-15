from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_price_cli_rejects_negative_spot():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "__main__.py"), "price",
         "--S", "-100", "--K", "100", "--T", "0.25", "--sigma", "0.2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode != 0
    assert "error" in completed.stderr.lower()


def test_require_llm_returns_nonzero_when_key_is_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "__main__.py"), "advise",
         "--view", "neutral_range", "--hv-30", "0.2", "--iv", "0.3",
         "--view-text", "range", "--use-llm", "--require-llm"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode != 0
    assert "llm" in completed.stderr.lower()


def test_scenario_cli_outputs_strict_json_report():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "__main__.py"), "scenario",
         "--view", "neutral_range", "--S", "100", "--T", "0.12",
         "--sigma", "0.25", "--hv-30", "0.20", "--iv", "0.30"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)
    assert report["strategy_name"] == "Iron Condor"
    assert report["max_profit_unbounded"] is False
    assert len(report["scenario_pnl"]) == 5
