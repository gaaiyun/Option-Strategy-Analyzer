"""strategy_advisor.py 测试 —— mock LLM。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_advisor import (
    MARKET_VIEWS,
    StrategyRecommendation,
    _LLMClient,
    _STRATEGY_LIBRARY,
    _advise_llm,
    _heuristic,
    _strip_fences,
    advise,
)


# --- 规则启发式 -------------------------------------------------------------

def test_strong_bullish_picks_long_call():
    rec = _heuristic("strong_bullish", hv_30=0.25, iv=0.22, has_underlying=False)
    assert rec["strategy_name"] == "Long Call"


def test_strong_bearish_picks_long_put():
    rec = _heuristic("strong_bearish", hv_30=0.25, iv=0.22, has_underlying=False)
    assert rec["strategy_name"] == "Long Put"


def test_moderate_bullish_picks_bull_call_spread():
    rec = _heuristic("moderate_bullish", hv_30=0.25, iv=0.22, has_underlying=False)
    assert rec["strategy_name"] == "Bull Call Spread"


def test_moderate_bearish_picks_bear_put_spread():
    rec = _heuristic("moderate_bearish", hv_30=0.25, iv=0.22, has_underlying=False)
    assert rec["strategy_name"] == "Bear Put Spread"


def test_high_volatility_with_rich_iv_picks_iron_condor():
    """IV > HV30 × 1.1 → 卖波动率铁鹰。"""
    rec = _heuristic("high_volatility", hv_30=0.20, iv=0.30, has_underlying=False)
    assert rec["strategy_name"] == "Iron Condor"


def test_high_volatility_low_iv_picks_straddle():
    """IV ≤ HV30 → 买跨式。"""
    rec = _heuristic("high_volatility", hv_30=0.30, iv=0.25, has_underlying=False)
    assert rec["strategy_name"] == "Long Straddle"


def test_neutral_range_with_underlying_picks_covered_call():
    """有标的 + 横盘 + IV 不富裕 → 备兑。"""
    rec = _heuristic("neutral_range", hv_30=0.30, iv=0.25, has_underlying=True)
    assert rec["strategy_name"] == "Covered Call"


def test_neutral_range_rich_iv_picks_iron_condor():
    rec = _heuristic("neutral_range", hv_30=0.20, iv=0.30, has_underlying=True)
    assert rec["strategy_name"] == "Iron Condor"


def test_hedging_with_underlying_picks_protective_put():
    rec = _heuristic("hedging_protection", hv_30=0.25, iv=0.22, has_underlying=True)
    assert rec["strategy_name"] == "Protective Put"


def test_hedging_without_underlying_picks_long_put():
    rec = _heuristic("hedging_protection", hv_30=0.25, iv=0.22, has_underlying=False)
    assert rec["strategy_name"] == "Long Put"


def test_neutral_range_low_iv_no_underlying_picks_iron_condor():
    """无标的 + 横盘 + IV 不富裕 → 仍是铁鹰（宽适用）。"""
    rec = _heuristic("neutral_range", hv_30=0.30, iv=0.25, has_underlying=False)
    assert rec["strategy_name"] == "Iron Condor"


# --- _strip_fences ---------------------------------------------------------

def test_strip_fences_json_block():
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fences_bare_block():
    assert _strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fences_no_block_returns_as_is():
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


# --- advise（综合 + mock LLM）---------------------------------------------

def test_advise_without_llm_uses_heuristic():
    rec = advise(
        market_view="strong_bullish", hv_30=0.25, iv=0.22,
        has_underlying=False,
    )
    assert isinstance(rec, StrategyRecommendation)
    assert rec.backend == "heuristic"
    assert rec.strategy_name == "Long Call"


def test_advise_with_no_llm_no_text_uses_heuristic():
    rec = advise(market_view="neutral_range", hv_30=0.20, iv=0.30,
                 view_text=None, backend=None)
    assert rec.backend == "heuristic"


def test_advise_to_dict_serializable():
    import json
    rec = advise(market_view="strong_bullish", hv_30=0.25, iv=0.22)
    d = rec.to_dict()
    s = json.dumps(d, ensure_ascii=False)
    assert "Long Call" in s


def test_advise_with_mocked_llm():
    client = _LLMClient(backend="deepseek", api_key="sk-test")
    client.chat = MagicMock(return_value=
        '{"strategy_name": "Iron Condor", '
        '"parameters": {"expiry_days": 30}, '
        '"rationale": "横盘且 IV 高，卖波动率"}'
    )
    rec = advise(market_view="neutral_range", hv_30=0.20, iv=0.30,
                 view_text="股价应该在 100-110 区间震荡",
                 llm_client=client)
    assert rec.strategy_name == "Iron Condor"
    assert "llm" in rec.backend
    assert rec.parameters["expiry_days"] == 30


def test_advise_llm_falls_back_on_bad_json():
    client = _LLMClient(backend="deepseek", api_key="sk-test")
    client.chat = MagicMock(return_value="LLM 没出合规 JSON")
    rec = advise(market_view="strong_bullish", hv_30=0.25, iv=0.22,
                 view_text="bullish", llm_client=client)
    # 退化到 heuristic
    assert rec.backend == "heuristic"
    assert rec.strategy_name == "Long Call"


def test_advise_llm_handles_code_fence():
    client = _LLMClient(backend="openai", api_key="sk-test")
    client.chat = MagicMock(return_value=
        '```json\n{"strategy_name": "Long Straddle", '
        '"parameters": {}, "rationale": "x"}\n```'
    )
    rec = advise(market_view="high_volatility", hv_30=0.25, iv=0.22,
                 view_text="x", llm_client=client)
    assert rec.strategy_name == "Long Straddle"


def test_advise_llm_without_view_text_uses_heuristic():
    """LLM 路径需要 view_text；缺时应直接走规则。"""
    client = _LLMClient(backend="deepseek", api_key="sk-test")
    client.chat = MagicMock(return_value='{"strategy_name": "Long Call"}')
    rec = advise(market_view="moderate_bearish", hv_30=0.25, iv=0.22,
                 view_text=None, llm_client=client)
    # 没传 view_text → 不调 LLM
    assert rec.backend == "heuristic"


# --- _LLMClient ------------------------------------------------------------

def test_llm_client_is_available_with_key():
    c = _LLMClient(backend="deepseek", api_key="sk-test")
    assert c.is_available()


def test_llm_client_is_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = _LLMClient(backend="deepseek")
    assert not c.is_available()


def test_llm_client_default_models():
    assert _LLMClient(backend="openai", api_key="x").model == "gpt-4o-mini"
    assert _LLMClient(backend="anthropic", api_key="x").model == "claude-3-5-haiku-20241022"
    assert _LLMClient(backend="deepseek", api_key="x").model == "deepseek-chat"


# --- 注册表 / 枚举 ---------------------------------------------------------

def test_market_views_list_complete():
    expected = {"strong_bullish", "moderate_bullish", "strong_bearish",
                "moderate_bearish", "neutral_range", "high_volatility",
                "hedging_protection"}
    assert set(MARKET_VIEWS) == expected


def test_strategy_library_has_nine_strategies():
    assert len(_STRATEGY_LIBRARY) >= 9
    for s in ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread",
              "Long Straddle", "Long Strangle", "Iron Condor",
              "Covered Call", "Protective Put"]:
        assert s in _STRATEGY_LIBRARY
