"""data_fetch.py 测试 —— 不发网络。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_fetch import (
    MarketContext,
    _IV_PROXIES,
    annualized_volatility,
    fetch_history,
    fetch_iv_proxy,
    fetch_market_context,
    fetch_spot,
    synthetic_market_context,
)


# --- synthetic_market_context -----------------------------------------------

def test_synthetic_default_shape():
    ctx = synthetic_market_context()
    assert isinstance(ctx, MarketContext)
    assert ctx.source == "synthetic"
    assert ctx.symbol == "FAKE"
    assert ctx.spot == 100.0
    assert ctx.iv_proxy_source == "synthetic"


def test_synthetic_custom():
    ctx = synthetic_market_context(symbol="AAPL", spot=180,
                                    hv30=0.4, hv60=0.45, iv=0.35)
    assert ctx.symbol == "AAPL"
    assert ctx.spot == 180
    assert ctx.historical_vol_30d == 0.4
    assert ctx.iv_proxy == 0.35


def test_market_context_to_dict():
    ctx = synthetic_market_context()
    d = ctx.to_dict()
    assert d["symbol"] == "FAKE"
    assert d["spot"] == 100.0
    assert "historical_vol_30d" in d


# --- annualized_volatility --------------------------------------------------

def test_annualized_volatility_constant_prices_zero():
    """常数价 → 波动率 = 0。"""
    closes = pd.Series([100.0] * 60)
    vol = annualized_volatility(closes, window=30)
    assert vol == 0.0


def test_annualized_volatility_uses_log_returns():
    """简单数据：std × √252 应在合理量级。"""
    rng = np.random.RandomState(42)
    closes = pd.Series(100 * np.exp(rng.normal(0, 0.01, 100).cumsum()))
    vol = annualized_volatility(closes, window=30)
    # 0.01 daily std × √252 ≈ 15.9% 年化
    assert 0.10 < vol < 0.25


def test_annualized_volatility_too_short():
    closes = pd.Series([100, 101])
    vol = annualized_volatility(closes, window=30)
    # 至少不崩，返回有限值
    assert isinstance(vol, float)
    assert np.isfinite(vol)


def test_annualized_volatility_handles_single_point():
    closes = pd.Series([100.0])
    vol = annualized_volatility(closes, window=30)
    assert vol == 0.0


# --- yfinance（mock）-------------------------------------------------------

def _fake_yf_df(prices, columns=("close",)):
    """构造 yfinance 风格 DataFrame。"""
    df = pd.DataFrame({col: prices for col in columns})
    df.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in df.columns])
    return df


def test_fetch_spot_parses_response():
    fake = _fake_yf_df([170, 171, 172, 173, 174])
    with patch("data_fetch._require_yfinance") as mock_yf_mod:
        mock_yf = mock_yf_mod.return_value
        mock_yf.download.return_value = fake
        spot, change_pct = fetch_spot("AAPL")
    assert spot == 174.0
    # 174/173 - 1 = 0.578%
    assert abs(change_pct - 0.578) < 0.01


def test_fetch_spot_raises_on_empty():
    with patch("data_fetch._require_yfinance") as mock_yf_mod:
        mock_yf = mock_yf_mod.return_value
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="没数据"):
            fetch_spot("AAPL")


def test_fetch_spot_raises_when_too_few_points():
    fake = pd.DataFrame({"close": [170.0]})
    fake.columns = pd.MultiIndex.from_tuples([("close", "AAPL")])
    with patch("data_fetch._require_yfinance") as mock_yf_mod:
        mock_yf = mock_yf_mod.return_value
        mock_yf.download.return_value = fake
        with pytest.raises(RuntimeError, match="数据点不足"):
            fetch_spot("AAPL")


def test_fetch_history_returns_df_with_close():
    fake = _fake_yf_df([100, 101, 102, 103, 104])
    with patch("data_fetch._require_yfinance") as mock_yf_mod:
        mock_yf = mock_yf_mod.return_value
        mock_yf.download.return_value = fake
        df = fetch_history("AAPL", days=10)
    assert "close" in df.columns
    assert len(df) == 5


def test_fetch_history_empty_raises():
    with patch("data_fetch._require_yfinance") as mock_yf_mod:
        mock_yf = mock_yf_mod.return_value
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="没数据"):
            fetch_history("AAPL")


def test_iv_proxies_table_has_common_tickers():
    for t in ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]:
        assert t in _IV_PROXIES
        assert _IV_PROXIES[t] == "^VIX"


def test_fetch_iv_proxy_returns_none_for_unknown():
    iv, src = fetch_iv_proxy("UNKNOWN_TICKER_X")
    assert iv is None
    assert src is None


def test_fetch_iv_proxy_converts_vix_to_decimal():
    """VIX = 30 (percentage points) → 0.30 (decimal)。"""
    fake = pd.DataFrame({"close": [25.0, 30.0]})
    fake.columns = pd.MultiIndex.from_tuples([("close", "^VIX")])
    with patch("data_fetch._require_yfinance") as mock_yf_mod:
        mock_yf = mock_yf_mod.return_value
        mock_yf.download.return_value = fake
        iv, src = fetch_iv_proxy("AAPL")
    assert iv == 0.30
    assert src == "^VIX"


def test_require_yfinance_raises_when_missing(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "yfinance":
            raise ImportError("simulated")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from data_fetch import _require_yfinance
    with pytest.raises(ImportError, match="yfinance"):
        _require_yfinance()


# --- fetch_market_context（端到端 mock）----------------------------------

def test_fetch_market_context_assembles_all_pieces():
    """spot + history + iv 串到一起，输出 MarketContext。"""
    # 现价 + history 都用同一个 yfinance mock 返回
    rng = np.random.RandomState(0)
    closes = 100 * np.exp(rng.normal(0, 0.01, 90).cumsum())
    history_df = pd.DataFrame({
        "close": closes, "open": closes, "high": closes,
        "low": closes, "volume": [1e6] * 90,
    })
    history_df.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in history_df.columns])

    spot_df = pd.DataFrame({"close": list(closes[-5:])})
    spot_df.columns = pd.MultiIndex.from_tuples([("close", "AAPL")])

    iv_df = pd.DataFrame({"close": [20.0, 22.0]})
    iv_df.columns = pd.MultiIndex.from_tuples([("close", "^VIX")])

    # download 被调用 3 次：spot / history / iv
    call_idx = {"i": 0}
    def fake_download(symbol, **kw):
        call_idx["i"] += 1
        if symbol == "^VIX":
            return iv_df
        if "period" in kw:    # spot
            return spot_df
        return history_df

    with patch("data_fetch._require_yfinance") as mock_yf_mod:
        mock_yf = mock_yf_mod.return_value
        mock_yf.download.side_effect = fake_download
        ctx = fetch_market_context("AAPL", history_days=90)

    assert ctx.symbol == "AAPL"
    assert ctx.spot > 0
    assert 0 < ctx.historical_vol_30d < 1   # 年化波动 < 100%
    assert ctx.iv_proxy == 0.22
    assert ctx.iv_proxy_source == "^VIX"
