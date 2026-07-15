"""标的物市场数据 + 历史波动率 + IV 代理。

v1 让用户手工填 ``S`` / ``r`` / ``sigma``：对学术演示够，但**实际投资者经常
不知道当前 IV**（隐含波动率），导致 BSM 跑出来的"理论价格"和市场报价差一截。

v2 加几个免 API key 的获取入口：

- ``fetch_spot(symbol)`` —— 当前标的现价 + 24h 涨跌
- ``fetch_history(symbol, days)`` —— 历史日线收盘，喂给 volatility_analyzer
- ``fetch_market_context(symbol)`` —— 综合包：spot + 历史波动率 + IV proxy（VIX
  对美股 / BVOL 对 BTC）

数据源用 yfinance（免费、无 key、自带限流）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketContext:
    symbol: str
    spot: float
    change_24h_pct: float
    historical_vol_30d: float       # 年化历史波动率（30 日窗口）
    historical_vol_60d: float
    iv_proxy: Optional[float] = None  # VIX / BVOL 等代理（年化）
    iv_proxy_source: Optional[str] = None
    source: str = "yfinance"
    as_of: Optional[str] = None
    market_data_as_of: Optional[str] = None
    fetched_at: Optional[str] = None
    price_change_period: str = "previous_close_to_latest_close"
    history_observations: int = 0
    iv_proxy_scope: Optional[str] = None
    iv_proxy_comparable: bool = False
    option_expiry: Optional[str] = None
    option_dte: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "spot": float(self.spot),
            "change_24h_pct": float(self.change_24h_pct),
            "historical_vol_30d": float(self.historical_vol_30d),
            "historical_vol_60d": float(self.historical_vol_60d),
            "iv_proxy": (float(self.iv_proxy)
                         if self.iv_proxy is not None else None),
            "iv_proxy_source": self.iv_proxy_source,
            "source": self.source,
            "as_of": self.as_of,
            "market_data_as_of": self.market_data_as_of,
            "fetched_at": self.fetched_at,
            "price_change_period": self.price_change_period,
            "history_observations": self.history_observations,
            "iv_proxy_scope": self.iv_proxy_scope,
            "iv_proxy_comparable": self.iv_proxy_comparable,
            "option_expiry": self.option_expiry,
            "option_dte": self.option_dte,
        }


def _require_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError as e:
        raise ImportError(
            "yfinance 未装。pip install yfinance 后再调用本模块。"
        ) from e


def fetch_history(symbol: str, days: int = 90) -> pd.DataFrame:
    """抓 OHLCV 历史日线。"""
    if not isinstance(days, int) or days <= 0:
        raise ValueError("days must be a positive integer")
    yf = _require_yfinance()
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance 没数据：{symbol} 最近 {days} 天")
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    return df


def fetch_spot(symbol: str, include_as_of: bool = False) -> tuple:
    """抓现价 + 24h 涨跌。"""
    yf = _require_yfinance()
    df = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance 没数据：{symbol}")
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    closes = df["close"].dropna()
    if len(closes) < 2:
        raise RuntimeError(f"{symbol} 数据点不足")
    spot = float(closes.iloc[-1])
    change = float(closes.iloc[-1] / closes.iloc[-2] - 1)
    index_value = closes.index[-1]
    market_as_of = index_value.isoformat() if hasattr(index_value, "isoformat") else None
    if include_as_of:
        return spot, change * 100, market_as_of
    return spot, change * 100


def annualized_volatility(closes: pd.Series, window: int = 30,
                          require_full_window: bool = False) -> float:
    """从收盘价算年化历史波动率。"""
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer greater than one")
    if require_full_window and len(closes) < window + 1:
        raise ValueError(
            f"need at least {window + 1} closes for a {window}-return estimate"
        )
    if len(closes) < window:
        window = max(2, len(closes))
    log_rets = np.log(closes / closes.shift(1)).dropna()
    if len(log_rets) < 2:
        return 0.0
    return float(log_rets.tail(window).std() * np.sqrt(252))


# IV 代理对应表 ----------------------------------------------------------------

_IV_PROXIES = {
    # 美股 → VIX
    "SPY": "^VIX", "QQQ": "^VIX",
    "AAPL": "^VIX", "MSFT": "^VIX", "GOOG": "^VIX", "GOOGL": "^VIX",
    "AMZN": "^VIX", "META": "^VIX", "TSLA": "^VIX", "NVDA": "^VIX",
    # 加密 → BVOL（Volmex BTC 30 日隐含波动率，yfinance ticker = BVOLUSD-X 不稳定）
    # 这里给 yfinance 能拉得到的 VIX 作为通用回退
}


def fetch_iv_proxy(symbol: str) -> tuple:
    """抓 IV 代理（VIX 等）。返回 (annualized_iv_pct_decimal, source) or (None, None)。"""
    proxy_symbol = _IV_PROXIES.get(symbol.upper())
    if not proxy_symbol:
        return None, None
    try:
        yf = _require_yfinance()
        df = yf.download(proxy_symbol, period="5d", progress=False,
                         auto_adjust=True)
        if df is None or df.empty:
            return None, None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        # VIX 已经是年化波动率百分点（30 = 30%），转小数
        return float(df["close"].dropna().iloc[-1] / 100), proxy_symbol
    except Exception:
        return None, None


def fetch_atm_iv(symbol: str, spot: float, target_days: int = 30,
                 max_expiry_deviation_days: int = 14) -> tuple:
    """从标的自身期权链估计近月 ATM IV。

    返回 ``(iv, source, expiry, dte)``。只使用最接近平值的 call 与 put 的有效
    impliedVolatility 中位数；数据不可用时抛出 RuntimeError，由上层决定是否退回
    到不可直接比较的宽基指数代理。
    """
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be a positive finite number")
    if not isinstance(target_days, int) or target_days <= 0:
        raise ValueError("target_days must be a positive integer")
    if not isinstance(max_expiry_deviation_days, int) or max_expiry_deviation_days < 0:
        raise ValueError("max_expiry_deviation_days must be a non-negative integer")

    yf = _require_yfinance()
    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options)
    if not expiries:
        raise RuntimeError(f"no option expiries for {symbol}")

    target = datetime.now(timezone.utc).date() + timedelta(days=target_days)
    parsed = []
    for expiry in expiries:
        try:
            date = datetime.strptime(expiry, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if date >= datetime.now(timezone.utc).date():
            parsed.append((abs((date - target).days), expiry))
    if not parsed:
        raise RuntimeError(f"no future option expiry for {symbol}")
    deviation, expiry = min(parsed)
    if deviation > max_expiry_deviation_days:
        raise RuntimeError(
            f"nearest option expiry is {deviation} days from the {target_days}-day target"
        )
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    dte = (expiry_date - datetime.now(timezone.utc).date()).days

    chain = ticker.option_chain(expiry)
    estimates = []
    for frame in (chain.calls, chain.puts):
        required = {"strike", "impliedVolatility"}
        if frame is None or frame.empty or not required.issubset(frame.columns):
            continue
        valid = frame.loc[
            np.isfinite(frame["strike"])
            & np.isfinite(frame["impliedVolatility"])
            & (frame["impliedVolatility"] > 0)
            & (frame["impliedVolatility"] <= 5)
        ].copy()
        if valid.empty:
            continue
        liquidity = None
        if {"bid", "ask"}.issubset(valid.columns):
            liquidity = (valid["bid"].fillna(0) > 0) | (valid["ask"].fillna(0) > 0)
        if "openInterest" in valid.columns:
            open_interest = valid["openInterest"].fillna(0) > 0
            liquidity = open_interest if liquidity is None else (liquidity | open_interest)
        if liquidity is not None:
            valid = valid.loc[liquidity]
        if valid.empty:
            continue
        nearest = valid.iloc[(valid["strike"] - spot).abs().argsort()[:1]]
        estimates.extend(float(value) for value in nearest["impliedVolatility"])
    if not estimates:
        raise RuntimeError(f"no valid ATM implied volatility for {symbol} {expiry}")
    return float(np.median(estimates)), f"{symbol.upper()} option chain", expiry, dte


def fetch_market_context(symbol: str, history_days: int = 120) -> MarketContext:
    """一次拉齐 spot + 历史波动率 + IV 代理。"""
    spot, change_pct, market_as_of = fetch_spot(symbol, include_as_of=True)
    history = fetch_history(symbol, days=history_days)
    closes = history["close"].dropna()
    hv30 = annualized_volatility(closes, window=30, require_full_window=True)
    hv60 = annualized_volatility(closes, window=60, require_full_window=True)
    option_expiry = None
    option_dte = None
    try:
        iv, iv_src, option_expiry, option_dte = fetch_atm_iv(symbol, spot)
        iv_scope = "same_symbol_option_chain"
        iv_comparable = True
    except Exception:
        iv, iv_src = fetch_iv_proxy(symbol)
        iv_scope = "broad_market_index" if iv is not None else None
        iv_comparable = False
    return MarketContext(
        symbol=symbol.upper(), spot=spot, change_24h_pct=change_pct,
        historical_vol_30d=hv30, historical_vol_60d=hv60,
        iv_proxy=iv, iv_proxy_source=iv_src,
        as_of=market_as_of,
        market_data_as_of=market_as_of,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        history_observations=len(closes),
        iv_proxy_scope=iv_scope,
        iv_proxy_comparable=iv_comparable,
        option_expiry=option_expiry,
        option_dte=option_dte,
    )


def synthetic_market_context(symbol: str = "FAKE", spot: float = 100.0,
                             hv30: float = 0.25, hv60: float = 0.28,
                             iv: float = 0.22) -> MarketContext:
    """合成市场画像，供测试 / 离线 demo 用。"""
    return MarketContext(
        symbol=symbol.upper(), spot=spot, change_24h_pct=0.5,
        historical_vol_30d=hv30, historical_vol_60d=hv60,
        iv_proxy=iv, iv_proxy_source="synthetic",
        source="synthetic",
        as_of="synthetic",
        market_data_as_of="synthetic",
        fetched_at="synthetic",
        history_observations=0,
        iv_proxy_scope="synthetic",
        iv_proxy_comparable=False,
    )
