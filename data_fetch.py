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
from datetime import datetime, timedelta
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


def fetch_spot(symbol: str) -> tuple:
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
    return spot, change * 100


def annualized_volatility(closes: pd.Series, window: int = 30) -> float:
    """从收盘价算年化历史波动率。"""
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


def fetch_market_context(symbol: str, history_days: int = 90) -> MarketContext:
    """一次拉齐 spot + 历史波动率 + IV 代理。"""
    spot, change_pct = fetch_spot(symbol)
    history = fetch_history(symbol, days=history_days)
    closes = history["close"].dropna()
    hv30 = annualized_volatility(closes, window=30)
    hv60 = annualized_volatility(closes, window=60)
    iv, iv_src = fetch_iv_proxy(symbol)
    return MarketContext(
        symbol=symbol.upper(), spot=spot, change_24h_pct=change_pct,
        historical_vol_30d=hv30, historical_vol_60d=hv60,
        iv_proxy=iv, iv_proxy_source=iv_src,
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
    )
