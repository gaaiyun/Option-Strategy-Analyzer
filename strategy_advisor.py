"""LLM 驱动的期权策略建议器。

给一段市场观点（"我看 AAPL 未来一个月会大涨，但担心黑天鹅"），LLM 或基于
规则的启发式从 7 个内置策略里挑一个 + 给参数建议。

7 个策略对应市场观点表（学院派概览）：

| 策略 | 适用市场观点 |
|---|---|
| Long Call | 强烈看涨 |
| Long Put | 强烈看跌 |
| Bull Call Spread | 温和看涨 + 控成本 |
| Bear Put Spread | 温和看跌 + 控成本 |
| Straddle | 看大涨大跌，方向不明（高 IV 反对）|
| Strangle | 同 Straddle，更便宜但需要更大波动 |
| Iron Condor | 横盘 + 卖波动率（高 IV 时甜区） |
| Covered Call | 持有标的 + 想薅时间价值（中性偏多） |
| Protective Put | 持有标的 + 担心下跌（套保） |

LLM 缺 key 时退化为基于市场观点 + IV 状态的简单分支规则。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


LLMBackend = Literal["openai", "anthropic", "deepseek"]
MarketView = Literal[
    "strong_bullish", "moderate_bullish",
    "strong_bearish", "moderate_bearish",
    "neutral_range", "high_volatility", "hedging_protection",
]


@dataclass
class StrategyRecommendation:
    strategy_name: str
    rationale: str
    parameters: Dict
    market_view: str
    backend: str

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "rationale": self.rationale,
            "parameters": self.parameters,
            "market_view": self.market_view,
            "backend": self.backend,
        }


_STRATEGY_LIBRARY = {
    "Long Call": "Buy 1 ATM call. Max loss = premium paid. Use for strong bullish view.",
    "Long Put": "Buy 1 ATM put. Max loss = premium paid. Use for strong bearish view.",
    "Bull Call Spread": "Buy lower-strike call + sell higher-strike call (same expiry). "
                        "Caps upside but lowers cost. Use for moderate bullish.",
    "Bear Put Spread": "Buy higher-strike put + sell lower-strike put. Use for moderate bearish.",
    "Long Straddle": "Buy 1 ATM call + 1 ATM put. Profits on large move either direction. "
                     "Hurt by time decay; needs realized vol > implied vol.",
    "Long Strangle": "Buy OTM call + OTM put. Cheaper than straddle, needs bigger move.",
    "Iron Condor": "Sell OTM call spread + sell OTM put spread. Profits if price stays in "
                   "range. Best when IV is high (collect rich premium).",
    "Covered Call": "Long underlying + short OTM call. Income on flat-to-up moves; "
                    "caps upside.",
    "Protective Put": "Long underlying + long OTM put. Insurance against drawdown.",
}


def _heuristic(view: MarketView, hv_30: float, iv: Optional[float],
               has_underlying: bool = False) -> Dict:
    """规则建议器（LLM 不可用时使用）。

    主要根据：
    - 方向观点（强 / 温和 / 中性）
    - IV 与历史波动率的对比（IV > HV → 卖方有利；IV < HV → 买方有利）
    - 是否已持有标的（影响 covered call / protective put 的可用性）
    """
    iv_rich = (iv is not None and hv_30 > 0 and iv > hv_30 * 1.1)

    if view == "strong_bullish":
        return {"strategy_name": "Long Call",
                "parameters": {"strike": "ATM", "expiry_days": 30},
                "rationale": "强烈看涨，直接 long call 获取无限上行。"}
    if view == "strong_bearish":
        return {"strategy_name": "Long Put",
                "parameters": {"strike": "ATM", "expiry_days": 30},
                "rationale": "强烈看跌，long put 获取无限下行（受限于标的归零）。"}
    if view == "moderate_bullish":
        return {"strategy_name": "Bull Call Spread",
                "parameters": {"long_strike": "ATM", "short_strike": "OTM +5~10%",
                               "expiry_days": 45},
                "rationale": "温和看涨，价差控成本，让 theta 不那么伤。"}
    if view == "moderate_bearish":
        return {"strategy_name": "Bear Put Spread",
                "parameters": {"long_strike": "ATM", "short_strike": "OTM -5~10%",
                               "expiry_days": 45},
                "rationale": "温和看跌，put 价差控成本。"}
    if view == "high_volatility":
        if iv_rich:
            return {"strategy_name": "Iron Condor",
                    "parameters": {"call_short_strike": "OTM +5%",
                                   "call_long_strike": "OTM +10%",
                                   "put_short_strike": "OTM -5%",
                                   "put_long_strike": "OTM -10%",
                                   "expiry_days": 45},
                    "rationale": "IV 偏高且看横盘 → 卖波动率，铁鹰收时间价值。"}
        return {"strategy_name": "Long Straddle",
                "parameters": {"strike": "ATM", "expiry_days": 30},
                "rationale": "IV 不算高且预期大波动 → 买跨式赌方向不限。"}
    if view == "neutral_range":
        if iv_rich:
            return {"strategy_name": "Iron Condor",
                    "parameters": {"call_short_strike": "OTM +5%",
                                   "call_long_strike": "OTM +10%",
                                   "put_short_strike": "OTM -5%",
                                   "put_long_strike": "OTM -10%",
                                   "expiry_days": 45},
                    "rationale": "横盘 + IV 富裕 → 卖波动率铁鹰。"}
        if has_underlying:
            return {"strategy_name": "Covered Call",
                    "parameters": {"strike": "OTM +5%", "expiry_days": 30},
                    "rationale": "持有标的 + 横盘 → 备兑看涨薅时间价值。"}
        return {"strategy_name": "Iron Condor",
                "parameters": {"call_short_strike": "OTM +5%",
                               "call_long_strike": "OTM +10%",
                               "put_short_strike": "OTM -5%",
                               "put_long_strike": "OTM -10%",
                               "expiry_days": 30},
                "rationale": "横盘预期，铁鹰仍是宽适用区间。"}
    # hedging_protection
    if has_underlying:
        return {"strategy_name": "Protective Put",
                "parameters": {"strike": "OTM -5%", "expiry_days": 90},
                "rationale": "持仓套保，买 OTM put 锁底。"}
    return {"strategy_name": "Long Put",
            "parameters": {"strike": "ATM", "expiry_days": 30},
            "rationale": "无持仓但要 hedge → long put 直接 short delta。"}


# --- LLM ---------------------------------------------------------------------

class _LLMClient:
    def __init__(self, backend: LLMBackend = "deepseek",
                 model: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.backend = backend
        self.api_key = api_key or {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        }.get(backend)
        self.model = model or {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "deepseek": "deepseek-chat",
        }.get(backend, "gpt-4o-mini")
        self.base_url = {"deepseek": "https://api.deepseek.com/v1"}.get(backend)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str) -> str:
        if not self.is_available():
            raise RuntimeError(f"{self.backend} 没配 API key")
        if self.backend == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            resp = client.messages.create(
                model=self.model, max_tokens=1024, temperature=0.2,
                system=system, messages=[{"role": "user", "content": user}])
            return resp.content[0].text if resp.content else ""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model, temperature=0.2,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _advise_llm(view_text: str, hv_30: float, iv: Optional[float],
                has_underlying: bool, client: _LLMClient) -> Dict:
    system = (
        "你是期权策略顾问。根据用户的市场观点 + 当前波动率指标，从以下 9 个策略里"
        "**选一个**最适合的，并给出初始参数和理由。\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in _STRATEGY_LIBRARY.items())
        + "\n\n只输出 JSON，字段：strategy_name (上面 9 个之一), "
          "parameters (dict: strike + expiry_days 等), rationale (≤120 字中文)。"
    )
    iv_str = f"{iv:.1%}" if iv is not None else "未知"
    user = (
        f"用户市场观点：{view_text}\n"
        f"持有标的：{'是' if has_underlying else '否'}\n"
        f"历史波动率 (HV30)：{hv_30:.1%}\n"
        f"隐含波动率代理 (IV)：{iv_str}\n\n"
        f"按 system 要求输出 JSON。"
    )
    raw = _strip_fences(client.chat(system, user))
    data = json.loads(raw)
    return {
        "strategy_name": str(data.get("strategy_name", "Long Call")),
        "parameters": dict(data.get("parameters", {})),
        "rationale": str(data.get("rationale", "")),
    }


# --- 主入口 -----------------------------------------------------------------

def advise(
    market_view: MarketView,
    hv_30: float,
    iv: Optional[float] = None,
    has_underlying: bool = False,
    view_text: Optional[str] = None,
    backend: Optional[str] = None,
    llm_client: Optional[_LLMClient] = None,
) -> StrategyRecommendation:
    """主入口：给观点 + 市场状态 → 给策略建议。

    Parameters
    ----------
    market_view : 7 个枚举之一
    hv_30 / iv : 历史波动率 / 隐含波动率代理（年化小数）
    has_underlying : 是否持有标的
    view_text : 用户自由文本观点，LLM 路径用得到
    backend : LLM backend；None 时走规则启发式
    """
    client = llm_client
    if client is None and backend:
        client = _LLMClient(backend=backend)

    if client and client.is_available() and view_text:
        try:
            data = _advise_llm(view_text, hv_30, iv, has_underlying, client)
            return StrategyRecommendation(
                strategy_name=data["strategy_name"],
                parameters=data["parameters"],
                rationale=data["rationale"],
                market_view=market_view,
                backend=f"llm:{client.backend}",
            )
        except Exception:
            pass

    rec = _heuristic(market_view, hv_30, iv, has_underlying)
    return StrategyRecommendation(
        strategy_name=rec["strategy_name"],
        parameters=rec["parameters"],
        rationale=rec["rationale"],
        market_view=market_view,
        backend="heuristic",
    )


# 可用市场观点列表（CLI choices）
MARKET_VIEWS = [
    "strong_bullish", "moderate_bullish",
    "strong_bearish", "moderate_bearish",
    "neutral_range", "high_volatility", "hedging_protection",
]
