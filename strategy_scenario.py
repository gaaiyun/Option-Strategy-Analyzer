"""Connect deterministic strategy recommendations to reproducible model scenarios."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from strategy_advisor import _STRATEGY_LIBRARY, advise
from strategy_builder import OptionLeg, OptionStrategy, StrategyBuilder, UnderlyingLeg


@dataclass
class StrategyScenario:
    strategy_name: str
    rationale: str
    market_view: str
    initial_cost: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    greeks: Dict[str, float]
    scenario_pnl: List[Dict[str, float]]
    recommended_parameters: Dict[str, object]
    actual_legs: List[Dict[str, object]]
    assumptions: Dict[str, object]

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "rationale": self.rationale,
            "market_view": self.market_view,
            "initial_cost": float(self.initial_cost),
            "max_profit": float(self.max_profit) if math.isfinite(self.max_profit) else None,
            "max_profit_unbounded": bool(self.max_profit == math.inf),
            "max_loss": float(self.max_loss) if math.isfinite(self.max_loss) else None,
            "max_loss_unbounded": bool(self.max_loss == -math.inf),
            "breakeven_points": [float(value) for value in self.breakeven_points],
            "greeks": {key: float(value) for key, value in self.greeks.items()},
            "scenario_pnl": [
                {key: float(value) for key, value in row.items()}
                for row in self.scenario_pnl
            ],
            "recommended_parameters": self.recommended_parameters,
            "actual_legs": self.actual_legs,
            "assumptions": self.assumptions,
        }


def _resolve_strike(value: object, spot: float) -> float:
    """Resolve the advisor's compact strike notation to a model strike."""
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"unsupported strike descriptor: {value!r}")

    descriptor = value.strip().upper().replace(" ", "")
    if descriptor == "ATM":
        return float(spot)
    match = re.fullmatch(r"OTM([+-])(\d+(?:\.\d+)?)(?:~(\d+(?:\.\d+)?))?%", descriptor)
    if not match:
        raise ValueError(f"unsupported strike descriptor: {value!r}")
    low = float(match.group(2))
    high = float(match.group(3) or low)
    pct = (low + high) / 2 / 100
    direction = 1 if match.group(1) == "+" else -1
    return float(spot * (1 + direction * pct))


def _serialize_legs(strategy: OptionStrategy) -> List[Dict[str, object]]:
    legs: List[Dict[str, object]] = []
    for leg in strategy.legs:
        if isinstance(leg, UnderlyingLeg):
            legs.append({
                "type": "underlying",
                "quantity": float(leg.quantity),
                "entry_price": float(leg.entry_price),
            })
        else:
            legs.append({
                "type": "option",
                "option_type": leg.option_type,
                "strike": float(leg.strike),
                "quantity": int(leg.quantity),
                "side": "long" if leg.long else "short",
            })
    return legs


def materialize_strategy(name: str, S: float, T: float, r: float,
                         sigma: float, q: float = 0.0,
                         parameters: Optional[Dict[str, object]] = None) -> OptionStrategy:
    """Turn one registered recommendation into a concrete model portfolio."""
    if name not in _STRATEGY_LIBRARY:
        raise ValueError(f"strategy is not registered: {name!r}")
    params = dict(parameters or {})
    if "expiry_days" in params:
        expiry_days = float(params["expiry_days"])
        if not math.isfinite(expiry_days) or expiry_days <= 0:
            raise ValueError("expiry_days must be a positive finite number")
        if abs(T * 365 - expiry_days) > 7:
            raise ValueError(
                f"model expiry ({T * 365:.1f} days) does not match "
                f"recommended expiry ({expiry_days:.1f} days)"
            )

    strike = lambda key, default: _resolve_strike(params.get(key, default), S)
    builder = StrategyBuilder(S=S, T=T, r=r, sigma=sigma, q=q)
    constructors = {
        "Long Call": lambda: OptionStrategy(
            "Long Call", [OptionLeg("call", strike("strike", "ATM"))],
            S, T, r, sigma, q
        ),
        "Long Put": lambda: OptionStrategy(
            "Long Put", [OptionLeg("put", strike("strike", "ATM"))],
            S, T, r, sigma, q
        ),
        "Bull Call Spread": lambda: builder.bull_call_spread(
            strike("long_strike", "ATM"), strike("short_strike", "OTM +5~10%")
        ),
        "Bear Put Spread": lambda: builder.bear_put_spread(
            strike("long_strike", "ATM"), strike("short_strike", "OTM -5~10%")
        ),
        "Long Straddle": lambda: builder.straddle(strike("strike", "ATM")),
        "Long Strangle": lambda: builder.strangle(
            strike("call_strike", "OTM +5%"), strike("put_strike", "OTM -5%")
        ),
        "Iron Condor": lambda: builder.iron_condor(
            strike("put_long_strike", "OTM -10%"),
            strike("put_short_strike", "OTM -5%"),
            strike("call_short_strike", "OTM +5%"),
            strike("call_long_strike", "OTM +10%"),
        ),
        "Covered Call": lambda: builder.covered_call(strike("strike", "OTM +5%")),
        "Protective Put": lambda: builder.protective_put(strike("strike", "OTM -5%")),
    }
    return constructors[name]()


def analyze_scenario(
    market_view: str,
    S: float,
    T: float,
    r: float,
    sigma: float,
    hv_30: float,
    iv: Optional[float] = None,
    q: float = 0.0,
    has_underlying: bool = False,
    iv_is_comparable: bool = True,
) -> StrategyScenario:
    """Recommend, materialize and evaluate a strategy under five expiry moves.

    Prices and Greeks use BSM with the supplied volatility. The scenario grid is
    a deterministic teaching/research aid, not a market quote or probability model.
    """
    recommendation = advise(
        market_view=market_view,
        hv_30=hv_30,
        iv=iv,
        has_underlying=has_underlying,
        iv_is_comparable=iv_is_comparable,
    )
    strategy = materialize_strategy(
        recommendation.strategy_name, S=S, T=T, r=r, sigma=sigma, q=q,
        parameters=recommendation.parameters,
    )
    moves = (-0.20, -0.10, 0.0, 0.10, 0.20)
    scenario_pnl = [
        {
            "move_pct": move,
            "expiry_price": S * (1 + move),
            "profit": strategy.profit_at_expiration(S * (1 + move)),
        }
        for move in moves
    ]
    return StrategyScenario(
        strategy_name=recommendation.strategy_name,
        rationale=recommendation.rationale,
        market_view=market_view,
        initial_cost=strategy.initial_cost(),
        max_profit=strategy.max_profit(),
        max_loss=strategy.max_loss(),
        breakeven_points=strategy.breakeven_points(S_range=(0.0, S * 3)),
        greeks=strategy.calculate_strategy_greeks(),
        scenario_pnl=scenario_pnl,
        recommended_parameters=dict(recommendation.parameters),
        actual_legs=_serialize_legs(strategy),
        assumptions={
            "spot": S,
            "time_to_expiry_years": T,
            "risk_free_rate": r,
            "pricing_volatility": sigma,
            "dividend_yield": q,
            "historical_volatility_30d": hv_30,
            "implied_volatility": iv,
            "iv_is_comparable": iv_is_comparable,
            "transaction_costs": False,
            "market_quotes": False,
        },
    )
