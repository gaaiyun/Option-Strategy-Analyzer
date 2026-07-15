from __future__ import annotations

import math

import pytest

from strategy_builder import UnderlyingLeg
from strategy_scenario import analyze_scenario, materialize_strategy


@pytest.mark.parametrize(
    "name",
    [
        "Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread",
        "Long Straddle", "Long Strangle", "Iron Condor", "Covered Call",
        "Protective Put",
    ],
)
def test_materializes_every_registered_strategy(name):
    strategy = materialize_strategy(
        name, S=100, T=30 / 365, r=0.03, sigma=0.25
    )
    assert strategy.legs
    assert math.isfinite(strategy.initial_cost())


def test_covered_call_and_protective_put_include_underlying():
    for name in ("Covered Call", "Protective Put"):
        strategy = materialize_strategy(name, 100, 30 / 365, 0.03, 0.25)
        assert any(isinstance(leg, UnderlyingLeg) for leg in strategy.legs)


def test_scenario_report_exposes_bounds_greeks_and_pnl_grid():
    report = analyze_scenario(
        market_view="neutral_range",
        S=100,
        T=45 / 365,
        r=0.03,
        sigma=0.25,
        hv_30=0.20,
        iv=0.30,
    )
    data = report.to_dict()
    assert data["strategy_name"] == "Iron Condor"
    assert data["max_profit_unbounded"] is False
    assert data["max_loss_unbounded"] is False
    assert set(data["greeks"]) == {"delta", "gamma", "theta", "vega", "rho"}
    assert len(data["scenario_pnl"]) == 5
    assert data["recommended_parameters"]["put_short_strike"] == "OTM -5%"
    assert len(data["actual_legs"]) == 4
    assert data["actual_legs"][0]["strike"] == 90.0
    assert data["actual_legs"][1]["strike"] == 95.0
    center = next(row for row in data["scenario_pnl"] if row["move_pct"] == 0)
    assert center["profit"] > 0


def test_scenario_report_marks_long_call_upside_unbounded():
    report = analyze_scenario(
        market_view="strong_bullish",
        S=100,
        T=30 / 365,
        r=0.03,
        sigma=0.25,
        hv_30=0.25,
        iv=0.22,
    )
    data = report.to_dict()
    assert data["strategy_name"] == "Long Call"
    assert data["max_profit"] is None
    assert data["max_profit_unbounded"] is True


def test_unknown_strategy_cannot_be_materialized():
    with pytest.raises(ValueError, match="registered"):
        materialize_strategy("Wire Funds", 100, 30 / 365, 0.03, 0.25)


def test_scenario_rejects_large_expiry_mismatch():
    with pytest.raises(ValueError, match="expiry"):
        analyze_scenario(
            market_view="strong_bullish", S=100, T=180 / 365,
            r=0.03, sigma=0.25, hv_30=0.25, iv=0.22,
        )
