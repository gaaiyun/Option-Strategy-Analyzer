"""Option-Strategy-Analyzer CLI（v2）。

子命令：
    price       BSM 定价单合约
    greeks      算单合约希腊值
    fetch       yfinance 抓现价 + 历史波动率 + IV 代理
    advise      LLM 或规则建议策略
    list-strategies / list-views
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MARKET_VIEW_CHOICES = (
    "strong_bullish", "moderate_bullish", "strong_bearish",
    "moderate_bearish", "neutral_range", "high_volatility",
    "hedging_protection",
)


def cmd_price(args) -> int:
    from option_pricer import OptionPricer
    pricer = OptionPricer(S=args.S, K=args.K, T=args.T, r=args.r,
                          sigma=args.sigma, q=args.q)
    if args.american:
        price = (pricer.american_call_approx() if args.type == "call"
                 else pricer.american_put_approx())
    else:
        price = (pricer.european_call() if args.type == "call"
                 else pricer.european_put())

    print(f"标的价 S      : {args.S}")
    print(f"行权价 K      : {args.K}")
    print(f"到期(年) T    : {args.T}")
    print(f"无风险利率 r  : {args.r}")
    print(f"波动率 σ      : {args.sigma}")
    print(f"股息率 q      : {args.q}")
    print(f"期权类型      : {args.type}{'（美式近似）' if args.american else ''}")
    print(f"BSM 价格      : {price:.4f}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps({"price": price, "args": vars(args)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_greeks(args) -> int:
    from greeks_calculator import GreeksCalculator
    g = GreeksCalculator(S=args.S, K=args.K, T=args.T, r=args.r,
                         sigma=args.sigma, q=args.q)
    result = g.calculate_all(option_type=args.type)
    for k, v in result.items():
        print(f"{k:6}: {v:+.6f}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_fetch(args) -> int:
    from data_fetch import fetch_market_context, synthetic_market_context
    if args.synthetic:
        ctx = synthetic_market_context(symbol=args.symbol or "FAKE")
    else:
        if not args.symbol:
            sys.stderr.write("[error] 需要 --symbol 或 --synthetic\n")
            return 1
        try:
            ctx = fetch_market_context(args.symbol, history_days=args.days)
        except ImportError as e:
            sys.stderr.write(f"[error] {e}\n")
            return 2
        except Exception as e:
            sys.stderr.write(f"[error] {e}\n")
            return 3

    d = ctx.to_dict()
    print(json.dumps(d, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(d, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    return 0


def cmd_advise(args) -> int:
    from strategy_advisor import MARKET_VIEWS, advise
    if args.view not in MARKET_VIEWS:
        sys.stderr.write(f"[error] view 必须是 {MARKET_VIEWS}\n")
        return 1

    rec = advise(
        market_view=args.view,
        hv_30=args.hv_30,
        iv=args.iv,
        has_underlying=args.has_underlying,
        view_text=args.view_text,
        backend=args.backend if args.use_llm else None,
    )

    if rec.fallback_reason:
        sys.stderr.write(f"[warning] LLM fallback: {rec.fallback_reason}\n")
    if args.require_llm and not rec.backend.startswith("llm:"):
        sys.stderr.write("[error] --require-llm was set but no LLM result was produced\n")
        return 4

    print(f"市场观点      : {rec.market_view}")
    print(f"推荐策略      : {rec.strategy_name}")
    print(f"参数          : {rec.parameters}")
    print(f"理由          : {rec.rationale}")
    print(f"后端          : {rec.backend}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(rec.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_scenario(args) -> int:
    from strategy_scenario import analyze_scenario

    report = analyze_scenario(
        market_view=args.view,
        S=args.S,
        T=args.T,
        r=args.r,
        sigma=args.sigma,
        q=args.q,
        hv_30=args.hv_30,
        iv=args.iv,
        has_underlying=args.has_underlying,
        iv_is_comparable=not args.iv_is_broad_market_proxy,
    ).to_dict()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    return 0


def cmd_list_strategies(args) -> int:
    from strategy_advisor import _STRATEGY_LIBRARY
    for name, desc in _STRATEGY_LIBRARY.items():
        print(f"- {name}")
        print(f"    {desc}")
    return 0


def cmd_list_views(args) -> int:
    from strategy_advisor import MARKET_VIEWS
    for v in MARKET_VIEWS:
        print(f"  {v}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="option-analyzer",
        description="期权策略分析 CLI：BSM 定价 / 希腊值 / 数据 / 建议"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common_pricing = lambda sp: (
        sp.add_argument("--S", type=float, required=True, help="标的价"),
        sp.add_argument("--K", type=float, required=True, help="行权价"),
        sp.add_argument("--T", type=float, required=True, help="到期(年)"),
        sp.add_argument("--r", type=float, default=0.05, help="无风险利率"),
        sp.add_argument("--sigma", type=float, required=True, help="波动率"),
        sp.add_argument("--q", type=float, default=0.0, help="股息率"),
        sp.add_argument("--type", choices=["call", "put"], default="call"),
    )

    sp = sub.add_parser("price", help="BSM 定价单合约")
    common_pricing(sp)
    sp.add_argument("--american", action="store_true", help="美式近似")
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_price)

    sp = sub.add_parser("greeks", help="算单合约希腊值")
    common_pricing(sp)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_greeks)

    sp = sub.add_parser("fetch", help="yfinance 抓现价 + 波动率")
    sp.add_argument("--symbol")
    sp.add_argument("--days", type=int, default=120)
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("advise", help="LLM 或规则建议策略")
    sp.add_argument("--view", required=True, help="见 list-views")
    sp.add_argument("--hv-30", type=float, default=0.25, help="HV30 年化")
    sp.add_argument("--iv", type=float, help="IV proxy 年化（小数）")
    sp.add_argument("--has-underlying", action="store_true",
                    help="是否已持有标的（影响 covered call / protective put）")
    sp.add_argument("--view-text", help="自由文本市场观点（LLM 路径用）")
    sp.add_argument("--use-llm", action="store_true")
    sp.add_argument("--require-llm", action="store_true",
                    help="LLM 不可用或输出无效时返回非零，不退化为规则")
    sp.add_argument("--backend", default="deepseek",
                    choices=["openai", "anthropic", "deepseek"])
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_advise)

    sp = sub.add_parser("scenario", help="建议策略并输出可复算的 BSM 情景报告")
    sp.add_argument("--view", required=True, choices=MARKET_VIEW_CHOICES)
    sp.add_argument("--S", type=float, required=True, help="标的价")
    sp.add_argument("--T", type=float, required=True, help="到期年数")
    sp.add_argument("--r", type=float, default=0.05, help="无风险利率")
    sp.add_argument("--sigma", type=float, required=True, help="定价波动率")
    sp.add_argument("--q", type=float, default=0.0, help="股息率")
    sp.add_argument("--hv-30", type=float, required=True, help="30日历史波动率")
    sp.add_argument("--iv", type=float, help="同标的、同期限附近的隐含波动率")
    sp.add_argument("--iv-is-broad-market-proxy", action="store_true",
                    help="IV 是 VIX 等宽基代理，不用于 IV/HV 相对价值判断")
    sp.add_argument("--has-underlying", action="store_true")
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_scenario)

    sp = sub.add_parser("list-strategies", help="列 9 个内置策略")
    sp.set_defaults(func=cmd_list_strategies)

    sp = sub.add_parser("list-views", help="列可选市场观点")
    sp.set_defaults(func=cmd_list_views)

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError) as exc:
        sys.stderr.write(f"[error] {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
