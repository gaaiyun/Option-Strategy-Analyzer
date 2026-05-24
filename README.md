# Option-Strategy-Analyzer

Black-Scholes 期权定价 + 希腊值 + 7 种经典策略组合 + 波动率分析。v1 提供完整的
Streamlit 仪表板和 85 个单元测试。v2 在不动核心定价/希腊值/策略库的前提下补
三件实用的东西：

1. **CLI 入口** — v1 只能跑 Streamlit，没法脚本化。v2 加 `__main__.py` 子命令
   覆盖定价、希腊、数据抓取、策略建议。
2. **市场数据抓取** — v1 让用户手填 `S` / `r` / `sigma`，对 BSM 演示够，但实盘
   经常**不知道当前 IV**。v2 加 yfinance 自动抓现价 + 历史波动率 + IV 代理（美股
   接 VIX）。
3. **LLM/规则策略建议器** — 给一段市场观点（"未来一个月强烈看涨"），LLM 或规则
   启发式从 9 个策略里挑一个 + 给参数。LLM 缺 key 时退化到规则。

## v2 新增模块

| 文件 | 干什么 |
|---|---|
| `data_fetch.py` | `fetch_market_context(symbol)` 一次拉齐 spot + HV30/60 + IV 代理（VIX）+ `synthetic_market_context` 离线 demo |
| `strategy_advisor.py` | `advise(market_view, hv_30, iv, ...)` 返回 `StrategyRecommendation`：9 个策略 + 7 种市场观点，LLM 或规则双路径 |
| `__main__.py` | CLI：`price` / `greeks` / `fetch` / `advise` / `list-strategies` / `list-views` |
| `tests/test_data_fetch.py` | 17 测试：mock yfinance |
| `tests/test_strategy_advisor.py` | 26 测试：规则覆盖 7 种观点 + LLM mock |

总 128 测试通过（85 v1 + 43 v2），3 秒内跑完。

## v1 仍保留

| 模块 | 干什么 |
|---|---|
| `option_pricer.py` | BSM 欧式 / 美式近似定价 |
| `greeks_calculator.py` | Delta / Gamma / Theta / Vega / Rho |
| `strategy_builder.py` | 7 种策略（跨式 / 宽跨式 / 牛熊价差 / 铁鹰 / 备兑 / 保护性看跌） |
| `volatility_analyzer.py` | 历史 / 隐含波动率、GARCH 预测、期限结构、微笑 |
| `dashboard.py` | Streamlit 交互式仪表板 |

## 安装

```bash
pip install -r requirements.txt
# 可选：v2 数据抓取
pip install yfinance
# 可选：v2 LLM 策略建议
pip install openai      # openai / deepseek
pip install anthropic
```

## 快速开始

### v2 CLI 入口

```bash
# 定价单合约
python __main__.py price --S 100 --K 105 --T 0.25 --r 0.05 --sigma 0.25 --type call
# BSM 价格 : 3.4399

# 算希腊值
python __main__.py greeks --S 100 --K 95 --T 0.25 --r 0.05 --sigma 0.25 --type put
# delta : -0.283374
# gamma : +0.027086
# vega  : +0.169287
# ...

# 抓 AAPL 当前价 + HV30/60 + VIX（IV 代理）
python __main__.py fetch --symbol AAPL --days 90

# 离线 demo
python __main__.py fetch --synthetic --symbol AAPL

# 看可选市场观点（7 个）
python __main__.py list-views

# LLM/规则推荐策略
python __main__.py advise --view neutral_range --hv-30 0.20 --iv 0.30
# 推荐策略 : Iron Condor（IV > HV 富裕 → 卖波动率）

python __main__.py advise --view hedging_protection --hv-30 0.25 --has-underlying
# 推荐策略 : Protective Put（套保）

# 用 LLM（需要 DEEPSEEK_API_KEY）
python __main__.py advise --view neutral_range --hv-30 0.20 --iv 0.30 \
    --view-text "AAPL 接下来 1 个月应该在 170-190 区间震荡" \
    --use-llm --backend deepseek
```

### v1 Streamlit 仪表板（仍能用）

```bash
streamlit run dashboard.py
```

### 库调用

```python
from option_pricer import OptionPricer
from greeks_calculator import GreeksCalculator
from strategy_builder import OptionStrategy, OptionLeg
from data_fetch import fetch_market_context
from strategy_advisor import advise

# v1：定价
pricer = OptionPricer(S=100, K=105, T=0.25, r=0.05, sigma=0.25)
price = pricer.european_call()        # 3.44

# v1：希腊值
g = GreeksCalculator(S=100, K=105, T=0.25, r=0.05, sigma=0.25)
greeks = g.calculate_all(option_type="call")

# v2：抓市场数据
ctx = fetch_market_context("AAPL", history_days=90)
# ctx.spot / ctx.historical_vol_30d / ctx.iv_proxy

# v2：让规则或 LLM 建议策略
rec = advise(
    market_view="moderate_bullish",
    hv_30=ctx.historical_vol_30d,
    iv=ctx.iv_proxy,
    has_underlying=False,
    view_text="未来一个月温和看涨，担心黑天鹅",
    backend="deepseek",   # None 时用规则
)
print(rec.strategy_name, rec.parameters, rec.rationale)
```

## 市场观点 → 策略对照表（v2 规则启发式）

| 观点 | IV 富裕（IV > HV30 × 1.1） | IV 不富裕 |
|---|---|---|
| `strong_bullish` | Long Call | Long Call |
| `moderate_bullish` | Bull Call Spread | Bull Call Spread |
| `strong_bearish` | Long Put | Long Put |
| `moderate_bearish` | Bear Put Spread | Bear Put Spread |
| `high_volatility` | Iron Condor（卖 vol） | Long Straddle |
| `neutral_range` | Iron Condor | Covered Call (有标的) / Iron Condor |
| `hedging_protection` | Protective Put（有标的） | Long Put |

LLM 路径覆盖 9 个完整策略 + 自由文本观点（"我看这个月强烈看涨但担心 earnings 黑天鹅"）。

## 测试

```bash
pytest tests/
```

128 个测试，3 秒内跑完。yfinance / LLM 全部 mock，CI 友好。

## 设计取舍

- **CLI 没接策略组合构建**：`strategy_builder.OptionStrategy` 是组合多个 leg 的高
  级 API，CLI 暴露它会让接口爆炸（多 leg 怎么传？）。`advise` 给"应该用哪个策略
  + 初始参数"建议，组合细节让用户在 Streamlit 或 Python 里搭。
- **IV 代理只支持美股 VIX**：BTC 的 BVOL 在 yfinance 不稳定，没接。其他 ticker 的
  IV 代理表 `_IV_PROXIES` 可手动扩展。
- **LLM 必须传 view_text**：自由文本是 LLM 路径的核心 —— 缺了它和规则路径没区别，
  直接走规则更省 token。

## 项目结构

```
Option-Strategy-Analyzer/
├── __main__.py                # v2 CLI 统一入口
├── README.md
├── dashboard.py               # v1 Streamlit 仪表板
├── option_pricer.py           # v1 BSM 定价
├── greeks_calculator.py       # v1 希腊值
├── strategy_builder.py        # v1 策略组合
├── volatility_analyzer.py     # v1 波动率
├── data_fetch.py              # v2 yfinance 数据抓取
├── strategy_advisor.py        # v2 LLM/规则策略建议
├── tests/                     # 128 测试
│   ├── test_option_pricer.py
│   ├── test_greeks.py
│   ├── test_strategy.py
│   ├── test_volatility.py
│   ├── test_data_fetch.py     # v2 新增
│   └── test_strategy_advisor.py  # v2 新增
├── pytest.ini
└── requirements.txt
```

## 已知限制

- 美式期权定价是简单近似（取 max(欧式价, 内在价值)），不是 Barone-Adesi-Whaley
  完整解析解。学院派演示够，实盘需要换 BAW 或二叉树。
- IV 代理用 VIX 是粗代理 —— VIX 是 SPX 30 日 IV，对 individual stock 不一定准。
  实盘要用 option chain 反推 ATM IV。
- `advise` 输出的参数（"ATM" / "OTM +5%"）是字符串描述，需要用户自己换成具体行
  权价喂给 `strategy_builder`。

## 许可

MIT
