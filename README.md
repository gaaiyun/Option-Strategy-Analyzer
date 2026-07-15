# Option Strategy Analyzer

一个用于期权教学与研究的确定性工具：Black-Scholes-Merton 欧式期权定价、
Greeks、组合到期收益、理论盈亏边界、波动率分析和离线情景报告。Streamlit
仪表板保留，CLI 可以在脚本和 CI 中使用。

本项目不连接券商、不提交订单，也不提供投资建议。模型价格不是市场报价，
情景盈亏不是收益预测。

## 当前能力

| 模块 | 状态 | 已验证边界 |
| --- | --- | --- |
| 欧式期权定价 | 可用 | BSM call/put、股息率、到期边界、IV 反解 |
| 美式期权 | 教学近似 | 只取欧式价与内在价值较大者，不是完整 BAW 或二叉树 |
| Greeks | 可用 | Delta、Gamma、Theta、Vega、Rho |
| 组合策略 | 可用 | 9 种注册策略；Covered Call/Protective Put 包含标的腿 |
| 理论盈亏边界 | 可用 | 按分段线性到期收益识别有限值与 `unbounded`，不再用有限网格冒充理论极值 |
| 情景报告 | 可用 | 建议、组合、成本、Greeks、盈亏平衡点和 ±20% 五档到期 P&L |
| 市场数据 | 部分可用 | yfinance 现价、历史波动率、同标的期权链近月 ATM IV；含来源与时点 |
| LLM 观点解析 | 实验性 | OpenAI、Anthropic、DeepSeek；输出必须通过策略注册表，失败显式退回规则 |
| 自动交易 | 不支持 | 没有账户、订单、仓位管理或实时风险控制 |

## 安装

```bash
# 定价、Greeks、组合和离线情景
python -m pip install -e .

# 市场数据
python -m pip install -e ".[market]"

# Dashboard、市场数据和可选 LLM
python -m pip install -e ".[full]"

# 开发与测试
python -m pip install -e ".[dev,market]"
```

安装后可使用 `option-analyzer`。源码目录中的 `python __main__.py` 继续兼容。

## 快速开始

```bash
# 欧式 call 定价
option-analyzer price --S 100 --K 105 --T 0.25 --sigma 0.25

# put Greeks
option-analyzer greeks --S 100 --K 95 --T 0.25 --sigma 0.25 --type put

# 离线确定性情景：横盘、同标的 IV 高于 HV
option-analyzer scenario \
  --view neutral_range --S 100 --T 0.12 --sigma 0.25 \
  --hv-30 0.20 --iv 0.30

# 真实市场画像；输出含 as_of、样本数、IV 来源与是否可比较
option-analyzer fetch --symbol AAPL --days 120

# Streamlit 仪表板
streamlit run dashboard.py
```

`scenario` 的 `sigma` 是 BSM 定价假设；`hv-30` 和 `iv` 用于规则型策略选择。
推荐行权价会解析成实际组合腿并随报告披露；模型期限与推荐期限相差超过 7 天时会拒绝计算。
输出明确记录这些假设，并将理论无限利润写成 `null + unbounded=true`，保证 JSON
严格可解析。

## 策略语义

| 策略 | 组合 |
| --- | --- |
| Long Call / Long Put | 单腿买方 |
| Bull Call Spread | 买低执行价 call，卖高执行价 call |
| Bear Put Spread | 买高执行价 put，卖低执行价 put |
| Long Straddle / Strangle | 同执行价或不同执行价的 call + put 买方 |
| Iron Condor | 买低 put、卖高 put、卖低 call、买高 call |
| Covered Call | 持有标的 + 卖 call |
| Protective Put | 持有标的 + 买 put |

组合的 `initial_cost()` 包含所有期权权利金；涉及标的的策略也包含标的初始成本。
`max_profit()` 和 `max_loss()` 默认返回理论边界；只有显式传入价格范围时才做范围内
数值扫描。

## 市场数据边界

`fetch` 依次尝试：

1. 从同一标的、接近 30 天的期权链取最接近平值的 call/put IV 中位数；
2. 若链不可用，对少量美股返回 VIX 作为 `broad_market_index` 背景。

只有第一类被标记为 `iv_proxy_comparable=true`。VIX 是 SPX 30 日隐含波动率，
不能和 AAPL、TSLA 等个股 HV 直接比较后得出“个股 IV 富裕”。若把宽基代理传给
`scenario`，必须加 `--iv-is-broad-market-proxy`，规则不会据此触发卖波动率判断。

`change_24h_pct` 为“最新日收盘相对前一交易日收盘”的变化，不是严格滚动 24 小时。
HV30/HV60 需要完整的 30/60 个收益观察，样本不足会显式失败，不会用两三个点仍标成
HV30。

## LLM 边界

```bash
option-analyzer advise \
  --view neutral_range --hv-30 0.20 --iv 0.30 \
  --view-text "未来一个月可能在区间内震荡" \
  --use-llm --backend deepseek
```

- 每个 provider 只读取自己的环境变量；DeepSeek 不会复用 OpenAI key。
- LLM 只能从固定 9 个策略中选择，未知名称会被拒绝。
- 缺 key、网络失败或 JSON 无效时，stderr 和输出中的 `fallback_reason` 会说明原因。
- 加 `--require-llm` 可在 LLM 没有真正执行时返回非零，而不是接受规则结果。
- LLM 不负责定价、理论边界或情景计算。

## 测试

```bash
python -m pytest
python -m build
```

截至 2026-07-16，本地结果为 163 passed、1 skipped。CI 在 Python 3.11 和 3.13
运行测试、构建 wheel，并在干净虚拟环境安装 wheel 后执行 CLI smoke。

测试不证明 yfinance 有 SLA、市场报价可成交、模型适合真实交易，也不覆盖真实 LLM
调用、券商接口和 Streamlit 全量视觉交互。

## 项目结构

```text
option_pricer.py          BSM 定价与 IV 反解
greeks_calculator.py      Greeks
strategy_builder.py       期权腿、标的腿与组合收益
strategy_advisor.py       注册表约束的规则/LLM 建议
strategy_scenario.py      可复算的离线组合情景
data_fetch.py             yfinance 行情、HV 与 ATM option-chain IV
volatility_analyzer.py    历史波动率、GARCH 与曲面工具
option_analyzer_cli.py    可安装 CLI
dashboard.py              Streamlit 仪表板
tests/                    单元、CLI 与完整性测试
```

## License

[MIT](LICENSE)
