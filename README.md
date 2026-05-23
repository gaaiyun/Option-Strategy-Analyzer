# 📊 期权策略分析平台

专业的期权定价与策略分析工具，基于 Black-Scholes-Merton 模型实现。

## ✨ 功能特点

### 1. BSM 期权定价
- ✅ 欧式/美式期权定价
- ✅ 看涨/看跌期权支持
- ✅ 股息率调整
- ✅ 内在价值与时间价值分解

### 2. 希腊值计算
- ✅ **Delta** - 标的价格敏感性
- ✅ **Gamma** - Delta 的变化率
- ✅ **Theta** - 时间衰减
- ✅ **Vega** - 波动率敏感性
- ✅ **Rho** - 利率敏感性

### 3. 策略构建器
支持多种经典期权策略：
- 🎯 跨式策略 (Straddle)
- 🎯 宽跨式策略 (Strangle)
- 🎯 牛市价差 (Bull Call Spread)
- 🎯 熊市价差 (Bear Put Spread)
- 🎯 铁鹰策略 (Iron Condor)
- 🎯 备兑看涨 (Covered Call)
- 🎯 保护性看跌 (Protective Put)

### 4. 盈亏分析
- 📈 交互式盈亏曲线
- 💹 盈亏平衡点计算
- 🔥 利润敏感性热力图
- 📊 最大利润/损失分析

### 5. 波动率分析
- 📊 历史波动率计算
- 🌊 隐含波动率曲面
- 📈 波动率期限结构
- 🎭 波动率微笑/偏斜

## 🚀 快速开始

### 安装依赖

```bash
cd option-strategy-analyzer
pip install -r requirements.txt
```

### 运行应用

```bash
streamlit run dashboard.py
```

应用将在浏览器中自动打开（默认 http://localhost:8501）

## 📁 项目结构

```
option-strategy-analyzer/
├── dashboard.py           # 主界面 (Streamlit)
├── option_pricer.py       # 期权定价模块
├── greeks_calculator.py   # 希腊值计算模块
├── strategy_builder.py    # 策略构建器模块
├── volatility_analyzer.py # 波动率分析模块
├── tests/
│   ├── test_option_pricer.py
│   ├── test_greeks.py
│   ├── test_strategy.py
│   └── test_volatility.py
├── requirements.txt       # 依赖列表
└── README.md             # 项目文档
```

## 📖 使用示例

### 期权定价

```python
from option_pricer import OptionPricer

# 初始化定价器
pricer = OptionPricer(
    S=100,      # 标的价格
    K=100,      # 行权价
    T=0.25,     # 到期时间 (3 个月)
    r=0.05,     # 无风险利率 (5%)
    sigma=0.2,  # 波动率 (20%)
    q=0.0       # 股息率
)

# 计算期权价格
call_price = pricer.european_call()
put_price = pricer.european_put()

print(f"看涨期权价格：${call_price:.4f}")
print(f"看跌期权价格：${put_price:.4f}")
```

### 希腊值计算

```python
from greeks_calculator import GreeksCalculator

calc = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.2)

# 计算所有希腊值
greeks = calc.calculate_all('call')

print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Rho: {greeks['rho']:.4f}")
```

### 策略构建

```python
from strategy_builder import StrategyBuilder

builder = StrategyBuilder(S=100, T=0.25, r=0.05, sigma=0.2)

# 构建跨式策略
straddle = builder.straddle()

print(f"策略名称：{straddle.name}")
print(f"初始成本：${straddle.initial_cost():.4f}")
print(f"盈亏平衡点：{straddle.breakeven_points()}")
print(f"最大利润：${straddle.max_profit():.4f}")
print(f"最大损失：${straddle.max_loss():.4f}")
```

### 波动率分析

```python
from volatility_analyzer import VolatilityAnalyzer
import pandas as pd

# 使用历史价格数据
prices = pd.Series([100, 101, 99, 102, 100, ...])

analyzer = VolatilityAnalyzer(prices=prices)

# 计算历史波动率
hist_vol = analyzer.historical_volatility(window=20)
print(f"历史波动率：{hist_vol*100:.2f}%")

# 计算已实现波动率
realized_vol = analyzer.realized_volatility(n_days=20)
print(f"已实现波动率：{realized_vol*100:.2f}%")
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行测试并生成覆盖率报告
pytest tests/ -v --cov=. --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html  # macOS/Linux
start htmlcov\index.html  # Windows
```

## 📊 界面预览

### 期权计算器
- 输入标的价格、行权价、波动率等参数
- 实时计算欧式/美式期权价格
- 显示内在价值和时间价值

### 希腊值分析
- 计算并可视化所有希腊值
- 敏感性分析图表
- 交互式参数调整

### 策略构建器
- 选择预设策略模板
- 自定义行权价和参数
- 实时显示策略希腊值和盈亏图

### 波动率分析
- 历史波动率时间序列
- 隐含波动率曲面可视化
- 波动率统计指标

## 🎯 核心公式

### Black-Scholes-Merton 公式

**看涨期权:**
$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

**看跌期权:**
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

其中:
$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

### 希腊值公式

- **Delta:** $\frac{\partial V}{\partial S}$
- **Gamma:** $\frac{\partial^2 V}{\partial S^2}$
- **Theta:** $\frac{\partial V}{\partial t}$
- **Vega:** $\frac{\partial V}{\partial \sigma}$
- **Rho:** $\frac{\partial V}{\partial r}$

## ⚠️ 风险提示

本工具仅供教育和研究用途，不构成投资建议。期权交易涉及高风险，可能导致本金损失。使用本工具进行实际交易前，请咨询专业金融顾问。

## 📝 更新日志

### v1.0.0 (2026-03-04)
- ✅ 初始版本发布
- ✅ BSM 期权定价模型
- ✅ 完整希腊值计算
- ✅ 7 种经典策略支持
- ✅ 交互式 Streamlit 界面
- ✅ 波动率分析工具
- ✅ 单元测试覆盖

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 👨‍💻 作者

Created with ❤️ by OpenClaw Agent
