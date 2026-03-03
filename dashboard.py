"""
期权策略分析平台 - 主界面
专业的期权定价与策略分析工具
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List

# 导入自定义模块
from option_pricer import OptionPricer, price_option
from greeks_calculator import GreeksCalculator, calculate_greeks
from strategy_builder import StrategyBuilder, OptionStrategy
from volatility_analyzer import VolatilityAnalyzer


# 页面配置
st.set_page_config(
    page_title="期权策略分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """主函数"""
    
    # 标题
    st.markdown('<p class="main-header">📊 期权策略分析平台</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏 - 全局参数
    st.sidebar.header("⚙️ 全局参数")
    
    S = st.sidebar.number_input("标的价格 (S)", value=100.0, min_value=0.01)
    r = st.sidebar.number_input("无风险利率 (%)", value=5.0, min_value=0.0, max_value=100.0) / 100
    q = st.sidebar.number_input("股息率 (%)", value=0.0, min_value=0.0, max_value=100.0) / 100
    sigma = st.sidebar.number_input("波动率 (%)", value=20.0, min_value=0.1, max_value=200.0) / 100
    T = st.sidebar.number_input("到期时间 (月)", value=3.0, min_value=0.1, max_value=120.0) / 12
    
    # 主选项卡
    tabs = st.tabs([
        "🎯 期权计算器",
        "📈 希腊值分析",
        "🧩 策略构建器",
        "📊 盈亏分析",
        "🌊 波动率分析"
    ])
    
    # 选项卡 1: 期权计算器
    with tabs[0]:
        option_calculator_tab(S, K=None, T=T, r=r, sigma=sigma, q=q)
    
    # 选项卡 2: 希腊值分析
    with tabs[1]:
        greeks_analysis_tab(S, K=None, T=T, r=r, sigma=sigma, q=q)
    
    # 选项卡 3: 策略构建器
    with tabs[2]:
        strategy_builder_tab(S, T=T, r=r, sigma=sigma, q=q)
    
    # 选项卡 4: 盈亏分析
    with tabs[3]:
        pnl_analysis_tab(S, T=T, r=r, sigma=sigma, q=q)
    
    # 选项卡 5: 波动率分析
    with tabs[4]:
        volatility_analysis_tab(S)


def option_calculator_tab(S: float, K: float, T: float, r: float, sigma: float, q: float):
    """期权计算器选项卡"""
    
    st.header("🎯 期权计算器")
    st.markdown("基于 Black-Scholes-Merton 模型的期权定价工具")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("输入参数")
        K = st.number_input("行权价格 (K)", value=S, min_value=0.01)
        option_type = st.selectbox("期权类型", ["看涨期权 (Call)", "看跌期权 (Put)"])
        american = st.checkbox("美式期权", value=False)
        
        is_call = option_type == "看涨期权 (Call)"
    
    with col2:
        st.subheader("定价结果")
        
        pricer = OptionPricer(S, K, T, r, sigma, q)
        
        if is_call:
            if american:
                price = pricer.american_call_approx()
            else:
                price = pricer.european_call()
            option_name = "看涨期权"
        else:
            if american:
                price = pricer.american_put_approx()
            else:
                price = pricer.european_put()
            option_name = "看跌期权"
        
        st.metric(f"{option_name}价格", f"${price:.4f}")
        
        # 内在价值和时间价值
        if is_call:
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        time_value = price - intrinsic
        
        col_price1, col_price2 = st.columns(2)
        col_price1.metric("内在价值", f"${intrinsic:.4f}")
        col_price2.metric("时间价值", f"${time_value:.4f}")
    
    # 价格敏感性分析
    st.subheader("💹 价格敏感性分析")
    
    # 标的价格变化
    S_range = np.linspace(S * 0.7, S * 1.3, 50)
    prices = []
    
    for s in S_range:
        pricer = OptionPricer(s, K, T, r, sigma, q)
        if is_call:
            prices.append(pricer.european_call() if not american else pricer.american_call_approx())
        else:
            prices.append(pricer.european_put() if not american else pricer.american_put_approx())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines', name=option_name,
                            line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=S_range, y=[max(0, s-K) if is_call else max(0, K-s) for s in S_range],
                            mode='lines', name='内在价值', line=dict(color='red', width=2, dash='dash')))
    
    fig.update_layout(
        title=f"{option_name}价格 vs 标的价格",
        xaxis_title="标的价格",
        yaxis_title="期权价格",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def greeks_analysis_tab(S: float, K: float, T: float, r: float, sigma: float, q: float):
    """希腊值分析选项卡"""
    
    st.header("📈 希腊值分析")
    st.markdown("Delta、Gamma、Theta、Vega、Rho 计算与可视化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        K = st.number_input("行权价格", value=S, key="greeks_k")
        option_type = st.selectbox("期权类型", ["看涨期权", "看跌期权"], key="greeks_type")
        is_call = option_type == "看涨期权"
    
    with col2:
        calculator = GreeksCalculator(S, K, T, r, sigma, q)
        greeks = calculator.calculate_all('call' if is_call else 'put')
        
        # 显示希腊值
        col_g1, col_g2 = st.columns(2)
        col_g1.metric("Delta", f"{greeks['delta']:.4f}")
        col_g1.metric("Gamma", f"{greeks['gamma']:.4f}")
        col_g2.metric("Theta (每日)", f"{greeks['theta']:.4f}")
        col_g2.metric("Vega", f"{greeks['vega']:.4f}")
        st.metric("Rho", f"{greeks['rho']:.4f}")
    
    # 希腊值可视化
    st.subheader("📊 希腊值敏感性分析")
    
    greek_to_plot = st.selectbox("选择要分析的希腊值", 
                                 ["Delta", "Gamma", "Theta", "Vega"],
                                 key="greek_select")
    
    x_param = st.selectbox("X 轴参数", 
                          ["标的价格", "波动率", "到期时间", "行权价"],
                          key="x_param")
    
    # 生成数据
    if x_param == "标的价格":
        x_values = np.linspace(S * 0.5, S * 1.5, 100)
        x_label = "标的价格"
        greek_values = []
        for s in x_values:
            calc = GreeksCalculator(s, K, T, r, sigma, q)
            if greek_to_plot == "Delta":
                greek_values.append(calc.delta('call' if is_call else 'put'))
            elif greek_to_plot == "Gamma":
                greek_values.append(calc.gamma())
            elif greek_to_plot == "Theta":
                greek_values.append(calc.theta('call' if is_call else 'put'))
            else:
                greek_values.append(calc.vega())
    
    elif x_param == "波动率":
        x_values = np.linspace(0.05, 0.5, 100)
        x_label = "波动率"
        greek_values = []
        for vol in x_values:
            calc = GreeksCalculator(S, K, T, r, vol, q)
            if greek_to_plot == "Delta":
                greek_values.append(calc.delta('call' if is_call else 'put'))
            elif greek_to_plot == "Gamma":
                greek_values.append(calc.gamma())
            elif greek_to_plot == "Theta":
                greek_values.append(calc.theta('call' if is_call else 'put'))
            else:
                greek_values.append(calc.vega())
    
    elif x_param == "到期时间":
        x_values = np.linspace(0.01, 1.0, 100)
        x_label = "到期时间 (年)"
        greek_values = []
        for t in x_values:
            calc = GreeksCalculator(S, K, t, r, sigma, q)
            if greek_to_plot == "Delta":
                greek_values.append(calc.delta('call' if is_call else 'put'))
            elif greek_to_plot == "Gamma":
                greek_values.append(calc.gamma())
            elif greek_to_plot == "Theta":
                greek_values.append(calc.theta('call' if is_call else 'put'))
            else:
                greek_values.append(calc.vega())
    
    else:  # 行权价
        x_values = np.linspace(K * 0.5, K * 1.5, 100)
        x_label = "行权价格"
        greek_values = []
        for k in x_values:
            calc = GreeksCalculator(S, k, T, r, sigma, q)
            if greek_to_plot == "Delta":
                greek_values.append(calc.delta('call' if is_call else 'put'))
            elif greek_to_plot == "Gamma":
                greek_values.append(calc.gamma())
            elif greek_to_plot == "Theta":
                greek_values.append(calc.theta('call' if is_call else 'put'))
            else:
                greek_values.append(calc.vega())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=greek_values, mode='lines',
                            line=dict(color='green', width=3)))
    fig.update_layout(
        title=f"{greek_to_plot} vs {x_label}",
        xaxis_title=x_label,
        yaxis_title=greek_to_plot,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def strategy_builder_tab(S: float, T: float, r: float, sigma: float, q: float):
    """策略构建器选项卡"""
    
    st.header("🧩 策略构建器")
    st.markdown("构建和分析多种期权策略组合")
    
    # 策略选择
    strategy_options = [
        "跨式策略 (Straddle)",
        "宽跨式策略 (Strangle)",
        "牛市价差 (Bull Call Spread)",
        "熊市价差 (Bear Put Spread)",
        "铁鹰策略 (Iron Condor)",
        "备兑看涨 (Covered Call)",
        "保护性看跌 (Protective Put)"
    ]
    
    selected_strategy = st.selectbox("选择策略", strategy_options)
    
    builder = StrategyBuilder(S, T, r, sigma, q)
    
    # 根据策略获取参数
    if selected_strategy == "跨式策略 (Straddle)":
        strike = st.number_input("行权价格", value=S)
        strategy = builder.straddle(strike)
    
    elif selected_strategy == "宽跨式策略 (Strangle)":
        col1, col2 = st.columns(2)
        call_strike = col1.number_input("看涨行权价", value=S * 1.05)
        put_strike = col2.number_input("看跌行权价", value=S * 0.95)
        strategy = builder.strangle(call_strike, put_strike)
    
    elif selected_strategy == "牛市价差 (Bull Call Spread)":
        col1, col2 = st.columns(2)
        lower = col1.number_input("低行权价 (买入)", value=S * 0.95)
        upper = col2.number_input("高行权价 (卖出)", value=S * 1.05)
        strategy = builder.bull_call_spread(lower, upper)
    
    elif selected_strategy == "熊市价差 (Bear Put Spread)":
        col1, col2 = st.columns(2)
        higher = col1.number_input("高行权价 (买入)", value=S * 1.05)
        lower = col2.number_input("低行权价 (卖出)", value=S * 0.95)
        strategy = builder.bear_put_spread(higher, lower)
    
    elif selected_strategy == "铁鹰策略 (Iron Condor)":
        col1, col2 = st.columns(2)
        put_lower = col1.number_input("看跌低行权价", value=S * 0.90)
        put_higher = col2.number_input("看跌高行权价", value=S * 0.95)
        col3, col4 = st.columns(2)
        call_lower = col3.number_input("看涨低行权价", value=S * 1.05)
        call_higher = col4.number_input("看涨高行权价", value=S * 1.10)
        strategy = builder.iron_condor(put_lower, put_higher, call_lower, call_higher)
    
    elif selected_strategy == "备兑看涨 (Covered Call)":
        strike = st.number_input("行权价格", value=S * 1.05)
        strategy = builder.covered_call(strike)
    
    else:  # 保护性看跌
        strike = st.number_input("行权价格", value=S * 0.95)
        strategy = builder.protective_put(strike)
    
    # 显示策略信息
    st.subheader(f"📋 {strategy.name}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("初始成本", f"${strategy.initial_cost():.4f}")
    col2.metric("最大利润", f"${strategy.max_profit():.4f}")
    col3.metric("最大损失", f"${strategy.max_loss():.4f}")
    
    breakevens = strategy.breakeven_points()
    if breakevens:
        col4.metric("盈亏平衡点", f"${breakevens[0]:.2f}" if len(breakevens) == 1 else f"${breakevens[0]:.2f} / ${breakevens[1]:.2f}")
    
    # 策略希腊值
    st.subheader("📊 策略希腊值")
    strategy_greeks = strategy.calculate_strategy_greeks()
    
    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric("Delta", f"{strategy_greeks['delta']:.4f}")
    col_g1.metric("Gamma", f"{strategy_greeks['gamma']:.4f}")
    col_g2.metric("Theta", f"{strategy_greeks['theta']:.4f}")
    col_g2.metric("Vega", f"{strategy_greeks['vega']:.4f}")
    col_g3.metric("Rho", f"{strategy_greeks['rho']:.4f}")
    
    # 盈亏图
    st.subheader("📈 盈亏分析图")
    
    df = strategy.get_payoff_data()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['price'], y=df['profit'], mode='lines',
                            name='利润', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=df['price'], y=df['payoff'], mode='lines',
                            name='收益', line=dict(color='green', width=2, dash='dash')))
    fig.add_hline(y=0, line_dash="dot", line_color="red")
    
    fig.update_layout(
        title=f"{strategy.name} - 到期盈亏图",
        xaxis_title="到期时标的价格",
        yaxis_title="利润/收益",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def pnl_analysis_tab(S: float, T: float, r: float, sigma: float, q: float):
    """盈亏分析选项卡"""
    
    st.header("📊 盈亏分析")
    st.markdown("交互式盈亏曲线和敏感性分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        K = st.number_input("行权价格", value=S)
        option_type = st.selectbox("期权类型", ["看涨期权", "看跌期权"])
        position = st.selectbox("头寸方向", ["买入多头", "卖出空头"])
        is_call = option_type == "看涨期权"
        is_long = position == "买入多头"
    
    with col2:
        # 计算关键指标
        pricer = OptionPricer(S, K, T, r, sigma, q)
        price = pricer.calculate_price('call' if is_call else 'put')
        
        if is_call:
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        st.metric("期权价格", f"${price:.4f}")
        st.metric("内在价值", f"${intrinsic:.4f}")
    
    # 盈亏图
    st.subheader("📈 到期盈亏图")
    
    S_range = np.linspace(S * 0.5, S * 1.5, 200)
    
    if is_call:
        payoff = [max(0, s - K) for s in S_range]
    else:
        payoff = [max(0, K - s) for s in S_range]
    
    if not is_long:
        payoff = [-p for p in payoff]
    
    profit = [p - price if is_long else price - p for p in payoff]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=profit, mode='lines',
                            name='利润', fill='tozeroy',
                            line=dict(color='blue' if is_long else 'red', width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    # 标记盈亏平衡点
    if is_call:
        if is_long:
            breakeven = K + price
        else:
            breakeven = K - price
    else:
        if is_long:
            breakeven = K - price
        else:
            breakeven = K + price
    
    if S * 0.5 <= breakeven <= S * 1.5:
        fig.add_vline(x=breakeven, line_dash="dash", line_color="green",
                     annotation_text=f"盈亏平衡：${breakeven:.2f}")
    
    fig.update_layout(
        title=f"{'买入' if is_long else '卖出'}{option_type} - 到期盈亏",
        xaxis_title="到期时标的价格",
        yaxis_title="利润",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 敏感性分析热力图
    st.subheader("🔥 利润敏感性热力图")
    
    S_grid = np.linspace(S * 0.7, S * 1.3, 50)
    sigma_grid = np.linspace(sigma * 0.5, sigma * 1.5, 50)
    
    profit_matrix = np.zeros((len(S_grid), len(sigma_grid)))
    
    for i, s in enumerate(S_grid):
        for j, vol in enumerate(sigma_grid):
            pricer = OptionPricer(s, K, T, r, vol, q)
            option_price = pricer.calculate_price('call' if is_call else 'put')
            
            if is_call:
                payoff = max(0, s - K)
            else:
                payoff = max(0, K - s)
            
            if is_long:
                profit_matrix[i, j] = payoff - price
            else:
                profit_matrix[i, j] = price - payoff
    
    fig = go.Figure(data=go.Heatmap(
        z=profit_matrix,
        x=sigma_grid * 100,
        y=S_grid,
        colorscale='RdYlGn',
        colorbar=dict(title="利润")
    ))
    
    fig.update_layout(
        title="利润 vs 标的价格和波动率",
        xaxis_title="波动率 (%)",
        yaxis_title="标的价格",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def volatility_analysis_tab(S: float):
    """波动率分析选项卡"""
    
    st.header("🌊 波动率分析")
    st.markdown("历史波动率、隐含波动率曲面和波动率期限结构")
    
    # 示例数据生成
    st.subheader("📊 历史波动率分析")
    
    np.random.seed(42)
    n_days = 252
    returns = np.random.normal(0.0005, sigma / np.sqrt(252), n_days)
    prices = S * np.exp(np.cumsum(returns))
    prices = pd.Series(prices)
    
    # 计算历史波动率
    rolling_vol = prices.pct_change().rolling(window=20).std() * np.sqrt(252)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1,
                       row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Scatter(x=list(range(len(prices))), y=prices,
                            name='价格', line=dict(color='blue')),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=list(range(len(rolling_vol))), y=rolling_vol * 100,
                            name='历史波动率', line=dict(color='red', width=2)),
                 row=2, col=1)
    
    fig.update_layout(
        title="价格和波动率时间序列",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="交易日", row=2, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="波动率 (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 隐含波动率曲面
    st.subheader("🎭 隐含波动率曲面")
    
    # 生成模拟期权链
    strikes = np.linspace(S * 0.8, S * 1.2, 10)
    expirations = [0.25, 0.5, 1.0]
    
    iv_data = []
    for T_exp in expirations:
        for K in strikes:
            moneyness = K / S
            # 模拟波动率微笑
            iv = sigma * (1 + 0.2 * (moneyness - 1)**2)
            iv *= (1 + 0.1 * np.sqrt(T_exp))  # 期限结构
            iv_data.append({
                'strike': K,
                'moneyness': moneyness,
                'expiration': T_exp,
                'iv': iv
            })
    
    iv_df = pd.DataFrame(iv_data)
    
    # 3D 波动率曲面
    fig = go.Figure()
    
    for T_exp in expirations:
        subset = iv_df[iv_df['expiration'] == T_exp]
        fig.add_trace(go.Scatter(x=subset['moneyness'], y=subset['iv'] * 100,
                                mode='lines+markers',
                                name=f'{T_exp*12:.0f}个月到期'))
    
    fig.update_layout(
        title="隐含波动率微笑曲线",
        xaxis_title="虚值/实值程度 (Moneyness)",
        yaxis_title="隐含波动率 (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 波动率统计
    st.subheader("📈 波动率统计")
    
    col1, col2, col3 = st.columns(3)
    current_vol = sigma * 100
    avg_vol = rolling_vol.mean() * 100
    vol_percentile = (rolling_vol.dropna() < sigma).mean() * 100
    
    col1.metric("当前波动率", f"{current_vol:.2f}%")
    col2.metric("平均历史波动率", f"{avg_vol:.2f}%")
    col3.metric("历史百分位", f"{vol_percentile:.1f}%")


if __name__ == "__main__":
    main()
