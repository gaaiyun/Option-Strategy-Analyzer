"""
策略构建器模块 - 支持多种期权策略组合
跨式、宽跨式、牛市价差、熊市价差等
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from option_pricer import OptionPricer, price_option
from greeks_calculator import GreeksCalculator


class OptionLeg:
    """期权腿 - 单个期权头寸"""
    
    def __init__(self, option_type: str, strike: float, quantity: int = 1, 
                long: bool = True):
        """
        初始化期权腿
        
        参数:
            option_type: 'call' 或 'put'
            strike: 行权价
            quantity: 数量
            long: True 为买入，False 为卖出
        """
        self.option_type = option_type
        self.strike = strike
        self.quantity = quantity
        self.long = long
    
    def payoff_at_expiration(self, S_T: float) -> float:
        """
        计算到期时的收益
        
        参数:
            S_T: 到期时标的价格
        
        返回:
            收益
        """
        if self.option_type == 'call':
            intrinsic = max(0, S_T - self.strike)
        else:
            intrinsic = max(0, self.strike - S_T)
        
        if self.long:
            return self.quantity * intrinsic
        else:
            return -self.quantity * intrinsic


class OptionStrategy:
    """期权策略类"""
    
    def __init__(self, name: str, legs: List[OptionLeg], S: float, T: float, 
                r: float, sigma: float, q: float = 0.0):
        """
        初始化期权策略
        
        参数:
            name: 策略名称
            legs: 期权腿列表
            S: 当前标的价格
            T: 到期时间
            r: 无风险利率
            sigma: 波动率
            q: 股息率
        """
        self.name = name
        self.legs = legs
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def initial_cost(self) -> float:
        """
        计算初始成本（正值为净支出，负值为净收入）
        
        返回:
            初始成本
        """
        cost = 0.0
        for leg in self.legs:
            pricer = OptionPricer(self.S, leg.strike, self.T, self.r, self.sigma, self.q)
            price = pricer.calculate_price(leg.option_type)
            
            if leg.long:
                cost += leg.quantity * price
            else:
                cost -= leg.quantity * price
        
        return cost
    
    def payoff_at_expiration(self, S_T: float) -> float:
        """
        计算到期时的总收益
        
        参数:
            S_T: 到期时标的价格
        
        返回:
            总收益
        """
        total_payoff = 0.0
        for leg in self.legs:
            total_payoff += leg.payoff_at_expiration(S_T)
        return total_payoff
    
    def profit_at_expiration(self, S_T: float) -> float:
        """
        计算到期时的总利润（考虑初始成本）
        
        参数:
            S_T: 到期时标的价格
        
        返回:
            总利润
        """
        return self.payoff_at_expiration(S_T) - self.initial_cost()
    
    def breakeven_points(self, S_range: Tuple[float, float] = None, 
                        n_points: int = 1000) -> List[float]:
        """
        计算盈亏平衡点
        
        参数:
            S_range: 价格范围 (min, max)
            n_points: 采样点数
        
        返回:
            盈亏平衡点列表
        """
        if S_range is None:
            S_range = (self.S * 0.5, self.S * 1.5)
        
        S_values = np.linspace(S_range[0], S_range[1], n_points)
        profits = [self.profit_at_expiration(S) for S in S_values]
        
        breakevens = []
        for i in range(len(S_values) - 1):
            if profits[i] * profits[i+1] < 0:  # 符号变化
                # 线性插值
                S_be = S_values[i] + (S_values[i+1] - S_values[i]) * \
                       abs(profits[i]) / (abs(profits[i]) + abs(profits[i+1]))
                breakevens.append(S_be)
        
        return breakevens
    
    def max_profit(self, S_range: Tuple[float, float] = None) -> float:
        """
        计算最大利润
        
        参数:
            S_range: 价格范围
        
        返回:
            最大利润
        """
        if S_range is None:
            S_range = (self.S * 0.1, self.S * 2.0)
        
        S_values = np.linspace(S_range[0], S_range[1], 1000)
        profits = [self.profit_at_expiration(S) for S in S_values]
        return max(profits)
    
    def max_loss(self, S_range: Tuple[float, float] = None) -> float:
        """
        计算最大损失
        
        参数:
            S_range: 价格范围
        
        返回:
            最大损失（负值）
        """
        if S_range is None:
            S_range = (self.S * 0.1, self.S * 2.0)
        
        S_values = np.linspace(S_range[0], S_range[1], 1000)
        profits = [self.profit_at_expiration(S) for S in S_values]
        return min(profits)
    
    def calculate_strategy_greeks(self) -> Dict[str, float]:
        """
        计算策略的整体希腊值
        
        返回:
            希腊值字典
        """
        greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        
        for leg in self.legs:
            calc = GreeksCalculator(self.S, leg.strike, self.T, self.r, self.sigma, self.q)
            leg_greeks = calc.calculate_all(leg.option_type)
            
            multiplier = leg.quantity if leg.long else -leg.quantity
            
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                greeks[greek] += multiplier * leg_greeks[greek]
        
        return greeks
    
    def get_payoff_data(self, S_range: Tuple[float, float] = None, 
                       n_points: int = 200) -> pd.DataFrame:
        """
        获取盈亏数据用于绘图
        
        参数:
            S_range: 价格范围
            n_points: 数据点数
        
        返回:
            DataFrame 包含价格和盈亏数据
        """
        if S_range is None:
            S_range = (self.S * 0.5, self.S * 1.5)
        
        S_values = np.linspace(S_range[0], S_range[1], n_points)
        payoffs = [self.payoff_at_expiration(S) for S in S_values]
        profits = [self.profit_at_expiration(S) for S in S_values]
        
        return pd.DataFrame({
            'price': S_values,
            'payoff': payoffs,
            'profit': profits
        })


class StrategyBuilder:
    """策略构建器"""
    
    def __init__(self, S: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        初始化策略构建器
        
        参数:
            S: 当前标的价格
            T: 到期时间
            r: 无风险利率
            sigma: 波动率
            q: 股息率
        """
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def straddle(self, strike: float = None) -> OptionStrategy:
        """
        构建跨式策略（Straddle）- 买入相同行权价的看涨和看跌期权
        
        参数:
            strike: 行权价（默认等于当前价格）
        
        返回:
            OptionStrategy 对象
        """
        if strike is None:
            strike = self.S
        
        legs = [
            OptionLeg('call', strike, 1, True),
            OptionLeg('put', strike, 1, True)
        ]
        
        return OptionStrategy('跨式策略 (Straddle)', legs, self.S, self.T, 
                             self.r, self.sigma, self.q)
    
    def strangle(self, call_strike: float = None, put_strike: float = None) -> OptionStrategy:
        """
        构建宽跨式策略（Strangle）- 买入不同行权价的看涨和看跌期权
        
        参数:
            call_strike: 看涨期权行权价（默认高于当前价格）
            put_strike: 看跌期权行权价（默认低于当前价格）
        
        返回:
            OptionStrategy 对象
        """
        if call_strike is None:
            call_strike = self.S * 1.05
        if put_strike is None:
            put_strike = self.S * 0.95
        
        legs = [
            OptionLeg('call', call_strike, 1, True),
            OptionLeg('put', put_strike, 1, True)
        ]
        
        return OptionStrategy('宽跨式策略 (Strangle)', legs, self.S, self.T,
                             self.r, self.sigma, self.q)
    
    def bull_call_spread(self, lower_strike: float = None, 
                        upper_strike: float = None) -> OptionStrategy:
        """
        构建牛市价差策略（Bull Call Spread）
        
        参数:
            lower_strike: 低行权价（买入）
            upper_strike: 高行权价（卖出）
        
        返回:
            OptionStrategy 对象
        """
        if lower_strike is None:
            lower_strike = self.S * 0.95
        if upper_strike is None:
            upper_strike = self.S * 1.05
        
        legs = [
            OptionLeg('call', lower_strike, 1, True),
            OptionLeg('call', upper_strike, 1, False)
        ]
        
        return OptionStrategy('牛市价差 (Bull Call Spread)', legs, self.S, self.T,
                             self.r, self.sigma, self.q)
    
    def bear_put_spread(self, higher_strike: float = None, 
                       lower_strike: float = None) -> OptionStrategy:
        """
        构建熊市价差策略（Bear Put Spread）
        
        参数:
            higher_strike: 高行权价（买入）
            lower_strike: 低行权价（卖出）
        
        返回:
            OptionStrategy 对象
        """
        if higher_strike is None:
            higher_strike = self.S * 1.05
        if lower_strike is None:
            lower_strike = self.S * 0.95
        
        legs = [
            OptionLeg('put', higher_strike, 1, True),
            OptionLeg('put', lower_strike, 1, False)
        ]
        
        return OptionStrategy('熊市价差 (Bear Put Spread)', legs, self.S, self.T,
                             self.r, self.sigma, self.q)
    
    def iron_condor(self, put_lower: float = None, put_higher: float = None,
                   call_lower: float = None, call_higher: float = None) -> OptionStrategy:
        """
        构建铁鹰策略（Iron Condor）
        
        参数:
            put_lower: 看跌期权低行权价（卖出）
            put_higher: 看跌期权高行权价（买入）
            call_lower: 看涨期权低行权价（买入）
            call_higher: 看涨期权高行权价（卖出）
        
        返回:
            OptionStrategy 对象
        """
        if put_lower is None:
            put_lower = self.S * 0.90
        if put_higher is None:
            put_higher = self.S * 0.95
        if call_lower is None:
            call_lower = self.S * 1.05
        if call_higher is None:
            call_higher = self.S * 1.10
        
        legs = [
            OptionLeg('put', put_lower, 1, False),
            OptionLeg('put', put_higher, 1, True),
            OptionLeg('call', call_lower, 1, True),
            OptionLeg('call', call_higher, 1, False)
        ]
        
        return OptionStrategy('铁鹰策略 (Iron Condor)', legs, self.S, self.T,
                             self.r, self.sigma, self.q)
    
    def protective_put(self, strike: float = None) -> OptionStrategy:
        """
        构建保护性看跌策略（Protective Put）- 持有标的 + 买入看跌期权
        
        参数:
            strike: 行权价
        
        返回:
            OptionStrategy 对象
        """
        if strike is None:
            strike = self.S * 0.95
        
        # 简化：用合成方式近似
        legs = [
            OptionLeg('put', strike, 1, True)
        ]
        
        return OptionStrategy('保护性看跌 (Protective Put)', legs, self.S, self.T,
                             self.r, self.sigma, self.q)
    
    def covered_call(self, strike: float = None) -> OptionStrategy:
        """
        构建备兑看涨策略（Covered Call）- 持有标的 + 卖出看涨期权
        
        参数:
            strike: 行权价
        
        返回:
            OptionStrategy 对象
        """
        if strike is None:
            strike = self.S * 1.05
        
        legs = [
            OptionLeg('call', strike, 1, False)
        ]
        
        return OptionStrategy('备兑看涨 (Covered Call)', legs, self.S, self.T,
                             self.r, self.sigma, self.q)


if __name__ == "__main__":
    # 测试示例
    S = 100
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    builder = StrategyBuilder(S, T, r, sigma)
    
    # 测试跨式策略
    straddle = builder.straddle()
    print(f"\n{straddle.name}")
    print(f"初始成本：${straddle.initial_cost():.4f}")
    print(f"盈亏平衡点：{straddle.breakeven_points()}")
    print(f"最大利润：${straddle.max_profit():.4f}")
    print(f"最大损失：${straddle.max_loss():.4f}")
    
    # 测试牛市价差
    bull_spread = builder.bull_call_spread()
    print(f"\n{bull_spread.name}")
    print(f"初始成本：${bull_spread.initial_cost():.4f}")
    print(f"盈亏平衡点：{bull_spread.breakeven_points()}")
