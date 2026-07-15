"""
策略构建器模块单元测试
"""

import pytest
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_builder import OptionLeg, OptionStrategy, StrategyBuilder, UnderlyingLeg


class TestOptionLeg:
    """期权腿测试类"""
    
    def test_call_leg_payoff_long(self):
        """测试看涨期权多头收益"""
        leg = OptionLeg('call', strike=100, quantity=1, long=True)
        
        # 到期价格高于行权价
        assert leg.payoff_at_expiration(110) == 10
        assert leg.payoff_at_expiration(120) == 20
        
        # 到期价格低于行权价
        assert leg.payoff_at_expiration(90) == 0
        assert leg.payoff_at_expiration(100) == 0
    
    def test_call_leg_payoff_short(self):
        """测试看涨期权空头收益"""
        leg = OptionLeg('call', strike=100, quantity=1, long=False)
        
        # 到期价格高于行权价
        assert leg.payoff_at_expiration(110) == -10
        assert leg.payoff_at_expiration(120) == -20
        
        # 到期价格低于行权价
        assert leg.payoff_at_expiration(90) == 0
    
    def test_put_leg_payoff_long(self):
        """测试看跌期权多头收益"""
        leg = OptionLeg('put', strike=100, quantity=1, long=True)
        
        # 到期价格低于行权价
        assert leg.payoff_at_expiration(90) == 10
        assert leg.payoff_at_expiration(80) == 20
        
        # 到期价格高于行权价
        assert leg.payoff_at_expiration(110) == 0
        assert leg.payoff_at_expiration(100) == 0
    
    def test_put_leg_payoff_short(self):
        """测试看跌期权空头收益"""
        leg = OptionLeg('put', strike=100, quantity=1, long=False)
        
        # 到期价格低于行权价
        assert leg.payoff_at_expiration(90) == -10
        assert leg.payoff_at_expiration(80) == -20
        
        # 到期价格高于行权价
        assert leg.payoff_at_expiration(110) == 0
    
    def test_multiple_quantity(self):
        """测试多数量期权腿"""
        leg = OptionLeg('call', strike=100, quantity=5, long=True)
        
        assert leg.payoff_at_expiration(110) == 50
        assert leg.payoff_at_expiration(90) == 0


class TestOptionStrategy:
    """期权策略测试类"""
    
    @pytest.fixture
    def straddle_strategy(self):
        """跨式策略"""
        legs = [
            OptionLeg('call', 100, 1, True),
            OptionLeg('put', 100, 1, True)
        ]
        return OptionStrategy('Straddle', legs, S=100, T=0.25, r=0.05, sigma=0.2)
    
    @pytest.fixture
    def bull_spread_strategy(self):
        """牛市价差策略"""
        legs = [
            OptionLeg('call', 95, 1, True),
            OptionLeg('call', 105, 1, False)
        ]
        return OptionStrategy('Bull Spread', legs, S=100, T=0.25, r=0.05, sigma=0.2)
    
    def test_straddle_initial_cost(self, straddle_strategy):
        """测试跨式策略初始成本"""
        cost = straddle_strategy.initial_cost()
        assert cost > 0  # 买入策略需要成本
    
    def test_straddle_breakeven(self, straddle_strategy):
        """测试跨式策略盈亏平衡点"""
        breakevens = straddle_strategy.breakeven_points()
        assert len(breakevens) == 2  # 两个盈亏平衡点
        
        cost = straddle_strategy.initial_cost()
        # 盈亏平衡点应在行权价两侧
        assert breakevens[0] < 100
        assert breakevens[1] > 100
        
        # 近似验证
        assert abs(breakevens[0] - (100 - cost)) < 1
        assert abs(breakevens[1] - (100 + cost)) < 1
    
    def test_straddle_max_loss(self, straddle_strategy):
        """测试跨式策略最大损失"""
        max_loss = straddle_strategy.max_loss()
        cost = straddle_strategy.initial_cost()
        
        # 最大损失应在到期价格为行权价时
        assert abs(max_loss + cost) < 0.1  # 最大损失约等于初始成本
    
    def test_bull_spread_initial_cost(self, bull_spread_strategy):
        """测试牛市价差初始成本"""
        cost = bull_spread_strategy.initial_cost()
        assert cost > 0  # 净支出
    
    def test_bull_spread_max_profit(self, bull_spread_strategy):
        """测试牛市价差最大利润"""
        max_profit = bull_spread_strategy.max_profit()
        cost = bull_spread_strategy.initial_cost()
        
        # 最大利润 = 行权价差 - 初始成本
        max_theoretical = (105 - 95) - cost
        assert abs(max_profit - max_theoretical) < 0.5
    
    def test_bull_spread_max_loss(self, bull_spread_strategy):
        """测试牛市价差最大损失"""
        max_loss = bull_spread_strategy.max_loss()
        cost = bull_spread_strategy.initial_cost()
        
        # 最大损失 = 初始成本
        assert abs(max_loss + cost) < 0.5
    
    def test_bull_spread_breakeven(self, bull_spread_strategy):
        """测试牛市价差盈亏平衡点"""
        breakevens = bull_spread_strategy.breakeven_points()
        assert len(breakevens) == 1  # 一个盈亏平衡点
        
        cost = bull_spread_strategy.initial_cost()
        # 盈亏平衡点 = 低行权价 + 成本
        assert abs(breakevens[0] - (95 + cost)) < 1
    
    def test_strategy_greeks(self, straddle_strategy):
        """测试策略希腊值计算"""
        greeks = straddle_strategy.calculate_strategy_greeks()
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks
        
        # 跨式策略 Delta 应接近 0（平值）
        assert abs(greeks['delta']) < 0.2
        
        # Gamma 应为正
        assert greeks['gamma'] > 0
        
        # Theta 应为负（时间衰减）
        assert greeks['theta'] < 0
        
        # Vega 应为正（波动率有利）
        assert greeks['vega'] > 0
    
    def test_get_payoff_data(self, straddle_strategy):
        """测试盈亏数据生成"""
        df = straddle_strategy.get_payoff_data()
        
        assert 'price' in df.columns
        assert 'payoff' in df.columns
        assert 'profit' in df.columns
        
        assert len(df) == 200
        assert df['price'].min() >= 50
        assert df['price'].max() <= 150
    
    def test_profit_at_expiration(self, straddle_strategy):
        """测试到期利润计算"""
        cost = straddle_strategy.initial_cost()
        
        # 价格大幅上涨
        profit_up = straddle_strategy.profit_at_expiration(120)
        assert profit_up > 0
        
        # 价格大幅下跌
        profit_down = straddle_strategy.profit_at_expiration(80)
        assert profit_down > 0
        
        # 价格在行权价附近
        profit_atm = straddle_strategy.profit_at_expiration(100)
        assert profit_atm < 0
        assert abs(profit_atm + cost) < 0.1


class TestStrategyBuilder:
    """策略构建器测试类"""
    
    @pytest.fixture
    def builder(self):
        """策略构建器"""
        return StrategyBuilder(S=100, T=0.25, r=0.05, sigma=0.2)
    
    def test_straddle(self, builder):
        """测试跨式策略构建"""
        strategy = builder.straddle()
        
        assert strategy.name == '跨式策略 (Straddle)'
        assert len(strategy.legs) == 2
        assert strategy.legs[0].option_type == 'call'
        assert strategy.legs[1].option_type == 'put'
        assert strategy.legs[0].strike == 100
        assert strategy.legs[1].strike == 100
    
    def test_strangle(self, builder):
        """测试宽跨式策略构建"""
        strategy = builder.strangle(call_strike=105, put_strike=95)
        
        assert strategy.name == '宽跨式策略 (Strangle)'
        assert len(strategy.legs) == 2
        assert strategy.legs[0].strike == 105
        assert strategy.legs[1].strike == 95
    
    def test_bull_call_spread(self, builder):
        """测试牛市价差策略构建"""
        strategy = builder.bull_call_spread(lower_strike=95, upper_strike=105)
        
        assert strategy.name == '牛市价差 (Bull Call Spread)'
        assert len(strategy.legs) == 2
        assert strategy.legs[0].strike == 95
        assert strategy.legs[0].long == True
        assert strategy.legs[1].strike == 105
        assert strategy.legs[1].long == False
    
    def test_bear_put_spread(self, builder):
        """测试熊市价差策略构建"""
        strategy = builder.bear_put_spread(higher_strike=105, lower_strike=95)
        
        assert strategy.name == '熊市价差 (Bear Put Spread)'
        assert len(strategy.legs) == 2
        assert strategy.legs[0].option_type == 'put'
        assert strategy.legs[0].strike == 105
        assert strategy.legs[0].long == True
        assert strategy.legs[1].strike == 95
        assert strategy.legs[1].long == False
    
    def test_iron_condor(self, builder):
        """测试铁鹰策略构建"""
        strategy = builder.iron_condor(
            put_lower=90, put_higher=95,
            call_lower=105, call_higher=110
        )
        
        assert strategy.name == '铁鹰策略 (Iron Condor)'
        assert len(strategy.legs) == 4
        
        # 验证各腿
        assert strategy.legs[0].option_type == 'put'
        assert strategy.legs[0].strike == 90
        assert strategy.legs[0].long == True
        
        assert strategy.legs[1].option_type == 'put'
        assert strategy.legs[1].strike == 95
        assert strategy.legs[1].long == False
        
        assert strategy.legs[2].option_type == 'call'
        assert strategy.legs[2].strike == 105
        assert strategy.legs[2].long == False
        
        assert strategy.legs[3].option_type == 'call'
        assert strategy.legs[3].strike == 110
        assert strategy.legs[3].long == True
    
    def test_covered_call(self, builder):
        """测试备兑看涨策略构建"""
        strategy = builder.covered_call(strike=105)
        
        assert strategy.name == '备兑看涨 (Covered Call)'
        assert len(strategy.legs) == 2
        assert isinstance(strategy.legs[0], UnderlyingLeg)
        assert strategy.legs[0].quantity == 1
        assert strategy.legs[1].option_type == 'call'
        assert strategy.legs[1].strike == 105
        assert strategy.legs[1].long == False  # 卖出
    
    def test_protective_put(self, builder):
        """测试保护性看跌策略构建"""
        strategy = builder.protective_put(strike=95)
        
        assert strategy.name == '保护性看跌 (Protective Put)'
        assert len(strategy.legs) == 2
        assert isinstance(strategy.legs[0], UnderlyingLeg)
        assert strategy.legs[0].quantity == 1
        assert strategy.legs[1].option_type == 'put'
        assert strategy.legs[1].strike == 95
        assert strategy.legs[1].long == True  # 买入
    
    def test_iron_condor_profit_profile(self, builder):
        """测试铁鹰策略利润特征"""
        # 使用明确的行权价构建铁鹰
        strategy = builder.iron_condor(
            put_lower=90, put_higher=95,
            call_lower=105, call_higher=110
        )
        
        # 验证策略结构
        assert len(strategy.legs) == 4
        assert strategy.legs[0].option_type == 'put'
        assert strategy.legs[0].long == True   # 买入更低行权价看跌保护腿
        assert strategy.legs[1].option_type == 'put'
        assert strategy.legs[1].long == False  # 卖出较高行权价看跌
        assert strategy.legs[2].option_type == 'call'
        assert strategy.legs[2].long == False  # 卖出较低行权价看涨
        assert strategy.legs[3].option_type == 'call'
        assert strategy.legs[3].long == True   # 买入更高行权价看涨保护腿
        
        # 验证有初始成本（可能是 debit 或 credit）
        cost = strategy.initial_cost()
        assert cost != 0

        # 标准铁鹰应在中间区间盈利、两侧尾部受限亏损。
        assert strategy.profit_at_expiration(100) > 0
        assert strategy.profit_at_expiration(80) < 0
        assert strategy.profit_at_expiration(120) < 0
        assert np.isfinite(strategy.max_profit())
        assert np.isfinite(strategy.max_loss())
        
        # 验证盈亏数据可以生成
        df = strategy.get_payoff_data()
        assert len(df) > 0
        assert 'profit' in df.columns
    
    def test_bull_spread_profit_profile(self, builder):
        """测试牛市价差利润特征"""
        strategy = builder.bull_call_spread()
        
        # 价格上涨时盈利
        profit_up = strategy.profit_at_expiration(110)
        assert profit_up > 0
        
        # 价格下跌时亏损
        profit_down = strategy.profit_at_expiration(90)
        assert profit_down < 0


def test_unbounded_and_bounded_strategy_profit_limits():
    builder = StrategyBuilder(S=100, T=0.25, r=0.05, sigma=0.2)
    assert builder.straddle().max_profit() == np.inf
    assert np.isfinite(builder.straddle().max_loss())

    covered = builder.covered_call(strike=105)
    assert np.isfinite(covered.max_profit())
    assert np.isfinite(covered.max_loss())

    protective = builder.protective_put(strike=95)
    assert protective.max_profit() == np.inf
    assert np.isfinite(protective.max_loss())


def test_underlying_legs_change_payoff_and_delta():
    builder = StrategyBuilder(S=100, T=0.25, r=0.05, sigma=0.2)
    covered = builder.covered_call(strike=105)
    protective = builder.protective_put(strike=95)

    assert covered.payoff_at_expiration(90) == 90
    assert covered.payoff_at_expiration(120) == 105
    assert protective.payoff_at_expiration(80) == 95
    assert protective.payoff_at_expiration(120) == 120
    assert covered.calculate_strategy_greeks()["delta"] > 0
    assert protective.calculate_strategy_greeks()["delta"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
