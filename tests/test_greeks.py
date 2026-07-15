"""
希腊值计算模块单元测试
"""

import pytest
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greeks_calculator import GreeksCalculator, calculate_greeks


class TestGreeksCalculator:
    """希腊值计算器测试类"""
    
    @pytest.fixture
    def default_calculator(self):
        """默认计算器"""
        return GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.2)

    def test_rejects_invalid_financial_inputs(self):
        with pytest.raises(ValueError):
            GreeksCalculator(S=-100, K=100, T=0.25, r=0.05, sigma=0.2)

    def test_rejects_unknown_option_type_in_public_methods(self):
        calc = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        for method in (calc.delta, calc.gamma, calc.theta, calc.vega, calc.rho):
            with pytest.raises(ValueError, match="option_type"):
                method("other")
    
    def test_call_delta_atm(self, default_calculator):
        """测试平值看涨期权 Delta"""
        delta = default_calculator.delta('call')
        assert 0.4 < delta < 0.6  # 平值 Delta 约 0.5
        assert delta > 0  # 看涨 Delta 为正
    
    def test_put_delta_atm(self, default_calculator):
        """测试平值看跌期权 Delta"""
        delta = default_calculator.delta('put')
        assert -0.6 < delta < -0.4  # 平值 Delta 约 -0.5
        assert delta < 0  # 看跌 Delta 为负
    
    def test_call_delta_itm(self):
        """测试实值看涨期权 Delta"""
        calc = GreeksCalculator(S=110, K=100, T=0.25, r=0.05, sigma=0.2)
        delta = calc.delta('call')
        assert delta > 0.5  # 实值 Delta 更高
        assert delta < 1.0
    
    def test_call_delta_otm(self):
        """测试虚值看涨期权 Delta"""
        calc = GreeksCalculator(S=90, K=100, T=0.25, r=0.05, sigma=0.2)
        delta = calc.delta('call')
        assert delta < 0.5  # 虚值 Delta 更低
        assert delta > 0
    
    def test_gamma_positive(self, default_calculator):
        """测试 Gamma 为正"""
        gamma_call = default_calculator.gamma('call')
        gamma_put = default_calculator.gamma('put')
        
        assert gamma_call > 0
        assert gamma_put > 0
        assert abs(gamma_call - gamma_put) < 0.001  # 看涨看跌 Gamma 相同
    
    def test_gamma_atm_maximum(self):
        """测试平值时 Gamma 最大"""
        calc_atm = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        calc_itm = GreeksCalculator(S=110, K=100, T=0.25, r=0.05, sigma=0.2)
        calc_otm = GreeksCalculator(S=90, K=100, T=0.25, r=0.05, sigma=0.2)
        
        gamma_atm = calc_atm.gamma()
        gamma_itm = calc_itm.gamma()
        gamma_otm = calc_otm.gamma()
        
        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm
    
    def test_theta_call_negative(self, default_calculator):
        """测试看涨期权 Theta 为负（时间衰减）"""
        theta = default_calculator.theta('call')
        assert theta < 0  # 多头看涨期权时间衰减
    
    def test_theta_put_negative(self, default_calculator):
        """测试看跌期权 Theta 为负"""
        theta = default_calculator.theta('put')
        assert theta < 0  # 多头看跌期权时间衰减
    
    def test_theta_magnitude_atm(self):
        """测试平值期权 Theta 绝对值较大"""
        calc_atm = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        calc_ditm = GreeksCalculator(S=120, K=100, T=0.25, r=0.05, sigma=0.2)
        
        theta_atm = abs(calc_atm.theta('call'))
        theta_ditm = abs(calc_ditm.theta('call'))
        
        assert theta_atm > theta_ditm  # 平值时间衰减更快
    
    def test_vega_positive(self, default_calculator):
        """测试 Vega 为正"""
        vega_call = default_calculator.vega('call')
        vega_put = default_calculator.vega('put')
        
        assert vega_call > 0
        assert vega_put > 0
        assert abs(vega_call - vega_put) < 0.001  # 看涨看跌 Vega 相同
    
    def test_vega_atm_maximum(self):
        """测试平值时 Vega 最大"""
        calc_atm = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        calc_itm = GreeksCalculator(S=110, K=100, T=0.25, r=0.05, sigma=0.2)
        
        vega_atm = calc_atm.vega()
        vega_itm = calc_itm.vega()
        
        assert vega_atm > vega_itm
    
    def test_rho_call_positive(self, default_calculator):
        """测试看涨期权 Rho 为正"""
        rho = default_calculator.rho('call')
        assert rho > 0  # 利率上升对看涨有利
    
    def test_rho_put_negative(self, default_calculator):
        """测试看跌期权 Rho 为负"""
        rho = default_calculator.rho('put')
        assert rho < 0  # 利率上升对看跌不利
    
    def test_calculate_all(self, default_calculator):
        """测试计算所有希腊值"""
        greeks = default_calculator.calculate_all('call')
        
        assert 'price' in greeks
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks
        
        assert greeks['price'] > 0
        assert 0 < greeks['delta'] < 1
        assert greeks['gamma'] > 0
        assert greeks['theta'] < 0
        assert greeks['vega'] > 0
    
    def test_greek_table(self, default_calculator):
        """测试希腊值表格生成"""
        table = default_calculator.greek_table('call')
        
        assert isinstance(table, str)
        assert 'Delta' in table
        assert 'Gamma' in table
        assert 'Theta' in table
        assert 'Vega' in table
        assert 'Rho' in table
    
    def test_zero_time_to_expiry(self):
        """测试到期时希腊值"""
        calc = GreeksCalculator(S=100, K=100, T=0, r=0.05, sigma=0.2)
        
        assert calc.gamma() == 0
        assert calc.theta('call') == 0
        assert calc.vega() == 0
        assert calc.rho('call') == 0
    
    def test_delta_bounds(self):
        """测试 Delta 边界"""
        # 深度实值看涨
        calc_ditm = GreeksCalculator(S=150, K=100, T=0.25, r=0.05, sigma=0.2)
        assert 0.8 < calc_ditm.delta('call') < 1.0
        
        # 深度虚值看涨
        calc_dotm = GreeksCalculator(S=50, K=100, T=0.25, r=0.05, sigma=0.2)
        assert 0 < calc_dotm.delta('call') < 0.2
        
        # 深度实值看跌
        calc_ditm_put = GreeksCalculator(S=50, K=100, T=0.25, r=0.05, sigma=0.2)
        assert -1.0 < calc_ditm_put.delta('put') < -0.8
        
        # 深度虚值看跌
        calc_dotm_put = GreeksCalculator(S=150, K=100, T=0.25, r=0.05, sigma=0.2)
        assert -0.2 < calc_dotm_put.delta('put') < 0
    
    def test_gamma_with_volatility(self):
        """测试 Gamma 与波动率关系"""
        calc_low_vol = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.1)
        calc_high_vol = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.4)
        
        gamma_low = calc_low_vol.gamma()
        gamma_high = calc_high_vol.gamma()
        
        assert gamma_low > gamma_high  # 低波动率时 Gamma 更大
    
    def test_vega_with_time(self):
        """测试 Vega 与时间关系"""
        calc_short = GreeksCalculator(S=100, K=100, T=0.083, r=0.05, sigma=0.2)  # 1 个月
        calc_long = GreeksCalculator(S=100, K=100, T=1.0, r=0.05, sigma=0.2)  # 1 年
        
        vega_short = calc_short.vega()
        vega_long = calc_long.vega()
        
        assert vega_long > vega_short  # 长期限 Vega 更大
    
    def test_calculate_greeks_function(self):
        """测试便捷函数"""
        greeks = calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        
        assert 'price' in greeks
        assert 'delta' in greeks
        assert greeks['price'] > 0


class TestGreeksEdgeCases:
    """边界情况测试"""
    
    def test_very_high_volatility(self):
        """测试极高波动率下的希腊值"""
        calc = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=1.0)
        
        greeks = calc.calculate_all('call')
        assert greeks['delta'] > 0
        assert greeks['gamma'] > 0
        assert greeks['vega'] > 0
    
    def test_very_low_volatility(self):
        """测试极低波动率下的希腊值"""
        calc = GreeksCalculator(S=100, K=100, T=0.25, r=0.05, sigma=0.01)
        
        greeks = calc.calculate_all('call')
        assert greeks['gamma'] > 0  # Gamma 仍为正
        assert greeks['vega'] > 0
    
    def test_very_long_expiry(self):
        """测试极长期限下的希腊值"""
        calc = GreeksCalculator(S=100, K=100, T=10, r=0.05, sigma=0.2)
        
        greeks = calc.calculate_all('call')
        assert greeks['delta'] > 0.5
        assert greeks['theta'] < 0  # 仍为负


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
