"""
期权定价模块单元测试
"""

import pytest
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from option_pricer import OptionPricer, price_option


class TestOptionPricer:
    """期权定价器测试类"""
    
    @pytest.fixture
    def default_pricer(self):
        """默认定价器"""
        return OptionPricer(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
    
    def test_european_call_atm(self, default_pricer):
        """测试平值欧式看涨期权"""
        price = default_pricer.european_call()
        assert price > 0
        assert price < 10  # 合理范围
        # 使用已知值验证 (BSM 模型下约 4.6)
        assert 3.0 < price < 6.0
    
    def test_european_put_atm(self, default_pricer):
        """测试平值欧式看跌期权"""
        price = default_pricer.european_put()
        assert price > 0
        assert price < 10
        # 平价关系：C - P = S - K*e^(-rT)
        call_price = default_pricer.european_call()
        put_call_parity = call_price - price
        expected = 100 - 100 * np.exp(-0.05 * 0.25)
        assert abs(put_call_parity - expected) < 0.01
    
    def test_in_the_money_call(self):
        """测试实值看涨期权"""
        pricer = OptionPricer(S=110, K=100, T=0.25, r=0.05, sigma=0.2)
        price = pricer.european_call()
        intrinsic = 110 - 100
        assert price > intrinsic  # 期权价格应高于内在价值
    
    def test_out_of_the_money_call(self):
        """测试虚值看涨期权"""
        pricer = OptionPricer(S=90, K=100, T=0.25, r=0.05, sigma=0.2)
        price = pricer.european_call()
        assert price > 0
        assert price < 5  # 虚值期权价格较低
    
    def test_american_call_no_dividend(self, default_pricer):
        """测试无股息美式看涨期权（应等于欧式）"""
        american = default_pricer.american_call_approx()
        european = default_pricer.european_call()
        assert abs(american - european) < 0.001
    
    def test_american_put_early_exercise(self):
        """测试美式看跌期权提前执行价值"""
        pricer = OptionPricer(S=80, K=100, T=0.25, r=0.05, sigma=0.2)
        american = pricer.american_put_approx()
        european = pricer.european_put()
        assert american >= european  # 美式不应低于欧式
    
    def test_zero_time_to_expiry_call(self):
        """测试到期时期权价格"""
        pricer = OptionPricer(S=100, K=100, T=0, r=0.05, sigma=0.2)
        call_price = pricer.european_call()
        put_price = pricer.european_put()
        assert call_price == 0  # 平值到期无价值
        assert put_price == 0
    
    def test_deep_in_the_money_call(self):
        """测试深度实值看涨期权"""
        pricer = OptionPricer(S=150, K=100, T=0.25, r=0.05, sigma=0.2)
        price = pricer.european_call()
        intrinsic = 150 - 100 * np.exp(-0.05 * 0.25)
        assert abs(price - intrinsic) < 1  # 接近远期内在价值
    
    def test_price_with_dividends(self):
        """测试含股息期权定价"""
        pricer_no_div = OptionPricer(S=100, K=100, T=0.25, r=0.05, sigma=0.2, q=0)
        pricer_div = OptionPricer(S=100, K=100, T=0.25, r=0.05, sigma=0.2, q=0.03)
        
        call_no_div = pricer_no_div.european_call()
        call_div = pricer_div.european_call()
        
        assert call_div < call_no_div  # 股息降低看涨期权价值
        
        put_no_div = pricer_no_div.european_put()
        put_div = pricer_div.european_put()
        
        assert put_div > put_no_div  # 股息提高看跌期权价值
    
    def test_implied_volatility(self, default_pricer):
        """测试隐含波动率计算"""
        # 先计算理论价格
        theoretical_price = default_pricer.european_call()
        
        # 从理论价格反推隐含波动率
        iv = default_pricer.implied_volatility(theoretical_price, 'call')
        
        assert abs(iv - 0.2) < 0.001  # 应接近原始波动率
    
    def test_calculate_price_method(self, default_pricer):
        """测试通用定价方法"""
        call1 = default_pricer.european_call()
        call2 = default_pricer.calculate_price('call')
        assert abs(call1 - call2) < 0.001
        
        put1 = default_pricer.european_put()
        put2 = default_pricer.calculate_price('put')
        assert abs(put1 - put2) < 0.001
    
    def test_invalid_option_type(self, default_pricer):
        """测试无效期权类型"""
        with pytest.raises(ValueError):
            default_pricer.calculate_price('invalid')
    
    def test_get_option_chain(self):
        """测试期权链生成"""
        chain = OptionPricer.get_option_chain(
            S=100, K_range=(90, 110), T=0.25, r=0.05, sigma=0.2, n_strikes=5
        )
        
        assert 'strikes' in chain
        assert 'calls' in chain
        assert 'puts' in chain
        assert len(chain['strikes']) == 5
        assert len(chain['calls']) == 5
        assert len(chain['puts']) == 5
        
        # 看涨期权价格随行权价增加而递减
        assert all(chain['calls'][i] >= chain['calls'][i+1] for i in range(4))
        
        # 看跌期权价格随行权价增加而递增
        assert all(chain['puts'][i] <= chain['puts'][i+1] for i in range(4))
    
    def test_price_option_function(self):
        """测试便捷定价函数"""
        price = price_option(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        assert price > 0
        
        price_put = price_option(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='put')
        assert price_put > 0


class TestOptionPricerEdgeCases:
    """边界情况测试"""
    
    def test_very_high_volatility(self):
        """测试极高波动率"""
        pricer = OptionPricer(S=100, K=100, T=0.25, r=0.05, sigma=1.0)
        price = pricer.european_call()
        assert price > 0
        assert price < 100  # 不应超过标的价格
    
    def test_very_low_volatility(self):
        """测试极低波动率"""
        pricer = OptionPricer(S=100, K=100, T=0.25, r=0.05, sigma=0.01)
        price = pricer.european_call()
        assert price >= 0
        assert price < 2  # 波动率极低时期权价值很小 (主要是内在价值)
    
    def test_very_long_expiry(self):
        """测试极长期限"""
        pricer = OptionPricer(S=100, K=100, T=10, r=0.05, sigma=0.2)
        price = pricer.european_call()
        assert price > 0
        assert price < 100
    
    def test_very_short_expiry(self):
        """测试极短期限"""
        pricer = OptionPricer(S=100, K=100, T=0.001, r=0.05, sigma=0.2)
        price = pricer.european_call()
        assert price >= 0
        assert price < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
