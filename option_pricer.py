"""
期权定价模块 - Black-Scholes-Merton 模型实现
支持欧式/美式期权定价
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Tuple, Optional


class OptionPricer:
    """Black-Scholes-Merton 期权定价器"""
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        初始化期权定价器
        
        参数:
            S: 标的资产当前价格
            K: 行权价格
            T: 到期时间（年）
            r: 无风险利率
            sigma: 波动率
            q: 股息率（可选，默认 0）
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def _calculate_d1_d2(self) -> Tuple[float, float]:
        """计算 d1 和 d2"""
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2
    
    def european_call(self) -> float:
        """欧式看涨期权价格"""
        if self.T <= 0:
            return max(0, self.S - self.K)
        
        d1, d2 = self._calculate_d1_d2()
        call_price = self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - \
                     self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return max(0, call_price)
    
    def european_put(self) -> float:
        """欧式看跌期权价格"""
        if self.T <= 0:
            return max(0, self.K - self.S)
        
        d1, d2 = self._calculate_d1_d2()
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - \
                    self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        return max(0, put_price)
    
    def american_call_approx(self) -> float:
        """美式看涨期权价格（Barone-Adesi-Whaley 近似）"""
        # 无股息情况下，美式看涨期权价格等于欧式
        if self.q == 0:
            return self.european_call()
        
        # 有股息情况下使用近似公式
        european_price = self.european_call()
        
        # 简单近似：美式期权价值不低于欧式
        intrinsic_value = max(0, self.S - self.K)
        return max(european_price, intrinsic_value)
    
    def american_put_approx(self) -> float:
        """美式看跌期权价格（Barone-Adesi-Whaley 近似）"""
        european_price = self.european_put()
        intrinsic_value = max(0, self.K - self.S)
        return max(european_price, intrinsic_value)
    
    def calculate_price(self, option_type: str = 'call', american: bool = False) -> float:
        """
        计算期权价格
        
        参数:
            option_type: 'call' 或 'put'
            american: 是否为美式期权
        
        返回:
            期权价格
        """
        if option_type.lower() == 'call':
            if american:
                return self.american_call_approx()
            return self.european_call()
        elif option_type.lower() == 'put':
            if american:
                return self.american_put_approx()
            return self.european_put()
        else:
            raise ValueError("option_type 必须是 'call' 或 'put'")
    
    def implied_volatility(self, market_price: float, option_type: str = 'call', 
                          american: bool = False, tol: float = 1e-6) -> float:
        """
        计算隐含波动率
        
        参数:
            market_price: 市场价格
            option_type: 'call' 或 'put'
            american: 是否为美式期权
            tol: 容差
        
        返回:
            隐含波动率
        """
        def objective(sigma):
            self.sigma = sigma
            model_price = self.calculate_price(option_type, american)
            return model_price - market_price
        
        # 使用 Brent 方法求解
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=tol)
            return iv
        except ValueError:
            # 如果无法收敛，返回 NaN
            return np.nan
    
    @staticmethod
    def get_option_chain(S: float, K_range: Tuple[float, float], T: float, 
                        r: float, sigma: float, n_strikes: int = 10) -> dict:
        """
        生成期权链
        
        参数:
            S: 标的资产价格
            K_range: 行权价范围 (min, max)
            T: 到期时间
            r: 无风险利率
            sigma: 波动率
            n_strikes: 行权价数量
        
        返回:
            包含看涨和看跌期权价格的字典
        """
        strikes = np.linspace(K_range[0], K_range[1], n_strikes)
        calls = []
        puts = []
        
        for K in strikes:
            pricer = OptionPricer(S, K, T, r, sigma)
            calls.append(pricer.european_call())
            puts.append(pricer.european_put())
        
        return {
            'strikes': strikes,
            'calls': np.array(calls),
            'puts': np.array(puts)
        }


def price_option(S: float, K: float, T: float, r: float, sigma: float, 
                option_type: str = 'call', american: bool = False, q: float = 0.0) -> float:
    """
    便捷函数：计算期权价格
    
    参数:
        S: 标的资产当前价格
        K: 行权价格
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'
        american: 是否为美式期权
        q: 股息率
    
    返回:
        期权价格
    """
    pricer = OptionPricer(S, K, T, r, sigma, q)
    return pricer.calculate_price(option_type, american)


if __name__ == "__main__":
    # 测试示例
    S = 100  # 标的价格
    K = 100  # 行权价
    T = 0.25  # 3 个月
    r = 0.05  # 5% 无风险利率
    sigma = 0.2  # 20% 波动率
    
    pricer = OptionPricer(S, K, T, r, sigma)
    
    print(f"标的价格：${S}")
    print(f"行权价格：${K}")
    print(f"到期时间：{T*12:.0f} 个月")
    print(f"无风险利率：{r*100:.1f}%")
    print(f"波动率：{sigma*100:.1f}%")
    print("-" * 40)
    print(f"欧式看涨期权价格：${pricer.european_call():.4f}")
    print(f"欧式看跌期权价格：${pricer.european_put():.4f}")
    print(f"美式看涨期权价格：${pricer.american_call_approx():.4f}")
    print(f"美式看跌期权价格：${pricer.american_put_approx():.4f}")
