"""
希腊值计算模块 - Delta, Gamma, Theta, Vega, Rho
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple
from option_pricer import OptionPricer, validate_market_inputs


class GreeksCalculator:
    """期权希腊值计算器"""
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        初始化希腊值计算器
        
        参数:
            S: 标的资产当前价格
            K: 行权价格
            T: 到期时间（年）
            r: 无风险利率
            sigma: 波动率
            q: 股息率
        """
        validate_market_inputs(S, K, T, r, sigma, q)
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

    @staticmethod
    def _validate_option_type(option_type: str) -> str:
        normalized = option_type.lower()
        if normalized not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        return normalized
    
    def delta(self, option_type: str = 'call') -> float:
        """
        计算 Delta - 标的资产价格变化对期权价格的影响
        
        Delta = ∂V/∂S
        
        参数:
            option_type: 'call' 或 'put'
        
        返回:
            Delta 值
        """
        option_type = self._validate_option_type(option_type)
        if self.T <= 0:
            if option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        
        d1, d2 = self._calculate_d1_d2()
        
        if option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            return np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
    
    def gamma(self, option_type: str = 'call') -> float:
        """
        计算 Gamma - 标的资产价格变化对 Delta 的影响
        
        Gamma = ∂²V/∂S² = ∂Delta/∂S
        
        参数:
            option_type: 'call' 或 'put'（看涨看跌 Gamma 相同）
        
        返回:
            Gamma 值
        """
        self._validate_option_type(option_type)
        if self.T <= 0:
            return 0.0
        
        d1, d2 = self._calculate_d1_d2()
        gamma = np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        return gamma
    
    def theta(self, option_type: str = 'call') -> float:
        """
        计算 Theta - 时间流逝对期权价格的影响
        
        Theta = ∂V/∂t（通常表示为每天的变化）
        
        参数:
            option_type: 'call' 或 'put'
        
        返回:
            Theta 值（每日）
        """
        option_type = self._validate_option_type(option_type)
        if self.T <= 0:
            return 0.0
        
        d1, d2 = self._calculate_d1_d2()
        
        term1 = -self.S * self.sigma * np.exp(-self.q * self.T) * norm.pdf(d1) / (2 * np.sqrt(self.T))
        
        if option_type == 'call':
            term2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
            term3 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            theta = term1 + term2 + term3
        else:
            term2 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
            term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            theta = term1 + term2 + term3
        
        # 转换为每日 Theta
        return theta / 365.0
    
    def vega(self, option_type: str = 'call') -> float:
        """
        计算 Vega - 波动率变化对期权价格的影响
        
        Vega = ∂V/∂σ
        
        参数:
            option_type: 'call' 或 'put'（看涨看跌 Vega 相同）
        
        返回:
            Vega 值（波动率变化 1% 时的价格变化）
        """
        self._validate_option_type(option_type)
        if self.T <= 0:
            return 0.0
        
        d1, d2 = self._calculate_d1_d2()
        vega = self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * norm.pdf(d1)
        
        # 转换为波动率变化 1% 时的影响
        return vega / 100.0
    
    def rho(self, option_type: str = 'call') -> float:
        """
        计算 Rho - 利率变化对期权价格的影响
        
        Rho = ∂V/∂r
        
        参数:
            option_type: 'call' 或 'put'
        
        返回:
            Rho 值（利率变化 1% 时的价格变化）
        """
        option_type = self._validate_option_type(option_type)
        if self.T <= 0:
            return 0.0
        
        d1, d2 = self._calculate_d1_d2()
        
        if option_type == 'call':
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        # 转换为利率变化 1% 时的影响
        return rho / 100.0
    
    def calculate_all(self, option_type: str = 'call') -> Dict[str, float]:
        """
        计算所有希腊值
        
        参数:
            option_type: 'call' 或 'put'
        
        返回:
            包含所有希腊值的字典
        """
        option_type = self._validate_option_type(option_type)
        pricer = OptionPricer(self.S, self.K, self.T, self.r, self.sigma, self.q)
        price = pricer.calculate_price(option_type)
        
        return {
            'price': price,
            'delta': self.delta(option_type),
            'gamma': self.gamma(option_type),
            'theta': self.theta(option_type),
            'vega': self.vega(option_type),
            'rho': self.rho(option_type)
        }
    
    def greek_table(self, option_type: str = 'call') -> str:
        """
        生成希腊值表格
        
        参数:
            option_type: 'call' 或 'put'
        
        返回:
            格式化的表格字符串
        """
        greeks = self.calculate_all(option_type)
        
        table = f"""
{'='*50}
期权希腊值分析 ({'看涨' if option_type == 'call' else '看跌'})
{'='*50}
价格：  ${greeks['price']:>10.4f}
Delta:  {greeks['delta']:>10.4f}
Gamma:  {greeks['gamma']:>10.4f}
Theta:  {greeks['theta']:>10.4f} (每日)
Vega:   {greeks['vega']:>10.4f} (1% 波动率变化)
Rho:    {greeks['rho']:>10.4f} (1% 利率变化)
{'='*50}
"""
        return table


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = 'call', q: float = 0.0) -> Dict[str, float]:
    """
    便捷函数：计算所有希腊值
    
    参数:
        S: 标的资产当前价格
        K: 行权价格
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'
        q: 股息率
    
    返回:
        包含所有希腊值的字典
    """
    calculator = GreeksCalculator(S, K, T, r, sigma, q)
    return calculator.calculate_all(option_type)


if __name__ == "__main__":
    # 测试示例
    S = 100
    K = 100
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    calculator = GreeksCalculator(S, K, T, r, sigma)
    
    print("看涨期权希腊值:")
    print(calculator.greek_table('call'))
    
    print("\n看跌期权希腊值:")
    print(calculator.greek_table('put'))
