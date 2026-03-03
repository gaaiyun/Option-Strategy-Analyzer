"""
波动率分析模块 - 历史波动率、隐含波动率曲面
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from option_pricer import OptionPricer


class VolatilityAnalyzer:
    """波动率分析器"""
    
    def __init__(self, symbol: str = None, prices: pd.Series = None):
        """
        初始化波动率分析器
        
        参数:
            symbol: 股票代码（如果提供，将自动获取历史数据）
            prices: 价格序列（如果提供，将使用该数据）
        """
        self.symbol = symbol
        self.prices = prices
        self.returns = None
        
        if prices is not None:
            self.calculate_returns()
        elif symbol is not None and YFINANCE_AVAILABLE:
            self.load_prices()
    
    def load_prices(self, period: str = '1y', interval: str = '1d'):
        """
        从 Yahoo Finance 加载历史价格数据
        
        参数:
            period: 时间周期（'1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'）
            interval: 时间间隔（'1d', '1wk', '1mo'）
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance 未安装，请运行：pip install yfinance")
        
        ticker = yf.Ticker(self.symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if len(hist) == 0:
            raise ValueError(f"无法获取 {self.symbol} 的历史数据")
        
        self.prices = hist['Close']
        self.calculate_returns()
    
    def calculate_returns(self):
        """计算对数收益率"""
        if self.prices is None:
            raise ValueError("没有价格数据")
        
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
    
    def historical_volatility(self, window: int = None, annualize: bool = True) -> float:
        """
        计算历史波动率
        
        参数:
            window: 滚动窗口大小（None 表示使用全部数据）
            annualize: 是否年化
        
        返回:
            历史波动率
        """
        if self.returns is None:
            raise ValueError("没有收益率数据")
        
        if window is None:
            vol = self.returns.std()
        else:
            vol = self.returns.rolling(window=window).std().iloc[-1]
        
        if annualize:
            # 假设每年 252 个交易日
            vol *= np.sqrt(252)
        
        return vol
    
    def historical_volatility_series(self, window: int = 20, annualize: bool = True) -> pd.Series:
        """
        计算历史波动率时间序列
        
        参数:
            window: 滚动窗口大小
            annualize: 是否年化
        
        返回:
            历史波动率序列
        """
        if self.returns is None:
            raise ValueError("没有收益率数据")
        
        vol = self.returns.rolling(window=window).std()
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol
    
    def realized_volatility(self, n_days: int = 20) -> float:
        """
        计算已实现波动率（最近 N 天）
        
        参数:
            n_days: 天数
        
        返回:
            已实现波动率
        """
        if self.returns is None:
            raise ValueError("没有收益率数据")
        
        recent_returns = self.returns.iloc[-n_days:]
        vol = recent_returns.std() * np.sqrt(252)
        
        return vol
    
    def implied_volatility(self, option_price: float, S: float, K: float, T: float,
                          r: float, option_type: str = 'call') -> float:
        """
        计算单个期权的隐含波动率
        
        参数:
            option_price: 期权市场价格
            S: 标的价格
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            option_type: 'call' 或 'put'
        
        返回:
            隐含波动率
        """
        pricer = OptionPricer(S, K, T, r, 0.2)  # 初始猜测 20%
        iv = pricer.implied_volatility(option_price, option_type)
        return iv
    
    def volatility_surface(self, option_chain: pd.DataFrame, S: float, T: float,
                          r: float) -> pd.DataFrame:
        """
        构建隐含波动率曲面
        
        参数:
            option_chain: 期权链数据（包含 strike, call_price, put_price, expiry）
            S: 标的价格
            T: 到期时间
            r: 无风险利率
        
        返回:
            波动率曲面 DataFrame
        """
        iv_calls = []
        iv_puts = []
        
        for _, row in option_chain.iterrows():
            K = row['strike']
            
            # 计算看涨期权隐含波动率
            if 'call_price' in row and pd.notna(row['call_price']):
                iv_call = self.implied_volatility(row['call_price'], S, K, T, r, 'call')
                iv_calls.append(iv_call)
            else:
                iv_calls.append(np.nan)
            
            # 计算看跌期权隐含波动率
            if 'put_price' in row and pd.notna(row['put_price']):
                iv_put = self.implied_volatility(row['put_price'], S, K, T, r, 'put')
                iv_puts.append(iv_put)
            else:
                iv_puts.append(np.nan)
        
        result = pd.DataFrame({
            'strike': option_chain['strike'],
            'moneyness': option_chain['strike'] / S,
            'iv_call': iv_calls,
            'iv_put': iv_puts
        })
        
        # 计算平均隐含波动率
        result['iv_avg'] = (result['iv_call'] + result['iv_put']) / 2
        
        return result
    
    def volatility_skew(self, option_chain: pd.DataFrame, S: float, T: float,
                       r: float) -> Dict[str, float]:
        """
        计算波动率偏斜
        
        参数:
            option_chain: 期权链数据
            S: 标的价格
            T: 到期时间
            r: 无风险利率
        
        返回:
            偏斜指标字典
        """
        iv_surface = self.volatility_surface(option_chain, S, T, r)
        
        # 计算不同虚值程度的隐含波动率
        otm_puts = iv_surface[iv_surface['moneyness'] < 0.95]['iv_put'].mean()
        atm = iv_surface[(iv_surface['moneyness'] >= 0.95) & 
                        (iv_surface['moneyness'] <= 1.05)]['iv_avg'].mean()
        otm_calls = iv_surface[iv_surface['moneyness'] > 1.05]['iv_call'].mean()
        
        # 偏斜指标
        skew_25d = otm_puts - otm_calls  # 25 美元偏斜
        risk_reversal = otm_calls - otm_puts  # 风险逆转
        butterfly = (otm_puts + otm_calls) / 2 - atm  # 蝶式
        
        return {
            'otm_put_iv': otm_puts,
            'atm_iv': atm,
            'otm_call_iv': otm_calls,
            '25d_skew': skew_25d,
            'risk_reversal': risk_reversal,
            'butterfly': butterfly
        }
    
    def garch_volatility_forecast(self, n_days: int = 1, garch_params: Tuple = None) -> float:
        """
        使用简化 GARCH(1,1) 模型预测波动率
        
        参数:
            n_days: 预测天数
            garch_params: GARCH 参数 (omega, alpha, beta)，如果为 None 则使用估计值
        
        返回:
            预测波动率
        """
        if self.returns is None:
            raise ValueError("没有收益率数据")
        
        # 简化：使用样本矩估计 GARCH 参数
        if garch_params is None:
            # 长期波动率
            long_term_var = self.returns.var()
            # 简化假设
            omega = long_term_var * 0.1
            alpha = 0.1
            beta = 0.85
        else:
            omega, alpha, beta = garch_params
        
        # 当前条件方差
        current_var = self.returns.iloc[-1]**2
        
        # 预测未来方差
        forecast_var = omega + alpha * current_var + beta * self.returns.var()
        
        # 多步预测
        for _ in range(n_days - 1):
            forecast_var = omega + (alpha + beta) * forecast_var
        
        forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
        
        return forecast_vol
    
    def term_structure(self, expirations: List[float], atm_vols: List[float]) -> pd.DataFrame:
        """
        构建波动率期限结构
        
        参数:
            expirations: 到期时间列表（年）
            atm_vols: 平值隐含波动率列表
        
        返回:
            期限结构 DataFrame
        """
        df = pd.DataFrame({
            'expiration_years': expirations,
            'expiration_days': [t * 365 for t in expirations],
            'atm_volatility': atm_vols
        })
        
        return df
    
    @staticmethod
    def get_volatility_percentile(current_vol: float, historical_vols: pd.Series) -> float:
        """
        计算当前波动率的历史百分位
        
        参数:
            current_vol: 当前波动率
            historical_vols: 历史波动率序列
        
        返回:
            百分位（0-100）
        """
        percentile = (historical_vols < current_vol).mean() * 100
        return percentile


def calculate_historical_volatility(prices: pd.Series, window: int = None, 
                                   annualize: bool = True) -> float:
    """
    便捷函数：计算历史波动率
    
    参数:
        prices: 价格序列
        window: 滚动窗口
        annualize: 是否年化
    
    返回:
        历史波动率
    """
    analyzer = VolatilityAnalyzer(prices=prices)
    return analyzer.historical_volatility(window, annualize)


if __name__ == "__main__":
    # 测试示例
    if YFINANCE_AVAILABLE:
        print("测试波动率分析器...")
        
        # 使用示例数据
        np.random.seed(42)
        n_days = 252
        returns = np.random.normal(0.0005, 0.02, n_days)  # 日均收益 0.05%，日波动 2%
        prices = 100 * np.exp(np.cumsum(returns))
        prices = pd.Series(prices)
        
        analyzer = VolatilityAnalyzer(prices=prices)
        
        print(f"\n历史波动率：{analyzer.historical_volatility()*100:.2f}%")
        print(f"已实现波动率 (20 天): {analyzer.realized_volatility(20)*100:.2f}%")
        print(f"GARCH 预测波动率：{analyzer.garch_volatility_forecast()*100:.2f}%")
        
        # 波动率时间序列
        vol_series = analyzer.historical_volatility_series(window=20)
        print(f"\n最近 5 天波动率:")
        print(vol_series.tail() * 100)
    else:
        print("yfinance 未安装，跳过测试")
