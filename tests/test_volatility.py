"""
波动率分析模块单元测试
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volatility_analyzer import VolatilityAnalyzer, calculate_historical_volatility


class TestVolatilityAnalyzer:
    """波动率分析器测试类"""
    
    @pytest.fixture
    def sample_prices(self):
        """示例价格序列"""
        np.random.seed(42)
        n_days = 252
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices)
    
    @pytest.fixture
    def analyzer(self, sample_prices):
        """波动率分析器"""
        return VolatilityAnalyzer(prices=sample_prices)
    
    def test_historical_volatility(self, analyzer):
        """测试历史波动率计算"""
        vol = analyzer.historical_volatility()
        
        assert vol > 0
        assert vol < 2  # 年化波动率应在合理范围
        # 日波动 2% 对应年化约 31.7%
        assert 0.2 < vol < 0.5
    
    def test_historical_volatility_annualized(self, analyzer):
        """测试年化波动率"""
        vol_daily = analyzer.historical_volatility(annualize=False)
        vol_annual = analyzer.historical_volatility(annualize=True)
        
        assert vol_annual > vol_daily
        assert abs(vol_annual - vol_daily * np.sqrt(252)) < 0.001
    
    def test_historical_volatility_window(self, analyzer):
        """测试滚动窗口波动率"""
        vol_20 = analyzer.historical_volatility(window=20)
        vol_60 = analyzer.historical_volatility(window=60)
        
        assert vol_20 > 0
        assert vol_60 > 0
    
    def test_historical_volatility_series(self, analyzer):
        """测试波动率时间序列"""
        vol_series = analyzer.historical_volatility_series(window=20)
        
        # 由于收益率计算会丢失一个数据点，序列长度应为 prices - 1
        assert len(vol_series) == len(analyzer.prices) - 1
        assert vol_series.iloc[:19].isna().all()  # 前 19 个为 NaN
        assert not vol_series.iloc[19:].isna().any()  # 之后有值
    
    def test_realized_volatility(self, analyzer):
        """测试已实现波动率"""
        vol_20 = analyzer.realized_volatility(n_days=20)
        vol_60 = analyzer.realized_volatility(n_days=60)
        
        assert vol_20 > 0
        assert vol_60 > 0
    
    def test_implied_volatility(self, analyzer):
        """测试隐含波动率计算"""
        # 使用 BSM 理论价格反推
        from option_pricer import OptionPricer
        
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
        pricer = OptionPricer(S, K, T, r, sigma)
        call_price = pricer.european_call()
        
        iv = analyzer.implied_volatility(call_price, S, K, T, r, 'call')
        
        assert abs(iv - sigma) < 0.001  # 应接近原始波动率
    
    def test_volatility_surface(self, analyzer):
        """测试波动率曲面构建"""
        # 创建模拟期权链
        strikes = np.linspace(80, 120, 5)
        option_chain = pd.DataFrame({
            'strike': strikes,
            'call_price': [25, 18, 12, 7, 3],
            'put_price': [2, 5, 9, 14, 20]
        })
        
        iv_surface = analyzer.volatility_surface(option_chain, S=100, T=0.25, r=0.05)
        
        assert 'strike' in iv_surface.columns
        assert 'moneyness' in iv_surface.columns
        assert 'iv_call' in iv_surface.columns
        assert 'iv_put' in iv_surface.columns
        assert 'iv_avg' in iv_surface.columns
        
        assert len(iv_surface) == 5
    
    def test_volatility_skew(self, analyzer):
        """测试波动率偏斜计算"""
        # 创建模拟期权链
        strikes = np.linspace(80, 120, 9)
        option_chain = pd.DataFrame({
            'strike': strikes,
            'call_price': [25, 20, 16, 12, 8, 5, 3, 1.5, 0.5],
            'put_price': [0.5, 1.5, 3, 5, 8, 12, 16, 20, 25]
        })
        
        skew = analyzer.volatility_skew(option_chain, S=100, T=0.25, r=0.05)
        
        assert 'otm_put_iv' in skew
        assert 'atm_iv' in skew
        assert 'otm_call_iv' in skew
        assert '25d_skew' in skew
        assert 'risk_reversal' in skew
        assert 'butterfly' in skew
    
    def test_garch_volatility_forecast(self, analyzer):
        """测试 GARCH 波动率预测"""
        forecast = analyzer.garch_volatility_forecast(n_days=1)
        
        assert forecast > 0
        assert forecast < 2  # 年化波动率合理范围
    
    def test_garch_multi_day_forecast(self, analyzer):
        """测试多日 GARCH 预测"""
        forecast_1 = analyzer.garch_volatility_forecast(n_days=1)
        forecast_5 = analyzer.garch_volatility_forecast(n_days=5)
        forecast_10 = analyzer.garch_volatility_forecast(n_days=10)
        
        assert forecast_1 > 0
        assert forecast_5 > 0
        assert forecast_10 > 0
    
    def test_term_structure(self, analyzer):
        """测试波动率期限结构"""
        expirations = [0.25, 0.5, 1.0, 2.0]
        atm_vols = [0.25, 0.23, 0.22, 0.21]
        
        ts = analyzer.term_structure(expirations, atm_vols)
        
        assert 'expiration_years' in ts.columns
        assert 'expiration_days' in ts.columns
        assert 'atm_volatility' in ts.columns
        
        assert len(ts) == 4
        assert ts['expiration_days'].iloc[0] == 0.25 * 365
    
    def test_volatility_percentile(self, analyzer):
        """测试波动率百分位计算"""
        current_vol = 0.3
        historical_vols = analyzer.historical_volatility_series(window=20).dropna()
        
        percentile = VolatilityAnalyzer.get_volatility_percentile(current_vol, historical_vols)
        
        assert 0 <= percentile <= 100
    
    def test_no_prices_error(self):
        """测试无价格数据错误"""
        analyzer = VolatilityAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.historical_volatility()
    
    def test_calculate_returns(self, analyzer):
        """测试收益率计算"""
        assert analyzer.returns is not None
        assert len(analyzer.returns) == len(analyzer.prices) - 1


class TestVolatilityAnalyzerWithYFinance:
    """yfinance 数据测试类"""
    
    @pytest.mark.skip(reason="需要网络连接")
    def test_load_prices_from_yfinance(self):
        """测试从 Yahoo Finance 加载数据"""
        analyzer = VolatilityAnalyzer(symbol='AAPL')
        analyzer.load_prices(period='1mo')
        
        assert analyzer.prices is not None
        assert len(analyzer.prices) > 0


class TestVolatilityEdgeCases:
    """边界情况测试"""
    
    def test_constant_prices(self):
        """测试恒定价格序列"""
        prices = pd.Series([100.0] * 100)
        analyzer = VolatilityAnalyzer(prices=prices)
        
        vol = analyzer.historical_volatility()
        assert vol == 0  # 无波动
    
    def test_very_volatile_prices(self):
        """测试极高波动价格序列"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.1, 252)  # 日波动 10%
        prices = 100 * np.exp(np.cumsum(returns))
        prices = pd.Series(prices)
        
        analyzer = VolatilityAnalyzer(prices=prices)
        vol = analyzer.historical_volatility()
        
        assert vol > 1  # 年化波动率应很高
    
    def test_short_price_series(self):
        """测试短价格序列"""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 30)
        prices = 100 * np.exp(np.cumsum(returns))
        prices = pd.Series(prices)
        
        analyzer = VolatilityAnalyzer(prices=prices)
        vol = analyzer.historical_volatility()
        
        assert vol > 0
    
    def test_calculate_historical_volatility_function(self):
        """测试便捷函数"""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)
        prices = 100 * np.exp(np.cumsum(returns))
        prices = pd.Series(prices)
        
        vol = calculate_historical_volatility(prices)
        
        assert vol > 0
        assert vol < 2


class TestVolatilityAnalytics:
    """波动率分析特性测试"""
    
    @pytest.fixture
    def analyzer(self):
        """创建分析器"""
        np.random.seed(42)
        # 创建两段不同波动率的序列
        returns1 = np.random.normal(0.0005, 0.015, 126)  # 前半段低波动
        returns2 = np.random.normal(0.0005, 0.03, 126)   # 后半段高波动
        returns = np.concatenate([returns1, returns2])
        prices = 100 * np.exp(np.cumsum(returns))
        return VolatilityAnalyzer(prices=pd.Series(prices))
    
    def test_volatility_clustering(self, analyzer):
        """测试波动率聚集现象"""
        vol_series = analyzer.historical_volatility_series(window=20)
        
        # 后半段波动率应高于前半段
        first_half = vol_series.dropna().iloc[:len(vol_series.dropna())//2]
        second_half = vol_series.dropna().iloc[len(vol_series.dropna())//2:]
        
        assert second_half.mean() > first_half.mean()
    
    def test_volatility_mean_reversion(self, analyzer):
        """测试波动率均值回归"""
        vol_series = analyzer.historical_volatility_series(window=20).dropna()
        
        # 计算自相关
        autocorr = vol_series.autocorr(lag=1)
        
        assert autocorr > 0  # 波动率应有正自相关
        assert autocorr < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
