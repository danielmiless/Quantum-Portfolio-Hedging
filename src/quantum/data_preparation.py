# data_preparation.py
"""
Financial data preparation for quantum portfolio optimization
"""


import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PortfolioDataPreparer:
    """
    Data preparation with multiple estimation methods and robustness checks.
    """
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str, 
                 benchmark: str = '^GSPC'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.data = None
        self.returns = None
        self.statistics = {}
    
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Download prices, volumes, and benchmark data."""
        print(f"Downloading data for {len(self.tickers)} assets...")
        
        try:
            # Download main asset data
            all_tickers = self.tickers + [self.benchmark]
            main_data = yf.download(all_tickers, 
                                  start=self.start_date, 
                                  end=self.end_date,
                                  progress=False)
            
            # Check if download was successful
            if main_data.empty:
                raise ValueError("Downloaded data is empty - check tickers and date range")
            
            # Handle different yfinance output formats
            prices_dict = {}
            volumes_dict = {}
            
            # Check if we have MultiIndex columns
            if isinstance(main_data.columns, pd.MultiIndex):
                # Get level 0 values to determine structure
                level_0_values = main_data.columns.get_level_values(0).unique().tolist()
                
                # Determine which price column to use
                if 'Adj Close' in level_0_values:
                    price_col = 'Adj Close'
                elif 'Close' in level_0_values:
                    price_col = 'Close'
                else:
                    raise ValueError(f"No price column found. Available: {level_0_values}")
                
                # Check format: ('Price', 'TICKER') vs ('TICKER', 'Price')
                if price_col in level_0_values:
                    # Format: ('Close', 'AAPL') or ('Adj Close', 'AAPL')
                    for ticker in all_tickers:
                        if ticker in main_data[price_col].columns:
                            prices_dict[ticker] = main_data[price_col][ticker]
                            volumes_dict[ticker] = main_data['Volume'][ticker]
                else:
                    # Format: ('AAPL', 'Close') or ('AAPL', 'Adj Close')
                    for ticker in all_tickers:
                        if ticker in main_data.columns.get_level_values(0):
                            prices_dict[ticker] = main_data[ticker][price_col]
                            volumes_dict[ticker] = main_data[ticker]['Volume']
            else:
                # Single ticker - no MultiIndex
                ticker = all_tickers[0]
                if 'Adj Close' in main_data.columns:
                    prices_dict[ticker] = main_data['Adj Close']
                elif 'Close' in main_data.columns:
                    prices_dict[ticker] = main_data['Close']
                else:
                    raise ValueError("No price column found in single ticker data")
                volumes_dict[ticker] = main_data['Volume']
            
            # Check if we got any data
            if not prices_dict:
                raise ValueError("No price data extracted - check ticker symbols")
            
            prices = pd.DataFrame(prices_dict).dropna()
            volumes = pd.DataFrame(volumes_dict).dropna()
            
            # Check if data is empty after dropna
            if prices.empty:
                raise ValueError("All price data is NaN - check date range and tickers")
            
            # Separate benchmark
            if self.benchmark in prices.columns:
                benchmark_prices = prices[self.benchmark]
                asset_prices = prices.drop(columns=[self.benchmark])
                asset_volumes = volumes.drop(columns=[self.benchmark])
            else:
                benchmark_prices = None
                asset_prices = prices
                asset_volumes = volumes
            
            # Store data
            self.data = {
                'prices': asset_prices,
                'volumes': asset_volumes,
                'benchmark': benchmark_prices
            }
            
            print(f"Downloaded {len(asset_prices.columns)} price series")
            if len(asset_prices) > 0:
                print(f"Date range: {asset_prices.index[0]} to {asset_prices.index[-1]}")
                print(f"Assets: {list(asset_prices.columns)}")
            
            return self.data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
    
    def calculate_statistics(self, lookback_window: int = 252) -> Dict:
        """
        Calculate comprehensive return statistics with multiple estimation methods.
        """
        if self.data is None:
            self.download_data()
        
        prices = self.data['prices']
        returns = prices.pct_change().dropna()
        
        print(f"Calculating advanced statistics...")
        
        statistics = {}
        
        # Basic return statistics
        statistics['returns'] = returns
        statistics['mean_returns'] = returns.mean() * 252  # Annualized
        statistics['volatilities'] = returns.std() * np.sqrt(252)  # Annualized
        
        # Multiple covariance estimators
        statistics['sample_cov'] = returns.cov() * 252  # Sample covariance
        statistics['ledoit_wolf_cov'] = self._ledoit_wolf_shrinkage(returns) * 252
        statistics['exponential_cov'] = self._exponential_weighting(returns) * 252
        
        # Risk metrics
        statistics['var_95'] = returns.quantile(0.05) * np.sqrt(252)
        statistics['cvar_95'] = returns[returns <= returns.quantile(0.05)].mean() * np.sqrt(252)
        statistics['max_drawdown'] = self._calculate_max_drawdown(prices)
        statistics['sharpe_ratios'] = statistics['mean_returns'] / statistics['volatilities']
        
        # Market-relative metrics
        if self.data['benchmark'] is not None:
            benchmark_returns = self.data['benchmark'].pct_change().dropna()
            statistics['beta'] = self._calculate_beta(returns, benchmark_returns)
            statistics['alpha'] = statistics['mean_returns'] - statistics['beta'] * benchmark_returns.mean() * 252
        
        # Higher moments
        statistics['skewness'] = returns.skew()
        statistics['kurtosis'] = returns.kurt()
        
        # Store for later use
        self.statistics = statistics
        
        print(f"Calculated statistics for {len(returns.columns)} assets")
        print(f"Return range: {statistics['mean_returns'].min():.3f} to {statistics['mean_returns'].max():.3f}")
        print(f"Volatility range: {statistics['volatilities'].min():.3f} to {statistics['volatilities'].max():.3f}")
        
        return statistics
    
    def _ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator for covariance matrix.
        More robust than sample covariance for small samples.
        """
        from sklearn.covariance import LedoitWolf
        
        lw = LedoitWolf()
        shrunk_cov = lw.fit(returns.values).covariance_
        
        return pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)
    
    def _exponential_weighting(self, returns: pd.DataFrame, 
                              half_life: int = 30) -> pd.DataFrame:
        """
        Exponentially weighted covariance matrix.
        Gives more weight to recent observations.
        """
        return returns.ewm(halflife=half_life).cov().iloc[-len(returns.columns):]
    
    def _calculate_max_drawdown(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate maximum drawdown for each asset."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_beta(self, asset_returns: pd.DataFrame, 
                       benchmark_returns: pd.Series) -> pd.Series:
        """Calculate beta for each asset relative to benchmark."""
        aligned_returns = asset_returns.align(benchmark_returns, axis=0, join='inner')[0]
        aligned_benchmark = asset_returns.align(benchmark_returns, axis=0, join='inner')[1]
        
        betas = {}
        for asset in aligned_returns.columns:
            covariance = np.cov(aligned_returns[asset], aligned_benchmark)[0, 1]
            benchmark_var = np.var(aligned_benchmark)
            betas[asset] = covariance / benchmark_var if benchmark_var > 0 else 0
        
        return pd.Series(betas)
    
    def get_optimization_inputs(self, method: str = 'ledoit_wolf') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mean returns and covariance matrix for optimization.
        
        Args:
            method: Covariance estimation method ('sample', 'ledoit_wolf', 'exponential')
        """
        if not self.statistics:
            self.calculate_statistics()
        
        mu = self.statistics['mean_returns'].values
        
        if method == 'sample':
            sigma = self.statistics['sample_cov'].values
        elif method == 'ledoit_wolf':
            sigma = self.statistics['ledoit_wolf_cov'].values
        elif method == 'exponential':
            sigma = self.statistics['exponential_cov'].values
        else:
            raise ValueError(f"Unknown covariance method: {method}")
        
        # Ensure positive definite
        sigma = self._nearest_positive_definite(sigma)
        
        return mu, sigma
    
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """
        Find the nearest positive definite matrix using eigenvalue decomposition.
        """
        B = (A + A.T) / 2  # Ensure symmetry
        _, s, Vh = np.linalg.svd(B)
        
        # Set negative eigenvalues to small positive values
        s[s < 1e-8] = 1e-8
        
        return Vh.T @ np.diag(s) @ Vh
    
    def plot_correlation_matrix(self, method: str = 'ledoit_wolf'):
        """Plot correlation matrix heatmap."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        _, sigma = self.get_optimization_inputs(method)
        volatilities = np.sqrt(np.diag(sigma))
        correlation = sigma / np.outer(volatilities, volatilities)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   xticklabels=self.tickers,
                   yticklabels=self.tickers)
        plt.title(f'Asset Correlation Matrix ({method.title()})')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Test with tech stocks
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    preparer = PortfolioDataPreparer(tickers, '2020-01-01', '2024-01-01')
    
    # Download and analyze data
    data = preparer.download_data()
    stats = preparer.calculate_statistics()
    
    # Get inputs for optimization
    mu, sigma = preparer.get_optimization_inputs('ledoit_wolf')
    
    print(f"\nExpected Returns (Annualized):")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {mu[i]:.3f} ({mu[i]*100:.1f}%)")
    
    print(f"\nVolatilities (Annualized):")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {np.sqrt(sigma[i,i]):.3f} ({np.sqrt(sigma[i,i])*100:.1f}%)")
    
    # Plot correlation matrix
    preparer.plot_correlation_matrix()
