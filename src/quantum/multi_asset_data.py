# multi_asset_data.py
"""
Multi-asset class data integration and preprocessing
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AssetClassConfig:
    """Configuration for each asset class."""
    name: str
    tickers: List[str]
    currency: str = 'USD'
    liquidity_factor: float = 1.0


class MultiAssetDataManager:
    """Unified data manager for multiple asset classes."""
    
    def __init__(self):
        self.asset_classes = {}
        self.data = {}
        self.aligned_returns = None
        
    def add_asset_class(self, config: AssetClassConfig):
        """Add an asset class to the portfolio universe."""
        self.asset_classes[config.name] = config
        
    def download_all_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download data for all configured asset classes."""
        print(f"Downloading multi-asset data from {start_date} to {end_date}...")
        
        for class_name, config in self.asset_classes.items():
            print(f"  Downloading {class_name}...")
            
            try:
                # Download data
                data = yf.download(config.tickers, start=start_date, end=end_date, 
                                 progress=False, auto_adjust=True)
                
                # Extract prices with proper handling
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Adj Close' in data.columns.get_level_values(0):
                        prices = data['Adj Close']
                    elif 'Close' in data.columns.get_level_values(0):
                        prices = data['Close']
                    else:
                        print(f"    ✗ No price data found for {class_name}")
                        continue
                else:
                    prices = data['Close'] if 'Close' in data.columns else data['Adj Close']
                
                # Handle single ticker case
                if not isinstance(prices, pd.DataFrame):
                    prices = pd.DataFrame({config.tickers[0]: prices})
                
                # Store with asset class prefix
                prices.columns = [f"{class_name}_{col}" for col in prices.columns]
                self.data[class_name] = prices
                
                print(f"    ✓ Downloaded {len(prices.columns)} assets")
                
            except Exception as e:
                print(f"    ✗ Error downloading {class_name}: {e}")
                continue
        
        return self.data
    
    def align_returns(self, frequency: str = 'D') -> pd.DataFrame:
        """Align returns across all asset classes."""
        all_prices = []
        
        # Combine all price series
        for class_name, prices in self.data.items():
            all_prices.append(prices)
        
        if not all_prices:
            raise ValueError("No data available. Run download_all_data() first.")
        
        # Concatenate and align
        combined_prices = pd.concat(all_prices, axis=1)
        
        # Resample if needed
        if frequency != 'D':
            combined_prices = combined_prices.resample(frequency).last()
        
        # Calculate returns with new pandas syntax
        self.aligned_returns = combined_prices.pct_change(fill_method=None).dropna()
        
        print(f"Aligned returns: {self.aligned_returns.shape[0]} periods, {self.aligned_returns.shape[1]} assets")
        
        return self.aligned_returns
    
    def calculate_cross_asset_statistics(self) -> Dict:
        """Calculate statistics across asset classes."""
        if self.aligned_returns is None:
            self.align_returns()
        
        returns = self.aligned_returns
        
        stats = {
            'mean_returns': returns.mean() * 252,
            'volatilities': returns.std() * np.sqrt(252),
            'covariance': returns.cov() * 252,
            'correlation': returns.corr(),
            'sharpe_ratios': (returns.mean() / returns.std()) * np.sqrt(252)
        }
        
        # Asset class level aggregation
        class_stats = {}
        for class_name in self.asset_classes.keys():
            class_cols = [col for col in returns.columns if col.startswith(f"{class_name}_")]
            if class_cols:
                class_returns = returns[class_cols]
                class_stats[class_name] = {
                    'mean_return': class_returns.mean().mean() * 252,
                    'volatility': class_returns.std().mean() * np.sqrt(252),
                    'n_assets': len(class_cols)
                }
        
        stats['by_class'] = class_stats
        
        return stats
    
    def get_asset_class_weights(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate aggregate weights by asset class.
        
        Args:
            weights: Portfolio weights vector
            
        Returns:
            Dictionary of weights by asset class
        """
        if self.aligned_returns is None:
            raise ValueError("Must align returns first")
        
        asset_names = self.aligned_returns.columns
        class_weights = {}
        
        for class_name in self.asset_classes.keys():
            class_indices = [i for i, name in enumerate(asset_names) 
                           if name.startswith(f"{class_name}_")]
            class_weights[class_name] = sum(weights[i] for i in class_indices)
        
        return class_weights


# Example usage
if __name__ == "__main__":
    # Initialize data manager
    manager = MultiAssetDataManager()
    
    # Define asset classes
    equities = AssetClassConfig(
        name='Equities',
        tickers=['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
        currency='USD'
    )
    
    fixed_income = AssetClassConfig(
        name='Bonds',
        tickers=['TLT', 'IEF', 'LQD', 'HYG', 'EMB'],
        currency='USD'
    )
    
    commodities = AssetClassConfig(
        name='Commodities',
        tickers=['GLD', 'SLV', 'USO', 'DBA', 'DBC'],
        currency='USD'
    )
    
    crypto = AssetClassConfig(
        name='Crypto',
        tickers=['BTC-USD', 'ETH-USD'],
        currency='USD'
    )
    
    # Add asset classes
    manager.add_asset_class(equities)
    manager.add_asset_class(fixed_income)
    manager.add_asset_class(commodities)
    manager.add_asset_class(crypto)
    
    # Download data
    data = manager.download_all_data('2020-01-01', '2024-01-01')
    
    # Align returns
    returns = manager.align_returns(frequency='D')
    
    # Calculate statistics
    stats = manager.calculate_cross_asset_statistics()
    
    print("\n=== Multi-Asset Statistics ===")
    print("\nBy Asset Class:")
    for class_name, class_stat in stats['by_class'].items():
        print(f"  {class_name}:")
        print(f"    Mean Return: {class_stat['mean_return']:.2%}")
        print(f"    Volatility: {class_stat['volatility']:.2%}")
        print(f"    N Assets: {class_stat['n_assets']}")
    
    print(f"\nCross-Asset Correlation (sample):")
    print(stats['correlation'].iloc[:5, :5])
