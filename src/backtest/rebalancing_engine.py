# rebalancing_engine.py
"""
Dynamic portfolio rebalancing with transaction cost optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class RebalancingConfig:
    """Configuration for rebalancing strategy"""
    method: str = 'threshold'  # 'threshold', 'periodic', 'volatility_adjusted'
    threshold: float = 0.05  # For threshold-based rebalancing
    period_days: int = 30  # For periodic rebalancing
    transaction_cost: float = 0.001  # 10 basis points per trade
    min_trade_size: float = 0.01  # Minimum trade size (1%)
    max_turnover: float = 0.5  # Maximum portfolio turnover per rebalancing
    

class PortfolioRebalancer:
    """
    Dynamic rebalancing engine with multiple strategies.
    """
    
    def __init__(self, config: RebalancingConfig):
        self.config = config
        self.rebalance_history = []
        
    def calculate_drift(self, current_weights: np.ndarray, 
                       target_weights: np.ndarray) -> np.ndarray:
        """Calculate drift from target weights."""
        return current_weights - target_weights
    
    def calculate_turnover(self, current_weights: np.ndarray,
                          new_weights: np.ndarray) -> float:
        """Calculate portfolio turnover."""
        return np.sum(np.abs(new_weights - current_weights)) / 2.0
    
    def should_rebalance_threshold(self, current_weights: np.ndarray,
                                  target_weights: np.ndarray) -> bool:
        """
        Threshold-based rebalancing decision.
        Rebalance if any asset drifts beyond threshold.
        """
        drift = self.calculate_drift(current_weights, target_weights)
        max_drift = np.max(np.abs(drift))
        return max_drift > self.config.threshold
    
    def should_rebalance_periodic(self, last_rebalance_date: datetime,
                                 current_date: datetime) -> bool:
        """
        Time-based rebalancing decision.
        """
        days_elapsed = (current_date - last_rebalance_date).days
        return days_elapsed >= self.config.period_days
    
    def should_rebalance_volatility(self, current_weights: np.ndarray,
                                   target_weights: np.ndarray,
                                   recent_volatility: float) -> bool:
        """
        Volatility-adjusted rebalancing.
        Adjust threshold based on recent market volatility.
        """
        adjusted_threshold = self.config.threshold * (1 + recent_volatility)
        drift = self.calculate_drift(current_weights, target_weights)
        max_drift = np.max(np.abs(drift))
        return max_drift > adjusted_threshold
    
    def optimize_trades(self, current_weights: np.ndarray,
                       target_weights: np.ndarray,
                       prices: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Optimize trades considering transaction costs and constraints.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Desired target weights
            prices: Current asset prices
            
        Returns:
            optimized_weights: Optimal weights after considering costs
            trade_info: Dictionary with trade details
        """
        n_assets = len(current_weights)
        
        # Calculate raw trades needed
        raw_trades = target_weights - current_weights
        
        # Filter out small trades
        trades = np.where(np.abs(raw_trades) > self.config.min_trade_size,
                         raw_trades, 0.0)
        
        # Check turnover constraint
        turnover = np.sum(np.abs(trades)) / 2.0
        if turnover > self.config.max_turnover:
            # Scale down trades to meet turnover constraint
            scale_factor = self.config.max_turnover / turnover
            trades = trades * scale_factor
        
        # Calculate transaction costs
        total_cost = self.config.transaction_cost * np.sum(np.abs(trades))
        
        # New weights after trades
        optimized_weights = current_weights + trades
        
        # Renormalize
        optimized_weights = optimized_weights / np.sum(optimized_weights)
        
        trade_info = {
            'trades': trades,
            'turnover': turnover,
            'transaction_cost': total_cost,
            'n_trades': np.sum(np.abs(trades) > 0),
            'tracking_error': np.linalg.norm(optimized_weights - target_weights)
        }
        
        return optimized_weights, trade_info
    
    def simulate_rebalancing(self, returns_df: pd.DataFrame,
                            target_weights: np.ndarray,
                            initial_value: float = 100000.0) -> pd.DataFrame:
        """
        Simulate portfolio performance with rebalancing strategy.
        
        Args:
            returns_df: DataFrame with daily returns for each asset
            target_weights: Target portfolio weights
            initial_value: Initial portfolio value
            
        Returns:
            DataFrame with portfolio performance over time
        """
        n_assets = len(target_weights)
        dates = returns_df.index
        
        # Initialize
        portfolio_value = initial_value
        current_weights = target_weights.copy()
        last_rebalance_date = dates[0]
        
        results = []
        
        for i, date in enumerate(dates):
            # Get today's returns
            daily_returns = returns_df.iloc[i].values
            
            # Update portfolio value
            portfolio_return = np.dot(current_weights, daily_returns)
            portfolio_value *= (1 + portfolio_return)
            
            # Update weights due to price changes (drift)
            asset_values = current_weights * (1 + daily_returns)
            current_weights = asset_values / np.sum(asset_values)
            
            # Check if rebalancing needed
            should_rebalance = False
            if self.config.method == 'threshold':
                should_rebalance = self.should_rebalance_threshold(
                    current_weights, target_weights)
            elif self.config.method == 'periodic':
                should_rebalance = self.should_rebalance_periodic(
                    last_rebalance_date, date)
            
            # Perform rebalancing if needed
            if should_rebalance:
                prices = np.ones(n_assets)  # Simplified: assume unit prices
                new_weights, trade_info = self.optimize_trades(
                    current_weights, target_weights, prices)
                
                # Apply transaction costs
                cost = trade_info['transaction_cost'] * portfolio_value
                portfolio_value -= cost
                
                current_weights = new_weights
                last_rebalance_date = date
                
                self.rebalance_history.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'turnover': trade_info['turnover'],
                    'cost': cost,
                    'weights': current_weights.copy()
                })
            
            # Record results
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'portfolio_return': portfolio_return,
                'rebalanced': should_rebalance,
                **{f'weight_{j}': current_weights[j] for j in range(n_assets)}
            })
        
        return pd.DataFrame(results)
    
    def plot_rebalancing_analysis(self, performance_df: pd.DataFrame):
        """Plot rebalancing analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(performance_df['date'], performance_df['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weight evolution
        weight_cols = [col for col in performance_df.columns if col.startswith('weight_')]
        for col in weight_cols:
            axes[0, 1].plot(performance_df['date'], performance_df[col], 
                          label=col, alpha=0.7)
        axes[0, 1].set_title('Portfolio Weights Evolution')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rebalancing events
        rebalance_dates = performance_df[performance_df['rebalanced']]['date']
        rebalance_values = performance_df[performance_df['rebalanced']]['portfolio_value']
        axes[1, 0].plot(performance_df['date'], performance_df['portfolio_value'])
        axes[1, 0].scatter(rebalance_dates, rebalance_values, 
                          color='red', s=100, zorder=5, label='Rebalance')
        axes[1, 0].set_title('Rebalancing Events')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Portfolio Value ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Return distribution
        axes[1, 1].hist(performance_df['portfolio_return'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Return Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic return data
    np.random.seed(42)
    n_assets = 5
    n_days = 252
    
    # Simulate returns
    mu = np.array([0.10, 0.12, 0.08, 0.15, 0.11]) / 252  # Daily returns
    sigma = 0.02  # Daily volatility
    
    returns = np.random.normal(mu, sigma, (n_days, n_assets))
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    returns_df = pd.DataFrame(returns, index=dates, 
                             columns=[f'Asset_{i}' for i in range(n_assets)])
    
    # Target weights (equal-weighted)
    target_weights = np.ones(n_assets) / n_assets
    
    # Test threshold-based rebalancing
    config = RebalancingConfig(
        method='threshold',
        threshold=0.05,
        transaction_cost=0.001,
        min_trade_size=0.01
    )
    
    rebalancer = PortfolioRebalancer(config)
    
    print("Simulating rebalancing strategy...")
    performance = rebalancer.simulate_rebalancing(returns_df, target_weights)
    
    # Print summary statistics
    initial_value = performance['portfolio_value'].iloc[0]
    final_value = performance['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    n_rebalances = performance['rebalanced'].sum()
    
    print(f"\nPerformance Summary:")
    print(f"Initial Value: ${initial_value:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Rebalances: {n_rebalances}")
    
    if len(rebalancer.rebalance_history) > 0:
        total_costs = sum([r['cost'] for r in rebalancer.rebalance_history])
        print(f"Total Transaction Costs: ${total_costs:,.2f}")
    
    # Plot analysis
    rebalancer.plot_rebalancing_analysis(performance)
