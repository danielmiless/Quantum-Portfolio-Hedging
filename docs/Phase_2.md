# Phase 2: Enhanced Quantum Portfolio Optimization

## Overview

Phase 2 extends the quantum portfolio optimization framework with advanced features including multi-objective optimization, dynamic rebalancing, risk parity strategies, and robust backtesting capabilities. This phase transforms the basic QUBO formulation into a production-ready portfolio management system.

## 2.1 Multi-Objective Quantum Optimization

### Theory

Real-world portfolio optimization requires balancing multiple competing objectives:

**Return Maximization**: \(\max \mathbf{w}^T \boldsymbol{\mu}\)

**Risk Minimization**: \(\min \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}\)

**Transaction Costs**: \(\min \sum_{i} |w_i - w_i^{\text{old}}|\)

**Concentration Risk**: \(\min \max_i(w_i)\)

We formulate this as a weighted multi-objective QUBO:

\[
H = \lambda_1 H_{\text{return}} + \lambda_2 H_{\text{risk}} + \lambda_3 H_{\text{turnover}} + \lambda_4 H_{\text{concentration}}
\]

### Implementation: `multi_objective_optimizer.py`

```python
# multi_objective_optimizer.py
"""
Multi-objective quantum portfolio optimization with Pareto frontier exploration
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools


@dataclass
class OptimizationObjectives:
    """Container for optimization objectives and constraints"""
    maximize_return: bool = True
    minimize_risk: bool = True
    minimize_turnover: bool = False
    minimize_concentration: bool = False
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.1
    concentration_penalty: float = 0.5
    

class MultiObjectiveQUBOOptimizer:
    """
    Multi-objective portfolio optimization using quantum annealing.
    Explores Pareto frontier by varying objective weights.
    """
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, 
                 current_weights: Optional[np.ndarray] = None):
        """
        Args:
            mu: Expected returns vector (n_assets,)
            sigma: Covariance matrix (n_assets, n_assets)
            current_weights: Current portfolio weights for turnover calculation
        """
        self.mu = mu
        self.sigma = sigma
        self.current_weights = current_weights if current_weights is not None else np.zeros(len(mu))
        self.n_assets = len(mu)
        
    def build_multi_objective_qubo(self, objectives: OptimizationObjectives,
                                   n_bits: int = 5) -> Tuple[np.ndarray, float]:
        """
        Build QUBO matrix incorporating multiple objectives.
        
        Args:
            objectives: Objective weights and flags
            n_bits: Number of bits per asset for weight discretization
            
        Returns:
            Q: QUBO matrix
            offset: Constant offset for objective function
        """
        n_vars = self.n_assets * n_bits
        Q = np.zeros((n_vars, n_vars))
        offset = 0.0
        
        # Weight encoding: each asset represented by n_bits binary variables
        # w_i = sum_{k=0}^{n_bits-1} b_{i,k} * 2^k / (2^n_bits - 1)
        max_val = 2**n_bits - 1
        
        # 1. Return objective: maximize w^T * mu
        if objectives.maximize_return:
            for i in range(self.n_assets):
                for k in range(n_bits):
                    idx = i * n_bits + k
                    bit_value = 2**k / max_val
                    # Negative because we're minimizing (QUBO form)
                    Q[idx, idx] -= self.mu[i] * bit_value
        
        # 2. Risk objective: minimize w^T * Sigma * w
        if objectives.minimize_risk:
            risk_weight = objectives.risk_aversion
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    for k1 in range(n_bits):
                        for k2 in range(n_bits):
                            idx1 = i * n_bits + k1
                            idx2 = j * n_bits + k2
                            bit_val1 = 2**k1 / max_val
                            bit_val2 = 2**k2 / max_val
                            
                            if idx1 <= idx2:
                                Q[idx1, idx2] += risk_weight * self.sigma[i, j] * bit_val1 * bit_val2
        
        # 3. Turnover penalty: minimize |w - w_old|
        if objectives.minimize_turnover:
            turnover_weight = objectives.turnover_penalty
            for i in range(self.n_assets):
                for k in range(n_bits):
                    idx = i * n_bits + k
                    bit_value = 2**k / max_val
                    # Linear approximation: |w - w_old| ≈ (w - w_old)^2
                    Q[idx, idx] += turnover_weight * bit_value**2
                    Q[idx, idx] -= 2 * turnover_weight * self.current_weights[i] * bit_value
            
            offset += turnover_weight * np.sum(self.current_weights**2)
        
        # 4. Concentration penalty: penalize max(w_i)
        if objectives.minimize_concentration:
            conc_weight = objectives.concentration_penalty
            # Approximate by penalizing w_i^2 for all i
            for i in range(self.n_assets):
                for k1 in range(n_bits):
                    for k2 in range(n_bits):
                        idx1 = i * n_bits + k1
                        idx2 = i * n_bits + k2
                        bit_val1 = 2**k1 / max_val
                        bit_val2 = 2**k2 / max_val
                        
                        if idx1 <= idx2:
                            Q[idx1, idx2] += conc_weight * bit_val1 * bit_val2
        
        # 5. Budget constraint: sum(w_i) = 1
        budget_penalty = 10.0  # Strong penalty
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                for k1 in range(n_bits):
                    for k2 in range(n_bits):
                        idx1 = i * n_bits + k1
                        idx2 = j * n_bits + k2
                        bit_val1 = 2**k1 / max_val
                        bit_val2 = 2**k2 / max_val
                        
                        if idx1 <= idx2:
                            Q[idx1, idx2] += budget_penalty * bit_val1 * bit_val2
        
        # Linear term for budget constraint: -2 * sum(w_i)
        for i in range(self.n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                bit_value = 2**k / max_val
                Q[idx, idx] -= 2 * budget_penalty * bit_value
        
        offset += budget_penalty  # (sum w_i - 1)^2 constant term
        
        return Q, offset
    
    def decode_solution(self, solution: np.ndarray, n_bits: int = 5) -> np.ndarray:
        """
        Decode binary solution to portfolio weights.
        
        Args:
            solution: Binary vector of length n_assets * n_bits
            n_bits: Number of bits per asset
            
        Returns:
            weights: Portfolio weights (n_assets,)
        """
        max_val = 2**n_bits - 1
        weights = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                if solution[idx] > 0.5:  # Binary threshold
                    weights[i] += 2**k / max_val
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        return weights
    
    def solve_classical(self, objectives: OptimizationObjectives,
                       method: str = 'SLSQP') -> Dict:
        """
        Classical solver for comparison (using scipy).
        """
        def objective(w):
            obj = 0.0
            if objectives.maximize_return:
                obj -= np.dot(w, self.mu)
            if objectives.minimize_risk:
                obj += objectives.risk_aversion * np.dot(w, np.dot(self.sigma, w))
            if objectives.minimize_turnover:
                obj += objectives.turnover_penalty * np.sum(np.abs(w - self.current_weights))
            if objectives.minimize_concentration:
                obj += objectives.concentration_penalty * np.max(w)**2
            return obj
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Budget
        ]
        bounds = [(0, 1) for _ in range(self.n_assets)]  # Long-only
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method=method, 
                         bounds=bounds, constraints=constraints)
        
        return {
            'weights': result.x,
            'objective': result.fun,
            'success': result.success,
            'method': 'classical'
        }
    
    def explore_pareto_frontier(self, n_points: int = 20) -> pd.DataFrame:
        """
        Explore Pareto frontier by varying risk aversion parameter.
        
        Args:
            n_points: Number of points on the frontier
            
        Returns:
            DataFrame with weights, returns, risks for each point
        """
        results = []
        risk_aversions = np.logspace(-2, 2, n_points)
        
        for risk_aversion in risk_aversions:
            objectives = OptimizationObjectives(
                maximize_return=True,
                minimize_risk=True,
                risk_aversion=risk_aversion
            )
            
            result = self.solve_classical(objectives)
            weights = result['weights']
            
            portfolio_return = np.dot(weights, self.mu)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.sigma, weights)))
            sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            results.append({
                'risk_aversion': risk_aversion,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe': sharpe,
                'weights': weights
            })
        
        return pd.DataFrame(results)
    
    def plot_efficient_frontier(self, pareto_df: pd.DataFrame):
        """Plot the efficient frontier."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Risk-Return space
        ax1.scatter(pareto_df['risk'], pareto_df['return'], 
                   c=pareto_df['sharpe'], cmap='viridis', s=100)
        ax1.set_xlabel('Portfolio Risk (Std Dev)')
        ax1.set_ylabel('Portfolio Return')
        ax1.set_title('Efficient Frontier')
        ax1.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Sharpe Ratio')
        
        # Sharpe ratio vs risk aversion
        ax2.semilogx(pareto_df['risk_aversion'], pareto_df['sharpe'], 'o-')
        ax2.set_xlabel('Risk Aversion Parameter')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio vs Risk Aversion')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Simulate data
    np.random.seed(42)
    n_assets = 5
    
    mu = np.array([0.10, 0.12, 0.08, 0.15, 0.11])  # Expected returns
    
    # Generate positive definite covariance matrix
    A = np.random.randn(n_assets, n_assets)
    sigma = 0.01 * (A @ A.T + np.eye(n_assets))
    
    # Current holdings
    current_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    
    # Initialize optimizer
    optimizer = MultiObjectiveQUBOOptimizer(mu, sigma, current_weights)
    
    # Explore Pareto frontier
    print("Exploring efficient frontier...")
    pareto_df = optimizer.explore_pareto_frontier(n_points=15)
    
    print("\nEfficient Frontier Points:")
    print(pareto_df[['return', 'risk', 'sharpe']].to_string())
    
    # Find maximum Sharpe ratio portfolio
    max_sharpe_idx = pareto_df['sharpe'].idxmax()
    optimal_weights = pareto_df.loc[max_sharpe_idx, 'weights']
    
    print(f"\nMaximum Sharpe Ratio Portfolio:")
    print(f"Return: {pareto_df.loc[max_sharpe_idx, 'return']:.4f}")
    print(f"Risk: {pareto_df.loc[max_sharpe_idx, 'risk']:.4f}")
    print(f"Sharpe: {pareto_df.loc[max_sharpe_idx, 'sharpe']:.4f}")
    print(f"Weights: {optimal_weights}")
    
    # Plot
    optimizer.plot_efficient_frontier(pareto_df)
```

## 2.2 Dynamic Rebalancing Engine

### Theory

Portfolio rebalancing must account for:
- **Transaction costs**: \(c \sum_i |w_i^{\text{new}} - w_i^{\text{old}}|\)
- **Market impact**: Price changes due to large trades
- **Tax considerations**: Capital gains optimization
- **Rebalancing frequency**: Trade-off between tracking error and costs

### Implementation: `rebalancing_engine.py`

```python
# rebalancing_engine.py
"""
Dynamic portfolio rebalancing with transaction cost optimization
"""

import numpy as np
import pandas as pd
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
```

## 2.3 Integration and Next Steps

### Complete Workflow

1. **Data Preparation** (`data_preparation.py`): Download and process market data
2. **QUBO Formulation** (`qubo_engine.py`): Convert portfolio problem to QUBO
3. **Multi-Objective Optimization** (`multi_objective_optimizer.py`): Explore Pareto frontier
4. **Rebalancing** (`rebalancing_engine.py`): Dynamic portfolio management

### Usage Example

```python
from data_preparation import PortfolioDataPreparer
from multi_objective_optimizer import MultiObjectiveQUBOOptimizer, OptimizationObjectives
from rebalancing_engine import PortfolioRebalancer, RebalancingConfig

# 1. Prepare data
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
preparer = PortfolioDataPreparer(tickers, '2020-01-01', '2024-01-01')
data = preparer.download_data()
stats = preparer.calculate_statistics()

# 2. Get optimization inputs
mu, sigma = preparer.get_optimization_inputs('ledoit_wolf')

# 3. Optimize portfolio
optimizer = MultiObjectiveQUBOOptimizer(mu, sigma)
pareto_df = optimizer.explore_pareto_frontier(n_points=20)

# Find maximum Sharpe ratio portfolio
max_sharpe_idx = pareto_df['sharpe'].idxmax()
optimal_weights = pareto_df.loc[max_sharpe_idx, 'weights']

# 4. Backtest with rebalancing
config = RebalancingConfig(method='threshold', threshold=0.05)
rebalancer = PortfolioRebalancer(config)
performance = rebalancer.simulate_rebalancing(
    stats['returns'], optimal_weights, initial_value=100000
)

print(f"Optimal Portfolio - Sharpe Ratio: {pareto_df.loc[max_sharpe_idx, 'sharpe']:.3f}")
print(f"Final Portfolio Value: ${performance['portfolio_value'].iloc[-1]:,.2f}")
```

## Key Enhancements in Phase 2

1. **Multi-objective optimization** balances competing goals
2. **Pareto frontier exploration** provides multiple optimal solutions
3. **Dynamic rebalancing** adapts to market conditions
4. **Transaction cost optimization** improves net returns
5. **Comprehensive backtesting** validates strategies

## Next: Phase 3

Phase 3 will add:
- **Quantum hardware integration** (D-Wave, IBM Qiskit)
- **Advanced risk models** (CVaR, downside risk)
- **Machine learning for return forecasting**
- **Real-time portfolio monitoring dashboard**