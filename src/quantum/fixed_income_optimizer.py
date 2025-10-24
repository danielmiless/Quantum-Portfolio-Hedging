# Place in: src/quantum/fixed_income_optimizer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize


class FixedIncomeOptimizer:
    """
    Fixed income portfolio optimization with duration matching.
    """
    
    def __init__(self, returns: pd.DataFrame, durations: np.ndarray, 
                 yields: np.ndarray):
        """
        Args:
            returns: Historical bond returns
            durations: Duration for each bond
            yields: Current yield for each bond
        """
        self.returns = returns
        self.durations = durations
        self.yields = yields
        self.n_bonds = len(durations)
        
    def optimize_with_duration_target(self, target_duration: float,
                                     target_yield: Optional[float] = None) -> Dict:
        """
        Optimize bond portfolio with duration matching.
        
        Args:
            target_duration: Target portfolio duration
            target_yield: Minimum target yield (optional)
            
        Returns:
            Optimization result
        """
        def objective(weights):
            # Minimize variance
            returns_cov = self.returns.cov().values
            portfolio_var = weights.T @ returns_cov @ weights
            return portfolio_var
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Budget
            {'type': 'eq', 'fun': lambda w: np.dot(w, self.durations) - target_duration}  # Duration
        ]
        
        if target_yield is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, self.yields) - target_yield
            })
        
        bounds = [(0, 1) for _ in range(self.n_bonds)]
        x0 = np.ones(self.n_bonds) / self.n_bonds
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        portfolio_duration = np.dot(optimal_weights, self.durations)
        portfolio_yield = np.dot(optimal_weights, self.yields)
        
        return {
            'weights': optimal_weights,
            'duration': portfolio_duration,
            'yield': portfolio_yield,
            'volatility': np.sqrt(result.fun) * np.sqrt(252),
            'success': result.success
        }
    
    def calculate_key_rate_durations(self, weights: np.ndarray,
                                    key_rates: List[int] = [2, 5, 10, 30]) -> Dict:
        """
        Calculate key rate durations for interest rate risk.
        
        Args:
            weights: Portfolio weights
            key_rates: Key maturities in years
            
        Returns:
            Dictionary of key rate durations
        """
        # Simplified implementation
        # In practice, requires detailed bond cash flow analysis
        
        krd = {}
        for rate in key_rates:
            # Approximate KRD based on bond durations
            rate_exposure = sum(
                w * d * np.exp(-abs(d - rate))
                for w, d in zip(weights, self.durations)
            )
            krd[f'{rate}Y'] = rate_exposure
        
        return krd


# Example usage
if __name__ == "__main__":
    # Simulate bond data
    np.random.seed(42)
    n_bonds = 10
    n_periods = 252 * 3
    
    # Bond characteristics
    durations = np.array([2, 3, 5, 7, 10, 15, 20, 25, 30, 30])
    yields = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.052, 0.055, 0.055])
    
    # Simulate returns (inverse relationship with duration)
    returns_data = np.random.normal(yields/252, 0.01/np.sqrt(252), (n_periods, n_bonds))
    returns_df = pd.DataFrame(returns_data, 
                             columns=[f'Bond_{i}' for i in range(n_bonds)])
    
    # Optimize
    optimizer = FixedIncomeOptimizer(returns_df, durations, yields)
    
    result = optimizer.optimize_with_duration_target(
        target_duration=10.0,
        target_yield=0.04
    )
    
    print("Fixed Income Optimization Results:")
    print(f"Portfolio Duration: {result['duration']:.2f} years")
    print(f"Portfolio Yield: {result['yield']:.2%}")
    print(f"Portfolio Volatility: {result['volatility']:.2%}")
    print(f"\nWeights: {result['weights']}")
    
    # Key rate durations
    krd = optimizer.calculate_key_rate_durations(result['weights'])
    print(f"\nKey Rate Durations:")
    for rate, duration in krd.items():
        print(f"  {rate}: {duration:.3f}")