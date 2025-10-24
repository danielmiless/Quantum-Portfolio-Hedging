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
