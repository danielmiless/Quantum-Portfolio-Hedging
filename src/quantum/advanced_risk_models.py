# advanced_risk_models.py
"""
Advanced risk models for portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler


class CVaROptimizer:
    """
    Conditional Value at Risk portfolio optimization.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Confidence level for CVaR (e.g., 0.05 for 95% CVaR)
        """
        self.alpha = alpha
    
    def calculate_cvar(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate portfolio CVaR.
        
        Args:
            returns: Historical returns matrix (time x assets)
            weights: Portfolio weights
            
        Returns:
            CVaR value
        """
        portfolio_returns = returns @ weights
        var = np.percentile(portfolio_returns, self.alpha * 100)
        cvar = np.mean(portfolio_returns[portfolio_returns <= var])
        return cvar
    
    def optimize_cvar(self, returns: np.ndarray, 
                     target_return: Optional[float] = None) -> Dict:
        """
        Optimize portfolio to minimize CVaR.
        
        Args:
            returns: Historical returns matrix
            target_return: Minimum required return (optional)
            
        Returns:
            Optimization results
        """
        from scipy.optimize import minimize
        
        n_assets = returns.shape[1]
        
        def objective(weights):
            return -self.calculate_cvar(returns, weights)  # Negative for minimization
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Budget
        ]
        
        if target_return is not None:
            mu = np.mean(returns, axis=0)
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w: np.dot(w, mu) - target_return
            })
        
        bounds = [(0, 1) for _ in range(n_assets)]  # Long-only
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            cvar = self.calculate_cvar(returns, weights)
            portfolio_return = np.mean(returns @ weights)
            
            return {
                'weights': weights,
                'cvar': cvar,
                'return': portfolio_return,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}


class DownsideRiskModel:
    """
    Downside risk and semi-variance models.
    """
    
    def __init__(self, target_return: float = 0.0):
        """
        Args:
            target_return: Target return for downside calculation
        """
        self.target_return = target_return
    
    def calculate_downside_deviation(self, returns: np.ndarray) -> np.ndarray:
        """Calculate downside deviation for each asset."""
        downside_returns = np.minimum(returns - self.target_return, 0)
        return np.sqrt(np.mean(downside_returns**2, axis=0))
    
    def calculate_sortino_ratio(self, returns: np.ndarray) -> np.ndarray:
        """Calculate Sortino ratio for each asset."""
        mean_returns = np.mean(returns, axis=0)
        downside_dev = self.calculate_downside_deviation(returns)
        return (mean_returns - self.target_return) / downside_dev
    
    def build_downside_covariance(self, returns: np.ndarray) -> np.ndarray:
        """Build semi-covariance matrix using downside returns only."""
        downside_returns = np.minimum(returns - self.target_return, 0)
        return np.cov(downside_returns.T)


class FactorRiskModel:
    """
    Multi-factor risk model implementation.
    """
    
    def __init__(self, n_factors: int = 5):
        """
        Args:
            n_factors: Number of risk factors to extract
        """
        self.n_factors = n_factors
        self.factor_model = None
        self.scaler = StandardScaler()
        
    def fit_factor_model(self, returns: pd.DataFrame) -> Dict:
        """
        Fit factor model to return data.
        
        Args:
            returns: Return matrix (time x assets)
            
        Returns:
            Factor model results
        """
        # Standardize returns
        returns_scaled = self.scaler.fit_transform(returns)
        
        # Fit factor analysis
        self.factor_model = FactorAnalysis(n_components=self.n_factors, random_state=42)
        factors = self.factor_model.fit_transform(returns_scaled)
        
        # Calculate factor loadings (betas)
        factor_loadings = self.factor_model.components_.T  # n_assets x n_factors
        
        # Calculate specific risk (idiosyncratic variance)
        noise_variance = self.factor_model.noise_variance_
        
        # Factor covariance
        factor_cov = np.cov(factors.T)
        
        # Total covariance decomposition: Sigma = B * F * B' + Psi
        systematic_cov = factor_loadings @ factor_cov @ factor_loadings.T
        total_cov = systematic_cov + np.diag(noise_variance)
        
        return {
            'factor_loadings': factor_loadings,
            'factor_covariance': factor_cov,
            'specific_risk': noise_variance,
            'systematic_covariance': systematic_cov,
            'total_covariance': total_cov,
            'factors': factors,
            'explained_variance_ratio': self.factor_model.components_.var(axis=1) / 
                                     self.factor_model.components_.var()
        }
    
    def calculate_factor_exposure(self, weights: np.ndarray) -> np.ndarray:
        """Calculate portfolio's factor exposures."""
        if self.factor_model is None:
            raise ValueError("Factor model must be fitted first")
        
        factor_loadings = self.factor_model.components_.T
        return factor_loadings.T @ weights
    
    def calculate_risk_attribution(self, weights: np.ndarray, 
                                  factor_results: Dict) -> Dict:
        """
        Perform risk attribution analysis.
        
        Args:
            weights: Portfolio weights
            factor_results: Results from fit_factor_model
            
        Returns:
            Risk attribution breakdown
        """
        factor_loadings = factor_results['factor_loadings']
        factor_cov = factor_results['factor_covariance']
        specific_risk = factor_results['specific_risk']
        
        # Portfolio factor exposures
        factor_exposures = factor_loadings.T @ weights
        
        # Risk contributions
        systematic_risk = factor_exposures.T @ factor_cov @ factor_exposures
        specific_risk_contrib = weights.T @ np.diag(specific_risk) @ weights
        total_risk = systematic_risk + specific_risk_contrib
        
        # Factor contributions to risk
        factor_contributions = {}
        for i in range(self.n_factors):
            factor_risk = factor_exposures[i]**2 * factor_cov[i, i]
            factor_contributions[f'Factor_{i+1}'] = factor_risk / total_risk
        
        return {
            'total_risk': total_risk,
            'systematic_risk': systematic_risk,
            'specific_risk': specific_risk_contrib,
            'systematic_pct': systematic_risk / total_risk,
            'specific_pct': specific_risk_contrib / total_risk,
            'factor_contributions': factor_contributions,
            'factor_exposures': factor_exposures
        }


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_assets = 10
    n_periods = 252
    
    # Simulate returns with factor structure
    true_factors = np.random.normal(0, 0.02, (n_periods, 3))
    factor_loadings = np.random.normal(0, 1, (n_assets, 3))
    idiosyncratic = np.random.normal(0, 0.01, (n_periods, n_assets))
    
    returns = true_factors @ factor_loadings.T + idiosyncratic
    returns_df = pd.DataFrame(returns, 
                            columns=[f'Asset_{i}' for i in range(n_assets)])
    
    print("Advanced Risk Model Analysis")
    print("=" * 40)
    
    # 1. CVaR Analysis
    print("\n1. CVaR Optimization:")
    cvar_optimizer = CVaROptimizer(alpha=0.05)
    cvar_result = cvar_optimizer.optimize_cvar(returns)
    
    if cvar_result['success']:
        print(f"CVaR-optimal weights: {cvar_result['weights'][:5]}...")  # First 5
        print(f"Portfolio CVaR (95%): {cvar_result['cvar']:.4f}")
        print(f"Expected Return: {cvar_result['return']:.4f}")
    
    # 2. Downside Risk Analysis
    print("\n2. Downside Risk Analysis:")
    downside_model = DownsideRiskModel(target_return=0.0)
    downside_devs = downside_model.calculate_downside_deviation(returns)
    sortino_ratios = downside_model.calculate_sortino_ratio(returns)
    
    print(f"Average Downside Deviation: {np.mean(downside_devs):.4f}")
    print(f"Average Sortino Ratio: {np.mean(sortino_ratios):.4f}")
    
    # 3. Factor Model Analysis
    print("\n3. Factor Model Analysis:")
    factor_model = FactorRiskModel(n_factors=3)
    factor_results = factor_model.fit_factor_model(returns_df)
    
    print(f"Explained Variance by Factors: {factor_results['explained_variance_ratio']}")
    
    # Risk attribution for equal-weighted portfolio
    equal_weights = np.ones(n_assets) / n_assets
    risk_attr = factor_model.calculate_risk_attribution(equal_weights, factor_results)
    
    print(f"Systematic Risk %: {risk_attr['systematic_pct']*100:.1f}%")
    print(f"Specific Risk %: {risk_attr['specific_pct']*100:.1f}%")
    print("Factor Contributions:")
    for factor, contrib in risk_attr['factor_contributions'].items():
        print(f"  {factor}: {contrib*100:.1f}%")
