# Place in: src/quantum/esg_portfolio_optimizer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from quantum.esg_qubo import ESGQUBOBuilder
from quantum.quantum_hardware_interface import DWaveQUBOSolver
from alt_data.esg_data import ESGDataProvider, CarbonFootprintCalculator


class ESGPortfolioOptimizer:
    """
    Complete ESG-aware quantum portfolio optimizer.
    """
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.esg_provider = ESGDataProvider()
        self.carbon_calculator = CarbonFootprintCalculator()
        
        # Data storage
        self.esg_data = None
        self.mu = None
        self.sigma = None
        
    def load_esg_data(self, mu: np.ndarray, sigma: np.ndarray) -> Dict:
        """
        Load ESG data and prepare for optimization.
        
        Args:
            mu: Expected returns
            sigma: Covariance matrix
            
        Returns:
            Combined data dictionary
        """
        self.mu = mu
        self.sigma = sigma
        
        # Load ESG scores
        self.esg_data = self.esg_provider.load_sample_esg_data(self.tickers)
        
        # Calculate ESG momentum and SDG alignment
        momentum = self.esg_provider.calculate_esg_momentum(self.esg_data)
        sdg_data = self.esg_provider.map_to_sdg(self.esg_data)
        
        # Combine all ESG data
        combined_data = pd.concat([self.esg_data, momentum, sdg_data], axis=1)
        
        return {
            'esg_data': combined_data,
            'esg_scores': self.esg_data['esg_score'].values,
            'carbon_intensities': self.esg_data['carbon_intensity'].values
        }
    
    def optimize_esg_portfolio(self, 
                              esg_weight: float = 0.5,
                              carbon_penalty: float = 0.3,
                              min_esg_score: Optional[float] = None,
                              max_carbon_intensity: Optional[float] = None,
                              use_quantum: bool = True) -> Dict:
        """
        Optimize portfolio with ESG constraints.
        
        Args:
            esg_weight: Weight for ESG objective
            carbon_penalty: Penalty for carbon intensity
            min_esg_score: Minimum portfolio ESG score
            max_carbon_intensity: Maximum carbon intensity
            use_quantum: Use quantum solver vs classical
            
        Returns:
            Optimization results
        """
        if self.esg_data is None or self.mu is None:
            raise ValueError("Must load ESG data first using load_esg_data()")
        
        esg_scores = self.esg_data['esg_score'].values
        carbon_intensities = self.esg_data['carbon_intensity'].values
        
        # Build ESG QUBO
        builder = ESGQUBOBuilder(self.mu, self.sigma, esg_scores, carbon_intensities)
        
        Q, offset = builder.build_esg_qubo(
            esg_weight=esg_weight,
            carbon_penalty=carbon_penalty,
            min_esg_score=min_esg_score,
            max_carbon_intensity=max_carbon_intensity
        )
        
        if use_quantum:
            # Quantum optimization
            solver = DWaveQUBOSolver(use_hardware=False)
            result = solver.solve(Q, num_reads=1000)
            
            # Decode solution
            n_bits = 4
            weights = self._decode_quantum_solution(result, n_bits)
            method = 'Quantum'
        else:
            # Classical optimization fallback
            weights = self._classical_esg_optimization(
                esg_scores, carbon_intensities, min_esg_score, max_carbon_intensity
            )
            method = 'Classical'
        
        # Calculate comprehensive metrics
        metrics = self._calculate_all_metrics(weights, esg_scores, carbon_intensities)
        metrics['optimization_method'] = method
        metrics['weights'] = weights
        
        return metrics
    
    def _decode_quantum_solution(self, result: Dict, n_bits: int) -> np.ndarray:
        """Decode quantum solution to portfolio weights."""
        solution = result.get('best_solution', result.get('solution', np.zeros(self.n_assets * n_bits)))
        
        weights = np.zeros(self.n_assets)
        max_val = 2**n_bits - 1
        
        for i in range(self.n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                if idx < len(solution) and solution[idx] > 0.5:
                    weights[i] += 2**k / max_val
        
        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def _classical_esg_optimization(self, esg_scores: np.ndarray, 
                                   carbon_intensities: np.ndarray,
                                   min_esg_score: Optional[float] = None,
                                   max_carbon_intensity: Optional[float] = None) -> np.ndarray:
        """Classical ESG optimization using scipy."""
        from scipy.optimize import minimize
        
        def objective(weights):
            portfolio_return = np.dot(weights, self.mu)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.sigma, weights)))
            portfolio_esg = np.dot(weights, esg_scores)
            portfolio_carbon = np.dot(weights, carbon_intensities)
            
            # Multi-objective: maximize return, minimize risk, maximize ESG, minimize carbon
            return -(portfolio_return - portfolio_risk + 0.01 * portfolio_esg - 0.001 * portfolio_carbon)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if min_esg_score is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, esg_scores) - min_esg_score
            })
        
        if max_carbon_intensity is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_carbon_intensity - np.dot(w, carbon_intensities)
            })
        
        bounds = [(0, 1) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else np.ones(self.n_assets) / self.n_assets
    
    def _calculate_all_metrics(self, weights: np.ndarray, 
                              esg_scores: np.ndarray,
                              carbon_intensities: np.ndarray) -> Dict:
        """Calculate comprehensive portfolio metrics."""
        # Financial metrics
        portfolio_return = np.dot(weights, self.mu)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.sigma, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # ESG metrics
        portfolio_esg = np.dot(weights, esg_scores)
        portfolio_carbon = np.dot(weights, carbon_intensities)
        
        # ESG breakdown
        high_esg_weight = np.sum(weights[esg_scores > 80])
        medium_esg_weight = np.sum(weights[(esg_scores >= 60) & (esg_scores <= 80)])
        low_esg_weight = np.sum(weights[esg_scores < 60])
        
        # Carbon breakdown
        low_carbon_weight = np.sum(weights[carbon_intensities < 50])
        medium_carbon_weight = np.sum(weights[(carbon_intensities >= 50) & (carbon_intensities <= 200)])
        high_carbon_weight = np.sum(weights[carbon_intensities > 200])
        
        return {
            'expected_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'esg_score': portfolio_esg,
            'carbon_intensity': portfolio_carbon,
            'esg_efficiency': portfolio_esg / portfolio_carbon if portfolio_carbon > 0 else 0,
            'high_esg_allocation': high_esg_weight,
            'medium_esg_allocation': medium_esg_weight,
            'low_esg_allocation': low_esg_weight,
            'low_carbon_allocation': low_carbon_weight,
            'medium_carbon_allocation': medium_carbon_weight,
            'high_carbon_allocation': high_carbon_weight
        }
    
    def compare_esg_strategies(self, strategies: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple ESG strategies.

        Args:
            strategies: List of strategy configurations

        Returns:
            Comparison DataFrame
        """
        results = []

        for i, strategy in enumerate(strategies):
            # Extract strategy name
            strategy_name = strategy.get('name', f'Strategy_{i+1}')

            # Filter out 'name' key for optimization
            opt_params = {k: v for k, v in strategy.items() if k != 'name'}

            # Run optimization
            result = self.optimize_esg_portfolio(**opt_params)
            result['strategy'] = strategy_name
            results.append(result)

        # Convert to DataFrame for easy comparison
        comparison_df = pd.DataFrame(results)

        # Select key columns for comparison
        key_columns = ['strategy', 'expected_return', 'portfolio_risk', 'sharpe_ratio',
                      'esg_score', 'carbon_intensity', 'esg_efficiency']

        return comparison_df[key_columns]


# Example usage
if __name__ == "__main__":
    print("ESG Portfolio Optimization Test")
    print("=" * 40)
    
    # Setup
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'XOM']
    optimizer = ESGPortfolioOptimizer(tickers)
    
    # Generate sample financial data
    np.random.seed(42)
    mu = np.array([0.10, 0.12, 0.08, 0.15, 0.20, 0.06])  # XOM lowest return
    A = np.random.randn(6, 6)
    sigma = 0.01 * (A @ A.T + np.eye(6))
    
    # Load ESG data
    esg_info = optimizer.load_esg_data(mu, sigma)
    
    print("ESG Data Loaded:")
    print(esg_info['esg_data'][['esg_score', 'carbon_intensity']])
    
    # Define ESG strategies to compare
    strategies = [
        {
            'name': 'Traditional',
            'esg_weight': 0.0,
            'carbon_penalty': 0.0,
            'use_quantum': False
        },
        {
            'name': 'ESG_Tilted',
            'esg_weight': 0.3,
            'carbon_penalty': 0.2,
            'use_quantum': True
        },
        {
            'name': 'ESG_Constrained',
            'esg_weight': 0.5,
            'carbon_penalty': 0.3,
            'min_esg_score': 70,
            'use_quantum': True
        },
        {
            'name': 'Low_Carbon',
            'esg_weight': 0.2,
            'carbon_penalty': 0.8,
            'max_carbon_intensity': 50,
            'use_quantum': True
        }
    ]
    
    # Compare strategies
    comparison = optimizer.compare_esg_strategies(strategies)
    
    print("\nESG Strategy Comparison:")
    print(comparison.round(4))
    
    # Detailed analysis of best ESG strategy
    best_strategy = optimizer.optimize_esg_portfolio(
        esg_weight=0.5,
        carbon_penalty=0.3,
        min_esg_score=75,
        max_carbon_intensity=100,
        use_quantum=True
    )
    
    print(f"\nBest ESG Portfolio Analysis:")
    print(f"Expected Return: {best_strategy['expected_return']:.2%}")
    print(f"Portfolio Risk: {best_strategy['portfolio_risk']:.2%}")
    print(f"Sharpe Ratio: {best_strategy['sharpe_ratio']:.3f}")
    print(f"ESG Score: {best_strategy['esg_score']:.1f}")
    print(f"Carbon Intensity: {best_strategy['carbon_intensity']:.1f} tCO2e/$M")
    print(f"ESG Efficiency: {best_strategy['esg_efficiency']:.2f}")
    
    print(f"\nAllocation Breakdown:")
    for i, (ticker, weight) in enumerate(zip(tickers, best_strategy['weights'])):
        if weight > 0.01:
            esg_score = esg_info['esg_scores'][i]
            carbon = esg_info['carbon_intensities'][i]
            print(f"  {ticker}: {weight:.1%} (ESG: {esg_score:.0f}, Carbon: {carbon:.1f})")
