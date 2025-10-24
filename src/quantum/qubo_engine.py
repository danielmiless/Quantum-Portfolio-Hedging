# qubo_engine.py
"""
Advanced QUBO formulation for portfolio optimization with multiple constraints
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
import itertools

class PortfolioQUBO:
    """
    Advanced QUBO formulation with multiple constraints and optimization objectives.
    """
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, 
                 tickers: List[str], num_weight_levels: int = 5,
                 max_weight_per_asset: float = 0.4):
        """
        Initialize QUBO formulation.
        
        Args:
            mu: Expected returns vector (n,)
            sigma: Covariance matrix (n, n)
            tickers: Asset names for debugging
            num_weight_levels: Discrete weight levels per asset
            max_weight_per_asset: Maximum allocation per single asset
        """
        self.mu = mu
        self.sigma = sigma
        self.tickers = tickers
        self.n_assets = len(mu)
        self.num_weight_levels = num_weight_levels
        self.max_weight = max_weight_per_asset
        
        # Create weight levels (including 0)
        self.weight_levels = np.linspace(0, max_weight_per_asset, num_weight_levels)
        
        # Problem formulations
        self.qp = None
        self.qubo = None
        self.Q_matrix = None
        
        print(f"Initialized QUBO formulation")
        print(f" Assets: {self.n_assets}")
        print(f" Weight levels: {num_weight_levels}")
        print(f" Max weight per asset: {max_weight_per_asset}")
        print(f" Total variables: {self.n_assets * num_weight_levels}")
    
    def create_multi_objective_problem(self, 
                                     target_return: Optional[float] = None,
                                     risk_penalty: float = 1.0,
                                     budget_penalty: float = 100.0,
                                     return_penalty: float = 50.0,
                                     cardinality_penalty: float = 10.0,
                                     concentration_penalty: float = 5.0,
                                     max_assets: Optional[int] = None) -> QuadraticProgram:
        """
        Create comprehensive quadratic program with multiple objectives and constraints.
        
        Args:
            target_return: Target portfolio return (if None, maximize return)
            risk_penalty: Weight for risk minimization
            budget_penalty: Penalty for budget constraint violation
            return_penalty: Penalty for return target violation
            cardinality_penalty: Penalty for exceeding maximum number of assets
            concentration_penalty: Penalty for excessive concentration
            max_assets: Maximum number of assets to select
        """
        
        qp = QuadraticProgram("advanced_portfolio_optimization")
        
        # Create binary variables: x_{i,j} for asset i, weight level j
        var_names = []
        for i in range(self.n_assets):
            for j in range(self.num_weight_levels):
                var_name = f"x_{i}_{j}"
                qp.binary_var(var_name)
                var_names.append((i, j))
        
        print(f"Building multi-objective QUBO with {len(var_names)} variables...")
        
        # 1. Risk minimization objective
        self._add_risk_objective(qp, var_names, risk_penalty)
        print(f"   Added risk minimization (penalty: {risk_penalty})")
        
        # 2. Return objective
        if target_return is None:
            # Maximize expected return
            self._add_return_maximization(qp, var_names)
            print(f"   Added return maximization")
        else:
            # Target return constraint
            self._add_return_constraint_penalty(qp, var_names, target_return, return_penalty)
            print(f"   Added return target constraint (target: {target_return:.3f})")
        
        # 3. Budget constraint (weights sum to 1)
        self._add_budget_constraint_penalty(qp, var_names, budget_penalty)
        print(f"   Added budget constraint (penalty: {budget_penalty})")
        
        # 4. Cardinality constraints (each asset has at most one weight level)
        self._add_cardinality_constraints(qp, var_names, cardinality_penalty)
        print(f"   Added cardinality constraints (penalty: {cardinality_penalty})")
        
        # 5. Maximum assets constraint
        if max_assets is not None:
            self._add_max_assets_constraint(qp, var_names, max_assets, cardinality_penalty)
            print(f"   Added max assets constraint (max: {max_assets})")
        
        # 6. Concentration penalty (prevent excessive single-asset allocation)
        self._add_concentration_penalty(qp, var_names, concentration_penalty)
        print(f"   Added concentration penalty (penalty: {concentration_penalty})")
        
        self.qp = qp
        print(f"Multi-objective QUBO created successfully")
        
        return qp
    
    def _add_risk_objective(self, qp: QuadraticProgram, var_names: List[Tuple], penalty: float):
        """Add quadratic risk minimization term."""
        linear_terms = {}
        quadratic_terms = {}
        
        for idx1, (i, j) in enumerate(var_names):
            for idx2, (k, l) in enumerate(var_names):
                coeff = penalty * self.sigma[i, k] * self.weight_levels[j] * self.weight_levels[l]
                
                if abs(coeff) > 1e-10:  # Numerical threshold
                    var1, var2 = f"x_{i}_{j}", f"x_{k}_{l}"
                    
                    if idx1 == idx2:  # Diagonal terms
                        linear_terms[var1] = linear_terms.get(var1, 0) + coeff
                    else:  # Off-diagonal terms
                        if idx1 < idx2:  # Avoid duplicates
                            key = (var1, var2)
                            quadratic_terms[key] = quadratic_terms.get(key, 0) + coeff
        
        if linear_terms:
            qp.minimize(linear=linear_terms)
        if quadratic_terms:
            qp.minimize(quadratic=quadratic_terms)
    
    def _add_return_maximization(self, qp: QuadraticProgram, var_names: List[Tuple]):
        """Add return maximization objective (negative for maximization)."""
        linear_terms = {}
        
        for i, j in var_names:
            coeff = -self.mu[i] * self.weight_levels[j]  # Negative for maximization
            if abs(coeff) > 1e-10:
                var_name = f"x_{i}_{j}"
                linear_terms[var_name] = linear_terms.get(var_name, 0) + coeff
        
        if linear_terms:
            qp.minimize(linear=linear_terms)
    
    def _add_return_constraint_penalty(self, qp: QuadraticProgram, var_names: List[Tuple],
                                     target_return: float, penalty: float):
        """Add penalty for deviating from target return."""
        # (Σ μᵢwᵢ - target)²
        linear_terms = {}
        quadratic_terms = {}
        
        # Linear terms: -2 * target * Σ μᵢwᵢ
        for i, j in var_names:
            coeff = -2 * penalty * target_return * self.mu[i] * self.weight_levels[j]
            if abs(coeff) > 1e-10:
                var_name = f"x_{i}_{j}"
                linear_terms[var_name] = linear_terms.get(var_name, 0) + coeff
        
        # Quadratic terms: (Σ μᵢwᵢ)²
        for idx1, (i, j) in enumerate(var_names):
            for idx2, (k, l) in enumerate(var_names):
                coeff = penalty * self.mu[i] * self.mu[k] * self.weight_levels[j] * self.weight_levels[l]
                
                if abs(coeff) > 1e-10:
                    if idx1 == idx2:
                        var_name = f"x_{i}_{j}"
                        linear_terms[var_name] = linear_terms.get(var_name, 0) + coeff
                    elif idx1 < idx2:
                        var1, var2 = f"x_{i}_{j}", f"x_{k}_{l}"
                        key = (var1, var2)
                        quadratic_terms[key] = quadratic_terms.get(key, 0) + coeff
        
        # Constant term: penalty * target²
        qp.minimize(constant=penalty * target_return**2)
        
        if linear_terms:
            qp.minimize(linear=linear_terms)
        if quadratic_terms:
            qp.minimize(quadratic=quadratic_terms)
    
    def _add_budget_constraint_penalty(self, qp: QuadraticProgram, var_names: List[Tuple], penalty: float):
        """Add penalty for budget constraint violation: (Σwᵢ - 1)²"""
        linear_terms = {}
        quadratic_terms = {}
        
        # Linear terms: -2 * Σwᵢ
        for i, j in var_names:
            coeff = -2 * penalty * self.weight_levels[j]
            if abs(coeff) > 1e-10:
                var_name = f"x_{i}_{j}"
                linear_terms[var_name] = linear_terms.get(var_name, 0) + coeff
        
        # Quadratic terms: (Σwᵢ)²
        for idx1, (i, j) in enumerate(var_names):
            for idx2, (k, l) in enumerate(var_names):
                coeff = penalty * self.weight_levels[j] * self.weight_levels[l]
                
                if abs(coeff) > 1e-10:
                    if idx1 == idx2:
                        var_name = f"x_{i}_{j}"
                        linear_terms[var_name] = linear_terms.get(var_name, 0) + coeff
                    elif idx1 < idx2:
                        var1, var2 = f"x_{i}_{j}", f"x_{k}_{l}"
                        key = (var1, var2)
                        quadratic_terms[key] = quadratic_terms.get(key, 0) + coeff
        
        # Constant term: penalty * 1²
        qp.minimize(constant=penalty)
        
        if linear_terms:
            qp.minimize(linear=linear_terms)
        if quadratic_terms:
            qp.minimize(quadratic=quadratic_terms)
    
    def _add_cardinality_constraints(self, qp: QuadraticProgram, var_names: List[Tuple], penalty: float):
        """Ensure each asset has exactly one weight level selected."""
        for i in range(self.n_assets):
            # Penalty for (Σⱼ xᵢⱼ - 1)²
            asset_vars = [f"x_{i}_{j}" for j in range(self.num_weight_levels)]
            
            # Linear constraint approach (hard constraint)
            qp.linear_constraint(
                linear={var: 1 for var in asset_vars},
                sense="==",
                rhs=1.0,
                name=f"cardinality_{i}_{self.tickers[i] if i < len(self.tickers) else i}"
            )
    
    def _add_max_assets_constraint(self, qp: QuadraticProgram, var_names: List[Tuple],
                                 max_assets: int, penalty: float):
        """Add penalty for selecting more than max_assets."""
        # Count non-zero weight selections: Σᵢ max(xᵢ₁, xᵢ₂, ..., xᵢⱼ)
        # Approximated as: Σᵢ Σⱼ≠₀ xᵢⱼ ≤ max_assets
        
        selection_vars = []
        for i in range(self.n_assets):
            for j in range(1, self.num_weight_levels):  # Skip j=0 (zero weight)
                selection_vars.append(f"x_{i}_{j}")
        
        # Add as linear constraint
        qp.linear_constraint(
            linear={var: 1 for var in selection_vars},
            sense="<=",
            rhs=max_assets,
            name=f"max_assets_{max_assets}"
        )
    
    def _add_concentration_penalty(self, qp: QuadraticProgram, var_names: List[Tuple], penalty: float):
        """Add penalty for excessive concentration in any single asset."""
        # Penalty increases quadratically with individual asset weights
        linear_terms = {}
        
        for i, j in var_names:
            if j > 0:  # Only for non-zero weights
                # Quadratic penalty: penalty * w²
                coeff = penalty * (self.weight_levels[j] ** 2)
                if abs(coeff) > 1e-10:
                    var_name = f"x_{i}_{j}"
                    linear_terms[var_name] = linear_terms.get(var_name, 0) + coeff
        
        if linear_terms:
            qp.minimize(linear=linear_terms)
    
    def convert_to_qubo(self) -> Tuple[np.ndarray, float]:
        """Convert quadratic program to QUBO matrix format."""
        if self.qp is None:
            raise ValueError("Must create quadratic program first")
        
        print("Converting to QUBO format...")
        
        # Convert constraints to penalties if needed
        converter = QuadraticProgramToQubo()
        self.qubo = converter.convert(self.qp)
        
        # Extract QUBO matrix
        n_vars = self.qubo.get_num_vars()
        Q = np.zeros((n_vars, n_vars))
        
        # Add quadratic terms
        quadratic_dict = self.qubo.objective.quadratic.to_dict()
        for (i, j), coeff in quadratic_dict.items():
            Q[i, j] += coeff
            if i != j:  # Ensure symmetry for off-diagonal terms
                Q[j, i] += coeff
        
        # Add linear terms to diagonal
        linear_dict = self.qubo.objective.linear.to_dict()
        for i, coeff in linear_dict.items():
            Q[i, i] += coeff
        
        constant = self.qubo.objective.constant
        
        self.Q_matrix = Q
        
        print(f"QUBO conversion complete")
        print(f" Matrix size: {Q.shape}")
        print(f" Non-zero elements: {np.count_nonzero(Q)}")
        print(f" Constant term: {constant}")
        
        return Q, constant
    
    def decode_solution(self, solution: np.ndarray) -> Dict:
        """
        Convert binary solution back to portfolio weights and analyze.
        
        Args:
            solution: Binary solution vector
            
        Returns:
            Dictionary with decoded portfolio information
        """
        weights = np.zeros(self.n_assets)
        selections = {}
        
        # Decode binary variables
        var_idx = 0
        for i in range(self.n_assets):
            asset_selection = []
            for j in range(self.num_weight_levels):
                if var_idx < len(solution) and solution[var_idx] == 1:
                    weights[i] = self.weight_levels[j]
                    selections[self.tickers[i] if i < len(self.tickers) else f"Asset_{i}"] = j
                    asset_selection.append(j)
                var_idx += 1
            
            # Validation: each asset should have exactly one selection
            if len(asset_selection) != 1:
                print(f"Warning: Asset {i} has {len(asset_selection)} selections: {asset_selection}")
        
        # Normalize to ensure budget constraint (if needed)
        total_weight = np.sum(weights)
        if total_weight > 0:
            normalized_weights = weights / total_weight
        else:
            normalized_weights = weights
        
        # Calculate portfolio metrics
        expected_return = np.dot(normalized_weights, self.mu)
        portfolio_risk = np.sqrt(np.dot(normalized_weights, np.dot(self.sigma, normalized_weights)))
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Count selected assets
        n_selected = np.sum(normalized_weights > 1e-6)
        
        # Calculate concentration (Herfindahl index)
        concentration = np.sum(normalized_weights ** 2)
        
        return {
            'weights': normalized_weights,
            'raw_weights': weights,
            'total_weight': total_weight,
            'selections': selections,
            'expected_return': expected_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'n_assets_selected': n_selected,
            'concentration_index': concentration,
            'max_weight': np.max(normalized_weights),
            'portfolio_summary': self._create_portfolio_summary(normalized_weights)
        }
    
    def _create_portfolio_summary(self, weights: np.ndarray) -> pd.DataFrame:
        """Create a readable portfolio summary."""
        import pandas as pd
        
        portfolio_data = []
        for i, weight in enumerate(weights):
            if weight > 1e-6:  # Only include significant positions
                portfolio_data.append({
                    'Asset': self.tickers[i] if i < len(self.tickers) else f"Asset_{i}",
                    'Weight': weight,
                    'Weight_Pct': weight * 100,
                    'Expected_Return': self.mu[i],
                    'Volatility': np.sqrt(self.sigma[i, i]),
                    'Sharpe': self.mu[i] / np.sqrt(self.sigma[i, i])
                })
        
        df = pd.DataFrame(portfolio_data)
        if not df.empty:
            df = df.sort_values('Weight', ascending=False)
        
        return df
    
    def analyze_qubo_structure(self) -> Dict:
        """Analyze QUBO matrix properties for debugging and optimization."""
        if self.Q_matrix is None:
            raise ValueError("Must convert to QUBO first")
        
        Q = self.Q_matrix
        
        # Matrix properties
        eigenvals = np.linalg.eigvals(Q)
        condition_number = np.max(eigenvals) / np.min(eigenvals) if np.min(eigenvals) != 0 else np.inf
        
        # Sparsity analysis
        total_elements = Q.size
        nonzero_elements = np.count_nonzero(Q)
        sparsity = 1 - (nonzero_elements / total_elements)
        
        # Value distribution
        nonzero_values = Q[Q != 0]
        
        analysis = {
            'matrix_size': Q.shape,
            'eigenvalues': {
                'min': np.min(eigenvals),
                'max': np.max(eigenvals),
                'condition_number': condition_number
            },
            'sparsity': {
                'total_elements': total_elements,
                'nonzero_elements': nonzero_elements,
                'sparsity_ratio': sparsity
            },
            'value_distribution': {
                'min_value': np.min(nonzero_values) if len(nonzero_values) > 0 else 0,
                'max_value': np.max(nonzero_values) if len(nonzero_values) > 0 else 0,
                'mean_abs_value': np.mean(np.abs(nonzero_values)) if len(nonzero_values) > 0 else 0
            }
        }
        
        print(f"QUBO Matrix Analysis:")
        print(f" Size: {analysis['matrix_size']}")
        print(f" Condition number: {analysis['eigenvalues']['condition_number']:.2e}")
        print(f" Sparsity: {analysis['sparsity']['sparsity_ratio']:.3f}")
        print(f" Value range: [{analysis['value_distribution']['min_value']:.2e}, {analysis['value_distribution']['max_value']:.2e}]")
        
        return analysis

# Example usage and testing
if __name__ == "__main__":
    # Test with sample financial data
    np.random.seed(42)
    n_assets = 5
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate realistic returns and covariance
    mu = np.array([0.08, 0.12, 0.10, 0.15, 0.18])  # 8-18% expected returns
    
    # Create realistic covariance matrix
    volatilities = np.array([0.20, 0.25, 0.18, 0.30, 0.35])
    correlation = np.array([
        [1.00, 0.40, 0.50, 0.30, 0.20],
        [0.40, 1.00, 0.35, 0.45, 0.25],
        [0.50, 0.35, 1.00, 0.25, 0.15],
        [0.30, 0.45, 0.25, 1.00, 0.40],
        [0.20, 0.25, 0.15, 0.40, 1.00]
    ])
    sigma = np.outer(volatilities, volatilities) * correlation
    
    print("Testing Advanced QUBO Formulation...")
    print(f"Assets: {tickers}")
    print(f"Expected returns: {mu}")
    print(f"Volatilities: {volatilities}")
    
    # Create QUBO formulation
    qubo_engine = PortfolioQUBO(
        mu=mu, 
        sigma=sigma, 
        tickers=tickers,
        num_weight_levels=4,
        max_weight_per_asset=0.4
    )
    
    # Create multi-objective problem
    qp = qubo_engine.create_multi_objective_problem(
        target_return=0.12,
        risk_penalty=1.0,
        budget_penalty=100.0,
        return_penalty=50.0,
        cardinality_penalty=10.0,
        max_assets=3
    )
    
    print(f"Created quadratic program with {qp.get_num_vars()} variables and {qp.get_num_linear_constraints()} constraints")
    
    # Convert to QUBO
    Q, constant = qubo_engine.convert_to_qubo()
    
    # Analyze QUBO structure
    analysis = qubo_engine.analyze_qubo_structure()
    
    # Test solution decoding with random solution
    n_vars = Q.shape[0]
    random_solution = np.random.choice([0, 1], size=n_vars)
    decoded = qubo_engine.decode_solution(random_solution)
    
    print(f"\nTest Solution Analysis:")
    print(f" Expected return: {decoded['expected_return']:.4f}")
    print(f" Volatility: {decoded['volatility']:.4f}")
    print(f" Sharpe ratio: {decoded['sharpe_ratio']:.4f}")
    print(f" Assets selected: {decoded['n_assets_selected']}")
    print(f" Concentration index: {decoded['concentration_index']:.4f}")
    
    if not decoded['portfolio_summary'].empty:
        print(f"\n   Portfolio composition:")
        print(decoded['portfolio_summary'].to_string(index=False))
    
    print("\nQUBO formulation testing completed successfully!")
