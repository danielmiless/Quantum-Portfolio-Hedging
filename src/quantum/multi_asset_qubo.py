# Place in: src/quantum/multi_asset_qubo.py

import numpy as np
from typing import Dict, List, Tuple, Optional


class MultiAssetQUBOBuilder:
    """
    Build QUBO matrices for multi-asset class portfolios.
    """
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray,
                 asset_classes: Dict[int, str],
                 liquidity_scores: Optional[np.ndarray] = None):
        """
        Args:
            mu: Expected returns
            sigma: Covariance matrix
            asset_classes: Mapping from asset index to class name
            liquidity_scores: Liquidity score for each asset (0-1)
        """
        self.mu = mu
        self.sigma = sigma
        self.asset_classes = asset_classes
        self.n_assets = len(mu)
        
        if liquidity_scores is None:
            self.liquidity_scores = np.ones(self.n_assets)
        else:
            self.liquidity_scores = liquidity_scores
    
    def build_qubo_with_class_constraints(self,
                                         class_limits: Dict[str, Tuple[float, float]],
                                         risk_aversion: float = 1.0,
                                         n_bits: int = 4) -> Tuple[np.ndarray, float]:
        """
        Build QUBO with asset class allocation constraints.
        
        Args:
            class_limits: Dict mapping class name to (min_weight, max_weight)
            risk_aversion: Risk aversion parameter
            n_bits: Bits per asset for discretization
            
        Returns:
            Q: QUBO matrix
            offset: Constant offset
        """
        n_vars = self.n_assets * n_bits
        Q = np.zeros((n_vars, n_vars))
        offset = 0.0
        max_val = 2**n_bits - 1
        
        # 1. Return objective (maximize)
        for i in range(self.n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                bit_value = 2**k / max_val
                Q[idx, idx] -= self.mu[i] * bit_value
        
        # 2. Risk objective (minimize)
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                for k1 in range(n_bits):
                    for k2 in range(n_bits):
                        idx1 = i * n_bits + k1
                        idx2 = j * n_bits + k2
                        bit_val1 = 2**k1 / max_val
                        bit_val2 = 2**k2 / max_val
                        
                        if idx1 <= idx2:
                            Q[idx1, idx2] += risk_aversion * self.sigma[i, j] * bit_val1 * bit_val2
        
        # 3. Asset class constraints
        penalty_strength = 10.0
        
        for class_name, (min_weight, max_weight) in class_limits.items():
            # Find assets in this class
            class_assets = [i for i, c in self.asset_classes.items() if c == class_name]
            
            if not class_assets:
                continue
            
            # Minimum constraint: (sum w_i - min_weight)^2 if sum < min
            # Maximum constraint: (sum w_i - max_weight)^2 if sum > max
            
            # Add penalty terms for constraint violations
            for i in class_assets:
                for k in range(n_bits):
                    idx = i * n_bits + k
                    bit_value = 2**k / max_val
                    
                    # Simplified: penalize deviation from midpoint
                    target = (min_weight + max_weight) / 2
                    Q[idx, idx] += penalty_strength * (bit_value - target / len(class_assets))**2
        
        # 4. Budget constraint
        budget_penalty = 10.0
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
        
        for i in range(self.n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                bit_value = 2**k / max_val
                Q[idx, idx] -= 2 * budget_penalty * bit_value
        
        offset += budget_penalty
        
        return Q, offset
    
    def add_liquidity_penalty(self, Q: np.ndarray, 
                             max_illiquidity: float = 0.3,
                             n_bits: int = 4) -> np.ndarray:
        """
        Add liquidity constraints to QUBO.
        
        Args:
            Q: Existing QUBO matrix
            max_illiquidity: Maximum allowed illiquidity score
            n_bits: Bits per asset
            
        Returns:
            Updated QUBO matrix
        """
        max_val = 2**n_bits - 1
        penalty = 5.0
        
        for i in range(self.n_assets):
            illiquidity = 1.0 - self.liquidity_scores[i]
            
            if illiquidity > max_illiquidity:
                # Penalize allocations to illiquid assets
                for k in range(n_bits):
                    idx = i * n_bits + k
                    bit_value = 2**k / max_val
                    Q[idx, idx] += penalty * illiquidity * bit_value
        
        return Q


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n_assets = 10
    
    # Asset classes: 4 equities, 3 bonds, 2 commodities, 1 crypto
    asset_classes = {
        0: 'Equity', 1: 'Equity', 2: 'Equity', 3: 'Equity',
        4: 'Bond', 5: 'Bond', 6: 'Bond',
        7: 'Commodity', 8: 'Commodity',
        9: 'Crypto'
    }
    
    # Generate synthetic data
    mu = np.random.normal(0.08, 0.04, n_assets)
    A = np.random.randn(n_assets, n_assets)
    sigma = 0.01 * (A @ A.T + np.eye(n_assets))
    
    # Liquidity scores (lower for crypto and commodities)
    liquidity = np.array([0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.6, 0.6, 0.4])
    
    # Build QUBO
    builder = MultiAssetQUBOBuilder(mu, sigma, asset_classes, liquidity)
    
    # Class limits
    class_limits = {
        'Equity': (0.30, 0.60),
        'Bond': (0.20, 0.40),
        'Commodity': (0.05, 0.20),
        'Crypto': (0.00, 0.10)
    }
    
    Q, offset = builder.build_qubo_with_class_constraints(class_limits, risk_aversion=1.0)
    
    # Add liquidity constraints
    Q = builder.add_liquidity_penalty(Q, max_illiquidity=0.5)
    
    print(f"Multi-Asset QUBO Matrix Shape: {Q.shape}")
    print(f"Offset: {offset:.4f}")
    print(f"\nQUBO ready for quantum optimization!")
