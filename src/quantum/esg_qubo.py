# Place in: src/quantum/esg_qubo.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from quantum.multi_asset_qubo import MultiAssetQUBOBuilder


class ESGQUBOBuilder(MultiAssetQUBOBuilder):
    """
    Extended QUBO builder with ESG and sustainability constraints.
    """
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray,
                 esg_scores: np.ndarray, carbon_intensities: np.ndarray,
                 asset_classes: Optional[Dict[int, str]] = None,
                 liquidity_scores: Optional[np.ndarray] = None):
        """
        Args:
            mu: Expected returns
            sigma: Covariance matrix
            esg_scores: ESG scores for each asset (0-100)
            carbon_intensities: Carbon intensity for each asset
            asset_classes: Asset class mapping
            liquidity_scores: Liquidity scores
        """
        super().__init__(mu, sigma, asset_classes or {}, liquidity_scores)
        self.esg_scores = esg_scores
        self.carbon_intensities = carbon_intensities
        
    def build_esg_qubo(self, esg_weight: float = 0.5,
                      carbon_penalty: float = 0.3,
                      min_esg_score: Optional[float] = None,
                      max_carbon_intensity: Optional[float] = None,
                      n_bits: int = 4) -> Tuple[np.ndarray, float]:
        """
        Build QUBO with ESG objectives and constraints.
        
        Args:
            esg_weight: Weight for ESG maximization objective
            carbon_penalty: Penalty for high carbon intensity
            min_esg_score: Minimum portfolio ESG score constraint
            max_carbon_intensity: Maximum portfolio carbon intensity
            n_bits: Bits per asset for discretization
            
        Returns:
            Q: ESG-enhanced QUBO matrix
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
        risk_weight = 1.0
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
        
        # 3. ESG objective (maximize ESG scores)
        normalized_esg = self.esg_scores / 100.0  # Normalize to 0-1
        for i in range(self.n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                bit_value = 2**k / max_val
                # Negative because we're minimizing (QUBO form)
                Q[idx, idx] -= esg_weight * normalized_esg[i] * bit_value
        
        # 4. Carbon penalty (minimize carbon footprint)
        normalized_carbon = self.carbon_intensities / np.max(self.carbon_intensities)
        for i in range(self.n_assets):
            for k in range(n_bits):
                idx = i * n_bits + k
                bit_value = 2**k / max_val
                Q[idx, idx] += carbon_penalty * normalized_carbon[i] * bit_value
        
        # 5. ESG constraint (if specified)
        if min_esg_score is not None:
            constraint_penalty = 10.0
            target_esg = min_esg_score / 100.0  # Normalize
            
            # Penalty for not meeting ESG target: (sum w_i * esg_i - target)^2
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    for k1 in range(n_bits):
                        for k2 in range(n_bits):
                            idx1 = i * n_bits + k1
                            idx2 = j * n_bits + k2
                            bit_val1 = 2**k1 / max_val
                            bit_val2 = 2**k2 / max_val
                            
                            if idx1 <= idx2:
                                Q[idx1, idx2] += constraint_penalty * normalized_esg[i] * normalized_esg[j] * bit_val1 * bit_val2
            
            # Linear term
            for i in range(self.n_assets):
                for k in range(n_bits):
                    idx = i * n_bits + k
                    bit_value = 2**k / max_val
                    Q[idx, idx] -= 2 * constraint_penalty * target_esg * normalized_esg[i] * bit_value
            
            offset += constraint_penalty * target_esg**2
        
        # 6. Carbon constraint (if specified)
        if max_carbon_intensity is not None:
            constraint_penalty = 10.0
            max_carbon_norm = max_carbon_intensity / np.max(self.carbon_intensities)
            
            # Similar penalty structure for carbon constraint
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    for k1 in range(n_bits):
                        for k2 in range(n_bits):
                            idx1 = i * n_bits + k1
                            idx2 = j * n_bits + k2
                            bit_val1 = 2**k1 / max_val
                            bit_val2 = 2**k2 / max_val
                            
                            if idx1 <= idx2:
                                Q[idx1, idx2] += constraint_penalty * normalized_carbon[i] * normalized_carbon[j] * bit_val1 * bit_val2
            
            for i in range(self.n_assets):
                for k in range(n_bits):
                    idx = i * n_bits + k
                    bit_value = 2**k / max_val
                    Q[idx, idx] -= 2 * constraint_penalty * max_carbon_norm * normalized_carbon[i] * bit_value
            
            offset += constraint_penalty * max_carbon_norm**2
        
        # 7. Budget constraint
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
    
    def calculate_esg_metrics(self, weights: np.ndarray) -> Dict:
        """
        Calculate ESG metrics for a given portfolio.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            ESG metrics dictionary
        """
        portfolio_esg = np.dot(weights, self.esg_scores)
        portfolio_carbon = np.dot(weights, self.carbon_intensities)
        
        # ESG component scores
        # Note: Would need individual E, S, G scores for detailed breakdown
        
        return {
            'portfolio_esg_score': portfolio_esg,
            'portfolio_carbon_intensity': portfolio_carbon,
            'high_esg_weight': np.sum(weights[self.esg_scores > 80]),
            'low_carbon_weight': np.sum(weights[self.carbon_intensities < 50]),
            'esg_efficiency': portfolio_esg / portfolio_carbon if portfolio_carbon > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    from alt_data.esg_data import ESGDataProvider
    
    # Generate test data
    np.random.seed(42)
    n_assets = 6
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'XOM']
    
    # Expected returns and covariance
    mu = np.array([0.10, 0.12, 0.08, 0.15, 0.20, 0.06])
    A = np.random.randn(n_assets, n_assets)
    sigma = 0.01 * (A @ A.T + np.eye(n_assets))
    
    # ESG data
    esg_provider = ESGDataProvider()
    esg_data = esg_provider.load_sample_esg_data(tickers)
    
    esg_scores = esg_data['esg_score'].values
    carbon_intensities = esg_data['carbon_intensity'].values
    
    print("ESG QUBO Builder Test")
    print("=" * 30)
    print(f"Assets: {tickers}")
    print(f"ESG Scores: {esg_scores}")
    print(f"Carbon Intensities: {carbon_intensities}")
    
    # Build ESG QUBO
    builder = ESGQUBOBuilder(mu, sigma, esg_scores, carbon_intensities)
    
    Q, offset = builder.build_esg_qubo(
        esg_weight=0.5,
        carbon_penalty=0.3,
        min_esg_score=70,  # Minimum 70 ESG score
        max_carbon_intensity=100  # Max 100 tCO2e/$M
    )
    
    print(f"\nESG QUBO Matrix Shape: {Q.shape}")
    print(f"Offset: {offset:.4f}")
    
    # Test with sample weights
    sample_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.15, 0.05])  # Underweight XOM (high carbon)
    esg_metrics = builder.calculate_esg_metrics(sample_weights)
    
    print(f"\nSample Portfolio ESG Metrics:")
    for metric, value in esg_metrics.items():
        print(f"  {metric}: {value:.2f}")
