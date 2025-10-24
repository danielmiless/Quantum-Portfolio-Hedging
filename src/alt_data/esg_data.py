# src/alt_data/esg_data.py
"""
ESG data integration with environment variable support and fixed imports
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
import requests

# Fix imports by adding project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_dir = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import centralized config
try:
    from config import Config
except ImportError:
    class Config:
        MSCI_ESG_API_KEY = os.getenv('MSCI_ESG_API_KEY')
        SUSTAINALYTICS_API_KEY = os.getenv('SUSTAINALYTICS_API_KEY')
        REFINITIV_API_KEY = os.getenv('REFINITIV_API_KEY')


class ESGDataProvider:
    """
    ESG data integration from multiple sources.
    Handles data retrieval, normalization, and imputation.
    """
    
    def __init__(self, msci_api_key: Optional[str] = None,
                 sustainalytics_api_key: Optional[str] = None,
                 refinitiv_api_key: Optional[str] = None):
        """
        Initialize ESG data provider.
        
        Args:
            msci_api_key: MSCI ESG API key (uses Config.MSCI_ESG_API_KEY if None)
            sustainalytics_api_key: Sustainalytics API key
            refinitiv_api_key: Refinitiv API key
        """
        self.msci_api_key = msci_api_key or Config.MSCI_ESG_API_KEY
        self.sustainalytics_api_key = sustainalytics_api_key or Config.SUSTAINALYTICS_API_KEY
        self.refinitiv_api_key = refinitiv_api_key or Config.REFINITIV_API_KEY
        
        self.esg_scores = {}
        self.carbon_data = {}
        self.sdg_alignment = {}
        
    def load_sample_esg_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Load sample ESG data for testing.
        In production, integrate with ESG data providers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with ESG scores
        """
        # Sample ESG scores (in practice, fetch from MSCI/Sustainalytics APIs)
        esg_data = {
            'AAPL': {'esg_score': 82, 'e_score': 85, 's_score': 78, 'g_score': 84, 'carbon_intensity': 12.3},
            'GOOGL': {'esg_score': 79, 'e_score': 82, 's_score': 75, 'g_score': 80, 'carbon_intensity': 8.7},
            'MSFT': {'esg_score': 88, 'e_score': 92, 's_score': 84, 'g_score': 88, 'carbon_intensity': 5.2},
            'AMZN': {'esg_score': 65, 'e_score': 58, 's_score': 68, 'g_score': 69, 'carbon_intensity': 45.1},
            'TSLA': {'esg_score': 72, 'e_score': 89, 's_score': 65, 'g_score': 62, 'carbon_intensity': 2.1},
            'META': {'esg_score': 71, 'e_score': 75, 's_score': 68, 'g_score': 70, 'carbon_intensity': 15.8},
            'NVDA': {'esg_score': 76, 'e_score': 78, 's_score': 72, 'g_score': 78, 'carbon_intensity': 18.4},
            'JPM': {'esg_score': 68, 'e_score': 45, 's_score': 75, 'g_score': 84, 'carbon_intensity': 125.6},
            'XOM': {'esg_score': 38, 'e_score': 22, 's_score': 48, 'g_score': 45, 'carbon_intensity': 890.2},
            'JNJ': {'esg_score': 85, 'e_score': 80, 's_score': 88, 'g_score': 87, 'carbon_intensity': 28.4}
        }
        
        # Convert to DataFrame
        df_data = []
        for ticker in tickers:
            if ticker in esg_data:
                row = {'ticker': ticker}
                row.update(esg_data[ticker])
                df_data.append(row)
            else:
                # Generate synthetic data for missing tickers
                df_data.append({
                    'ticker': ticker,
                    'esg_score': np.random.normal(70, 15),
                    'e_score': np.random.normal(70, 20),
                    's_score': np.random.normal(70, 15),
                    'g_score': np.random.normal(70, 12),
                    'carbon_intensity': np.random.lognormal(3, 1)
                })
        
        df = pd.DataFrame(df_data)
        df.set_index('ticker', inplace=True)
        
        # Ensure scores are in valid range
        for col in ['esg_score', 'e_score', 's_score', 'g_score']:
            df[col] = np.clip(df[col], 0, 100)
        
        return df
    
    def integrate_msci_esg(self, tickers: List[str]) -> pd.DataFrame:
        """
        Integrate with MSCI ESG API (placeholder).
        In production, requires MSCI ESG Manager subscription.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with ESG scores
        """
        if not self.msci_api_key:
            warnings.warn("MSCI API key not found. Using sample data.")
            return self.load_sample_esg_data(tickers)
        
        # Placeholder for actual MSCI API integration
        # In production:
        # headers = {'Authorization': f'Bearer {self.msci_api_key}'}
        # response = requests.get(
        #     'https://api.msci.com/esg/v1/ratings',
        #     headers=headers,
        #     params={'tickers': ','.join(tickers)}
        # )
        # data = response.json()
        
        print("⚠️  MSCI API key found but using sample data (API not implemented)")
        return self.load_sample_esg_data(tickers)
    
    def calculate_esg_momentum(self, esg_scores: pd.DataFrame, 
                              historical_window: int = 12) -> pd.DataFrame:
        """
        Calculate ESG score momentum (improvement over time).
        
        Args:
            esg_scores: Current ESG scores
            historical_window: Lookback window in months
            
        Returns:
            DataFrame with ESG momentum
        """
        momentum_data = []
        
        for ticker in esg_scores.index:
            current_score = esg_scores.loc[ticker, 'esg_score']
            # Simulate past score with some trend
            past_score = current_score + np.random.normal(0, 5)
            momentum = current_score - past_score
            
            momentum_data.append({
                'ticker': ticker,
                'esg_momentum': momentum,
                'esg_trend': 'Improving' if momentum > 2 else 'Stable' if momentum > -2 else 'Declining'
            })
        
        momentum_df = pd.DataFrame(momentum_data)
        momentum_df.set_index('ticker', inplace=True)
        
        return momentum_df
    
    def map_to_sdg(self, esg_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Map ESG scores to UN Sustainable Development Goals alignment.
        
        Args:
            esg_scores: ESG scores DataFrame
            
        Returns:
            DataFrame with SDG alignment scores
        """
        sdg_mapping = []
        
        for ticker in esg_scores.index:
            e_score = esg_scores.loc[ticker, 'e_score']
            s_score = esg_scores.loc[ticker, 's_score']
            
            # Simple SDG alignment based on E&S scores
            sdg_climate = min(100, e_score * 1.2)  # SDG 13: Climate Action
            sdg_equality = min(100, s_score * 1.1)  # SDG 10: Reduced Inequality
            sdg_innovation = min(100, (e_score + s_score) / 2 * 1.15)  # SDG 9: Innovation
            
            sdg_mapping.append({
                'ticker': ticker,
                'sdg_climate_action': sdg_climate,
                'sdg_reduced_inequality': sdg_equality,
                'sdg_innovation': sdg_innovation,
                'avg_sdg_alignment': (sdg_climate + sdg_equality + sdg_innovation) / 3
            })
        
        sdg_df = pd.DataFrame(sdg_mapping)
        sdg_df.set_index('ticker', inplace=True)
        
        return sdg_df


class CarbonFootprintCalculator:
    """
    Carbon footprint calculation and tracking.
    """
    
    def __init__(self):
        self.carbon_data = {}
        
    def calculate_portfolio_carbon_footprint(self, weights: np.ndarray, 
                                           carbon_intensities: np.ndarray,
                                           market_caps: np.ndarray) -> Dict:
        """
        Calculate portfolio-level carbon metrics.
        
        Args:
            weights: Portfolio weights
            carbon_intensities: Carbon intensity per asset (tCO2e/$M revenue)
            market_caps: Market capitalizations
            
        Returns:
            Carbon footprint metrics
        """
        # Weighted Average Carbon Intensity (WACI)
        waci = np.sum(weights * carbon_intensities)
        
        # Total Carbon Footprint
        total_footprint = np.sum(weights * market_caps * carbon_intensities / 1e6)
        
        # Carbon efficiency (return per unit carbon)
        carbon_efficiency = 1.0 / waci if waci > 0 else 0
        
        return {
            'waci': waci,
            'total_carbon_footprint': total_footprint,
            'carbon_efficiency': carbon_efficiency,
            'low_carbon_weight': np.sum(weights[carbon_intensities < 50]),  # < 50 tCO2e/$M
            'high_carbon_weight': np.sum(weights[carbon_intensities > 200])  # > 200 tCO2e/$M
        }
    
    def carbon_risk_adjustment(self, returns: np.ndarray, 
                             carbon_intensities: np.ndarray,
                             carbon_penalty: float = 0.1) -> np.ndarray:
        """
        Adjust expected returns for carbon risk.
        
        Args:
            returns: Expected returns
            carbon_intensities: Carbon intensities
            carbon_penalty: Penalty per unit carbon intensity
            
        Returns:
            Carbon-adjusted returns
        """
        # Normalize carbon intensities
        normalized_carbon = carbon_intensities / np.max(carbon_intensities)
        
        # Apply carbon penalty
        adjusted_returns = returns - carbon_penalty * normalized_carbon
        
        return adjusted_returns


# Example usage
if __name__ == "__main__":
    print("ESG Data Provider Test")
    print("=" * 30)
    
    # Test ESG data integration
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'XOM']
    
    provider = ESGDataProvider()
    
    # Load ESG scores
    print("\n1. Loading ESG Scores:")
    esg_data = provider.load_sample_esg_data(tickers)
    print(esg_data)
    
    # Calculate ESG momentum
    print("\n2. ESG Momentum:")
    momentum = provider.calculate_esg_momentum(esg_data)
    print(momentum)
    
    # SDG alignment
    print("\n3. SDG Alignment:")
    sdg_data = provider.map_to_sdg(esg_data)
    print(sdg_data[['sdg_climate_action', 'avg_sdg_alignment']])
    
    # Carbon footprint
    print("\n4. Portfolio Carbon Footprint:")
    calculator = CarbonFootprintCalculator()
    
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    carbon_intensities = esg_data['carbon_intensity'].values
    market_caps = np.array([3000, 1800, 2800, 1500, 800, 400]) * 1e9  # Billions
    
    carbon_metrics = calculator.calculate_portfolio_carbon_footprint(
        weights, carbon_intensities, market_caps
    )
    
    print("Portfolio Carbon Metrics:")
    for metric, value in carbon_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    # Test with API key (if available)
    print("\n5. Testing MSCI Integration:")
    if Config.MSCI_ESG_API_KEY:
        print(f"  ✅ MSCI API key found: {Config.MSCI_ESG_API_KEY[:8]}...")
        esg_data_api = provider.integrate_msci_esg(tickers)
    else:
        print("  ⚠️  MSCI API key not found in .env")
        print("  Set MSCI_ESG_API_KEY to use production ESG data")
    
    print("\n✅ ESG Data Provider test completed")
