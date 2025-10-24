# Place in: src/quantum/currency_hedging.py

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional


class CurrencyHedger:
    """Currency hedging for international portfolios."""
    
    def __init__(self, base_currency: str = 'USD'):
        self.base_currency = base_currency
        self.fx_rates = {}
        
    def download_fx_rates(self, currencies: List[str], 
                         start_date: str, end_date: str) -> pd.DataFrame:
        """Download FX rates."""
        fx_pairs = [f"{curr}{self.base_currency}=X" for curr in currencies]
        
        fx_data = yf.download(fx_pairs, start=start_date, end=end_date, progress=False)
        
        if isinstance(fx_data.columns, pd.MultiIndex):
            prices = fx_data['Close']
        else:
            prices = fx_data['Close']
        
        prices.columns = currencies
        self.fx_rates = prices
        return prices
    
    def calculate_hedged_returns(self, asset_returns: pd.DataFrame,
                                asset_currencies: Dict[str, str],
                                hedge_ratio: float = 1.0) -> pd.DataFrame:
        """Calculate currency-hedged returns."""
        hedged_returns = asset_returns.copy()
        fx_returns = self.fx_rates.pct_change()
        
        for asset in asset_returns.columns:
            currency = asset_currencies.get(asset, self.base_currency)
            
            if currency != self.base_currency and currency in fx_returns.columns:
                common_dates = asset_returns.index.intersection(fx_returns.index)
                hedged_returns.loc[common_dates, asset] = (
                    asset_returns.loc[common_dates, asset] - 
                    hedge_ratio * fx_returns.loc[common_dates, currency]
                )
        
        return hedged_returns
