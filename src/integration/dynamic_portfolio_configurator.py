# src/integration/dynamic_portfolio_configurator.py
"""
Dynamic Portfolio Configuration System
=====================================

A comprehensive tool for creating, managing, and configuring dynamic portfolios
for the Quantum Portfolio Live Trading System. Supports multiple portfolio 
universes, filtering, validation, and real-time configuration.

Author: Daniel Miles
Date: October 2025
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

# Fix imports by adding project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_dir = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Third-party imports
import pandas as pd
import numpy as np

# Optional imports for enhanced functionality
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    warnings.warn("yfinance not available. Some features will be limited.")

try:
    from integration.broker_interface import BrokerFactory
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False
    warnings.warn("Broker interface not available. Ticker validation will be limited.")


@dataclass
class StockInfo:
    """Information about a single stock."""
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    price: float
    beta: float
    dividend_yield: float
    pe_ratio: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PortfolioConfig:
    """Portfolio configuration with metadata."""
    name: str
    description: str
    tickers: List[str]
    weights: Optional[Dict[str, float]] = None
    rebalance_threshold: float = 0.05
    max_position_size: float = 0.25
    min_position_size: float = 0.01
    sector_limits: Optional[Dict[str, float]] = None
    created_at: str = None
    last_modified: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PortfolioConfig':
        return cls(**data)


class PortfolioUniverseManager:
    """Manages predefined and custom portfolio universes."""
    
    def __init__(self):
        self.universes = {
            'sp500': self._get_sp500_tickers(),
            'sp100': self._get_sp100_tickers(),
            'nasdaq100': self._get_nasdaq100_tickers(),
            'dow30': self._get_dow30_tickers(),
            'mega_cap': self._get_mega_cap_tickers(),
            'growth_tech': self._get_growth_tech_tickers(),
            'dividend_aristocrats': self._get_dividend_aristocrats(),
            'clean_energy': self._get_clean_energy_tickers(),
            'biotech': self._get_biotech_tickers(),
            'fintech': self._get_fintech_tickers()
        }
        
        # Sector mappings
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'ADBE', 'CRM', 'NFLX', 'INTC'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'DHR', 'BMY', 'ABBV', 'MRK', 'LLY'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CME'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'F', 'GM'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'KMI', 'OKE', 'PSX', 'VLO', 'MPC'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UNP', 'LMT', 'RTX', 'DE', 'UPS'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'PSA', 'WELL', 'DLR', 'EXR'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'KHC', 'TSN']
        }
    
    def get_universe(self, name: str) -> List[str]:
        """Get tickers from a predefined universe."""
        return self.universes.get(name.lower(), [])
    
    def list_universes(self) -> List[str]:
        """List all available universes."""
        return list(self.universes.keys())
    
    def get_sector_tickers(self, sector: str) -> List[str]:
        """Get tickers from a specific sector."""
        return self.sectors.get(sector, [])
    
    def list_sectors(self) -> List[str]:
        """List all available sectors."""
        return list(self.sectors.keys())
    
    def _get_sp500_tickers(self) -> List[str]:
        """S&P 500 sample tickers (top holdings)."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK', 'UNH',
            'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
            'PFE', 'AVGO', 'KO', 'LLY', 'WMT', 'TMO', 'MRK', 'COST', 'ABT', 'CSCO',
            'ACN', 'DHR', 'VZ', 'ADBE', 'TXN', 'NEE', 'NKE', 'PM', 'CRM', 'DIS',
            'WFC', 'RTX', 'ORCL', 'NFLX', 'AMD', 'QCOM', 'INTC', 'T', 'COP', 'UPS',
            'HON', 'IBM', 'GS', 'SPGI', 'LOW', 'SBUX', 'CAT', 'INTU', 'AXP', 'DE'
        ]
    
    def _get_sp100_tickers(self) -> List[str]:
        """S&P 100 tickers."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE',
            'AVGO', 'KO', 'LLY', 'WMT', 'TMO', 'MRK', 'COST', 'ABT', 'CSCO', 'ACN',
            'DHR', 'VZ', 'ADBE', 'TXN', 'NEE', 'NKE', 'PM', 'CRM', 'DIS', 'WFC',
            'RTX', 'ORCL', 'NFLX', 'AMD', 'QCOM', 'INTC', 'T', 'COP', 'UPS', 'HON'
        ]
    
    def _get_nasdaq100_tickers(self) -> List[str]:
        """NASDAQ 100 sample tickers."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'AVGO', 'ASML',
            'AZN', 'COST', 'PEP', 'NFLX', 'ADBE', 'CSCO', 'TMUS', 'AMD', 'TXN', 'QCOM',
            'CMCSA', 'HON', 'INTC', 'AMGN', 'INTU', 'ISRG', 'BKNG', 'ADI', 'AMAT', 'PYPL',
            'GILD', 'REGN', 'MU', 'MELI', 'LRCX', 'KLAC', 'MDLZ', 'SNPS', 'CDNS', 'MAR',
            'ORLY', 'CSX', 'MRVL', 'FTNT', 'ADSK', 'CRWD', 'ROP', 'NXPI', 'ABNB', 'WDAY'
        ]
    
    def _get_dow30_tickers(self) -> List[str]:
        """Dow Jones 30 tickers."""
        return [
            'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'CAT', 'MCD', 'V', 'AMGN', 'CRM',
            'AXP', 'BA', 'JPM', 'JNJ', 'PG', 'CVX', 'WMT', 'DIS', 'MRK', 'NKE',
            'KO', 'MMM', 'TRV', 'IBM', 'HON', 'CSCO', 'DOW', 'INTC', 'WBA', 'VZ'
        ]
    
    def _get_mega_cap_tickers(self) -> List[str]:
        """Mega-cap stocks (>$500B market cap)."""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B']
    
    def _get_growth_tech_tickers(self) -> List[str]:
        """High-growth technology stocks."""
        return [
            'NVDA', 'AMD', 'CRM', 'ADBE', 'NOW', 'SNOW', 'PLTR', 'U', 'NET', 'DDOG',
            'ZS', 'CRWD', 'OKTA', 'MDB', 'TEAM', 'ZM', 'DOCN', 'TWLO', 'SQ', 'SHOP',
            'RBLX', 'UNITY', 'AI', 'SMCI', 'ARM', 'MRVL', 'AVGO', 'QCOM', 'TXN', 'ADI'
        ]
    
    def _get_dividend_aristocrats(self) -> List[str]:
        """Dividend aristocrat stocks."""
        return [
            'MMM', 'ABT', 'ABBV', 'ADP', 'AFL', 'APD', 'AMCR', 'AOS', 'ATO', 'BDX',
            'BF.B', 'BEN', 'CAH', 'CAT', 'CB', 'CHD', 'CL', 'CLX', 'KO', 'CWT',
            'DOV', 'ECL', 'EMR', 'ESS', 'FRT', 'GD', 'GPC', 'HRL', 'IBM', 'ITW'
        ]
    
    def _get_clean_energy_tickers(self) -> List[str]:
        """Clean energy and ESG-focused stocks."""
        return [
            'TSLA', 'NEE', 'ENPH', 'SEDG', 'BE', 'PLUG', 'FSLR', 'SPWR', 'RUN', 'NOVA',
            'CSIQ', 'JKS', 'DQ', 'SOL', 'MAXN', 'ARRY', 'AMPS', 'SHLS', 'HASI', 'BEP',
            'NEP', 'AES', 'EIX', 'XEL', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE'
        ]
    
    def _get_biotech_tickers(self) -> List[str]:
        """Biotechnology stocks."""
        return [
            'GILD', 'AMGN', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA', 'BNTX', 'ALNY', 'BMRN',
            'INCY', 'EXEL', 'TECH', 'SGEN', 'NBIX', 'RARE', 'FOLD', 'ARWR', 'IONS', 'SRPT',
            'BLUE', 'SAGE', 'ACAD', 'PTCT', 'ALLK', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VCYT'
        ]
    
    def _get_fintech_tickers(self) -> List[str]:
        """Financial technology stocks."""
        return [
            'SQ', 'PYPL', 'MA', 'V', 'AXP', 'COIN', 'HOOD', 'AFRM', 'LC', 'UPST',
            'SOFI', 'NU', 'BILL', 'FLYW', 'TOST', 'FISV', 'FIS', 'GPN', 'JKHY', 'WU',
            'MKTX', 'ICE', 'CME', 'NDAQ', 'CBOE', 'SPGI', 'MCO', 'TRU', 'ENVA', 'TREE'
        ]


class StockInfoProvider:
    """Provides detailed stock information and validation."""
    
    def __init__(self):
        self.cache = {}
        self.broker = None
        
        # Initialize broker if available
        if BROKER_AVAILABLE:
            try:
                from config import Config
                self.broker = BrokerFactory.create_broker('alpaca')
                if not self.broker.connect():
                    self.broker = None
            except Exception:
                pass
    
    def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """Get comprehensive stock information."""
        if ticker in self.cache:
            return self.cache[ticker]
        
        try:
            if YF_AVAILABLE:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                stock_info = StockInfo(
                    ticker=ticker,
                    name=info.get('longName', ticker),
                    sector=info.get('sector', 'Unknown'),
                    industry=info.get('industry', 'Unknown'),
                    market_cap=info.get('marketCap', 0),
                    price=info.get('currentPrice', 0),
                    beta=info.get('beta', 1.0),
                    dividend_yield=info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                    pe_ratio=info.get('trailingPE', 0)
                )
                
                self.cache[ticker] = stock_info
                return stock_info
            else:
                # Fallback without yfinance
                return StockInfo(
                    ticker=ticker,
                    name=ticker,
                    sector='Unknown',
                    industry='Unknown',
                    market_cap=0,
                    price=0,
                    beta=1.0,
                    dividend_yield=0,
                    pe_ratio=0
                )
        except Exception as e:
            print(f"Warning: Could not get info for {ticker}: {e}")
            return None
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if ticker is tradeable."""
        try:
            if self.broker:
                # Try to get market data from broker
                market_data = self.broker.get_market_data(ticker)
                return market_data.get('last_price', 0) > 0
            elif YF_AVAILABLE:
                # Fallback to yfinance validation
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                return not hist.empty
            else:
                # Basic validation - just check format
                return ticker.isalpha() and len(ticker) <= 5
        except Exception:
            return False
    
    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """Validate a list of tickers."""
        valid = []
        invalid = []
        
        print(f"Validating {len(tickers)} tickers...")
        for i, ticker in enumerate(tickers):
            if self.validate_ticker(ticker):
                valid.append(ticker)
            else:
                invalid.append(ticker)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Validated {i + 1}/{len(tickers)} tickers...")
        
        return valid, invalid


class PortfolioFilter:
    """Filters stocks based on various criteria."""
    
    def __init__(self, stock_provider: StockInfoProvider):
        self.stock_provider = stock_provider
    
    def filter_by_market_cap(self, tickers: List[str], 
                           min_cap: float = 0, 
                           max_cap: float = float('inf')) -> List[str]:
        """Filter by market capitalization."""
        filtered = []
        for ticker in tickers:
            info = self.stock_provider.get_stock_info(ticker)
            if info and min_cap <= info.market_cap <= max_cap:
                filtered.append(ticker)
        return filtered
    
    def filter_by_sector(self, tickers: List[str], 
                        sectors: List[str]) -> List[str]:
        """Filter by sector."""
        filtered = []
        for ticker in tickers:
            info = self.stock_provider.get_stock_info(ticker)
            if info and info.sector in sectors:
                filtered.append(ticker)
        return filtered
    
    def filter_by_beta(self, tickers: List[str], 
                      min_beta: float = 0, 
                      max_beta: float = float('inf')) -> List[str]:
        """Filter by beta (volatility)."""
        filtered = []
        for ticker in tickers:
            info = self.stock_provider.get_stock_info(ticker)
            if info and min_beta <= info.beta <= max_beta:
                filtered.append(ticker)
        return filtered
    
    def filter_by_dividend_yield(self, tickers: List[str], 
                                min_yield: float = 0) -> List[str]:
        """Filter by minimum dividend yield."""
        filtered = []
        for ticker in tickers:
            info = self.stock_provider.get_stock_info(ticker)
            if info and info.dividend_yield >= min_yield:
                filtered.append(ticker)
        return filtered
    
    def exclude_sectors(self, tickers: List[str], 
                       exclude_sectors: List[str]) -> List[str]:
        """Exclude specific sectors."""
        filtered = []
        for ticker in tickers:
            info = self.stock_provider.get_stock_info(ticker)
            if info and info.sector not in exclude_sectors:
                filtered.append(ticker)
        return filtered


class PortfolioAnalyzer:
    """Analyzes portfolio composition and statistics."""
    
    def __init__(self, stock_provider: StockInfoProvider):
        self.stock_provider = stock_provider
    
    def analyze_portfolio(self, tickers: List[str]) -> Dict:
        """Comprehensive portfolio analysis."""
        stocks_info = []
        for ticker in tickers:
            info = self.stock_provider.get_stock_info(ticker)
            if info:
                stocks_info.append(info)
        
        if not stocks_info:
            return {}
        
        # Sector allocation
        sector_counts = {}
        total_market_cap = 0
        
        for stock in stocks_info:
            sector_counts[stock.sector] = sector_counts.get(stock.sector, 0) + 1
            total_market_cap += stock.market_cap
        
        sector_allocation = {
            sector: (count / len(stocks_info)) * 100 
            for sector, count in sector_counts.items()
        }
        
        # Calculate statistics
        betas = [s.beta for s in stocks_info if s.beta > 0]
        dividend_yields = [s.dividend_yield for s in stocks_info if s.dividend_yield > 0]
        pe_ratios = [s.pe_ratio for s in stocks_info if s.pe_ratio > 0]
        
        return {
            'total_stocks': len(stocks_info),
            'sector_allocation': sector_allocation,
            'market_cap_stats': {
                'total': total_market_cap,
                'average': total_market_cap / len(stocks_info) if stocks_info else 0,
                'median': np.median([s.market_cap for s in stocks_info])
            },
            'beta_stats': {
                'average': np.mean(betas) if betas else 0,
                'median': np.median(betas) if betas else 0,
                'min': np.min(betas) if betas else 0,
                'max': np.max(betas) if betas else 0
            },
            'dividend_stats': {
                'average_yield': np.mean(dividend_yields) if dividend_yields else 0,
                'dividend_count': len(dividend_yields),
                'dividend_percentage': (len(dividend_yields) / len(stocks_info)) * 100
            },
            'valuation_stats': {
                'average_pe': np.mean(pe_ratios) if pe_ratios else 0,
                'median_pe': np.median(pe_ratios) if pe_ratios else 0
            }
        }
    
    def print_analysis(self, analysis: Dict):
        """Print formatted analysis."""
        print(f"\n📊 Portfolio Analysis")
        print(f"{'='*50}")
        print(f"Total Stocks: {analysis['total_stocks']}")
        
        print(f"\n🏢 Sector Allocation:")
        for sector, percentage in sorted(analysis['sector_allocation'].items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"  {sector:25}: {percentage:5.1f}%")
        
        print(f"\n💰 Market Cap Statistics:")
        mc_stats = analysis['market_cap_stats']
        print(f"  Total Market Cap: ${mc_stats['total']/1e12:.2f}T")
        print(f"  Average: ${mc_stats['average']/1e9:.1f}B")
        print(f"  Median:  ${mc_stats['median']/1e9:.1f}B")
        
        print(f"\n📈 Risk Statistics:")
        beta_stats = analysis['beta_stats']
        print(f"  Average Beta: {beta_stats['average']:.2f}")
        print(f"  Beta Range:   {beta_stats['min']:.2f} - {beta_stats['max']:.2f}")
        
        div_stats = analysis['dividend_stats']
        print(f"\n💵 Dividend Statistics:")
        print(f"  Dividend Paying: {div_stats['dividend_count']} ({div_stats['dividend_percentage']:.1f}%)")
        print(f"  Average Yield:   {div_stats['average_yield']:.2f}%")


class DynamicPortfolioConfigurator:
    """Main portfolio configuration system."""
    
    def __init__(self, config_dir: str = "portfolio_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.universe_manager = PortfolioUniverseManager()
        self.stock_provider = StockInfoProvider()
        self.filter = PortfolioFilter(self.stock_provider)
        self.analyzer = PortfolioAnalyzer(self.stock_provider)
        
        self.current_portfolio = None
        
        # Create example configurations
        self._create_example_configs()
    
    def run_interactive_menu(self):
        """Run the interactive portfolio configuration menu."""
        print("\n" + "="*60)
        print("🚀 QUANTUM PORTFOLIO DYNAMIC CONFIGURATOR")
        print("="*60)
        print("Configure dynamic portfolios for live quantum trading")
        
        while True:
            try:
                self._display_main_menu()
                choice = input("\nSelect option (1-12, q to quit): ").strip().lower()
                
                if choice == 'q' or choice == 'quit':
                    print("\n👋 Goodbye!")
                    break
                
                self._handle_menu_choice(choice)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\nPress Enter to continue...")
    
    def _display_main_menu(self):
        """Display the main menu."""
        print(f"\n📋 MAIN MENU")
        print(f"{'='*40}")
        print(f"1.  📦 Browse Predefined Universes")
        print(f"2.  🔍 Create Custom Portfolio")
        print(f"3.  📂 Load Saved Configuration")
        print(f"4.  💾 Save Current Portfolio")
        print(f"5.  🔧 Filter Current Portfolio")
        print(f"6.  ✅ Validate Tickers")
        print(f"7.  📊 Analyze Portfolio")
        print(f"8.  ⚙️  Set Portfolio Parameters")
        print(f"9.  🎯 Generate Trading Config")
        print(f"10. 📁 Manage Saved Configs")
        print(f"11. 🧪 Quick Test Portfolios")
        print(f"12. ℹ️  System Information")
        
        if self.current_portfolio:
            print(f"\n🎯 Current: {self.current_portfolio.name}")
            print(f"   Tickers: {len(self.current_portfolio.tickers)}")
    
    def _handle_menu_choice(self, choice: str):
        """Handle menu selection."""
        handlers = {
            '1': self._browse_universes,
            '2': self._create_custom_portfolio,
            '3': self._load_configuration,
            '4': self._save_portfolio,
            '5': self._filter_portfolio,
            '6': self._validate_tickers,
            '7': self._analyze_current_portfolio,
            '8': self._set_parameters,
            '9': self._generate_trading_config,
            '10': self._manage_configs,
            '11': self._quick_test_portfolios,
            '12': self._system_info
        }
        
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print("❌ Invalid choice. Please try again.")
    
    def _browse_universes(self):
        """Browse predefined portfolio universes."""
        print(f"\n📦 PREDEFINED UNIVERSES")
        print(f"{'='*40}")
        
        universes = self.universe_manager.list_universes()
        for i, universe in enumerate(universes, 1):
            tickers = self.universe_manager.get_universe(universe)
            print(f"{i:2}. {universe.upper():20} ({len(tickers):3} stocks)")
        
        try:
            choice = input(f"\nSelect universe (1-{len(universes)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(universes):
                    universe_name = universes[idx]
                    tickers = self.universe_manager.get_universe(universe_name)
                    
                    self.current_portfolio = PortfolioConfig(
                        name=f"{universe_name.upper()} Portfolio",
                        description=f"Portfolio based on {universe_name.upper()} universe",
                        tickers=tickers
                    )
                    
                    print(f"\n✅ Loaded {universe_name.upper()} portfolio with {len(tickers)} stocks")
                    self._show_portfolio_preview()
        except ValueError:
            print("❌ Invalid selection")
    
    def _create_custom_portfolio(self):
        """Create a custom portfolio."""
        print(f"\n🔍 CREATE CUSTOM PORTFOLIO")
        print(f"{'='*40}")
        
        name = input("Portfolio name: ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return
        
        description = input("Description: ").strip()
        
        print(f"\nChoose creation method:")
        print(f"1. Enter tickers manually")
        print(f"2. Combine multiple universes")
        print(f"3. Select by sector")
        
        method = input("Method (1-3): ").strip()
        
        if method == '1':
            tickers = self._input_tickers_manually()
        elif method == '2':
            tickers = self._combine_universes()
        elif method == '3':
            tickers = self._select_by_sector()
        else:
            print("❌ Invalid method")
            return
        
        if tickers:
            self.current_portfolio = PortfolioConfig(
                name=name,
                description=description,
                tickers=tickers
            )
            print(f"\n✅ Created portfolio '{name}' with {len(tickers)} stocks")
            self._show_portfolio_preview()
    
    def _input_tickers_manually(self) -> List[str]:
        """Input tickers manually."""
        print("Enter tickers separated by spaces or commas:")
        print("Example: AAPL MSFT GOOGL or AAPL, MSFT, GOOGL")
        
        ticker_input = input("Tickers: ").strip().upper()
        if not ticker_input:
            return []
        
        # Parse tickers
        tickers = []
        for ticker in ticker_input.replace(',', ' ').split():
            ticker = ticker.strip()
            if ticker:
                tickers.append(ticker)
        
        return tickers
    
    def _combine_universes(self) -> List[str]:
        """Combine multiple universes."""
        universes = self.universe_manager.list_universes()
        selected_tickers = set()
        
        print("\nAvailable universes:")
        for i, universe in enumerate(universes, 1):
            tickers = self.universe_manager.get_universe(universe)
            print(f"{i:2}. {universe.upper():20} ({len(tickers):3} stocks)")
        
        selection = input("Select universes (comma-separated numbers): ").strip()
        
        try:
            for num in selection.split(','):
                idx = int(num.strip()) - 1
                if 0 <= idx < len(universes):
                    universe_tickers = self.universe_manager.get_universe(universes[idx])
                    selected_tickers.update(universe_tickers)
                    print(f"Added {universes[idx].upper()} ({len(universe_tickers)} stocks)")
        except ValueError:
            print("❌ Invalid selection format")
            return []
        
        return list(selected_tickers)
    
    def _select_by_sector(self) -> List[str]:
        """Select tickers by sector."""
        sectors = self.universe_manager.list_sectors()
        selected_tickers = set()
        
        print("\nAvailable sectors:")
        for i, sector in enumerate(sectors, 1):
            tickers = self.universe_manager.get_sector_tickers(sector)
            print(f"{i:2}. {sector:25} ({len(tickers):2} stocks)")
        
        selection = input("Select sectors (comma-separated numbers): ").strip()
        
        try:
            for num in selection.split(','):
                idx = int(num.strip()) - 1
                if 0 <= idx < len(sectors):
                    sector_tickers = self.universe_manager.get_sector_tickers(sectors[idx])
                    selected_tickers.update(sector_tickers)
                    print(f"Added {sectors[idx]} ({len(sector_tickers)} stocks)")
        except ValueError:
            print("❌ Invalid selection format")
            return []
        
        return list(selected_tickers)
    
    def _load_configuration(self):
        """Load a saved portfolio configuration."""
        configs = list(self.config_dir.glob("*.json"))
        
        if not configs:
            print("📁 No saved configurations found")
            return
        
        print(f"\n📂 SAVED CONFIGURATIONS")
        print(f"{'='*40}")
        
        for i, config_file in enumerate(configs, 1):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                name = config_data.get('name', config_file.stem)
                ticker_count = len(config_data.get('tickers', []))
                print(f"{i:2}. {name:30} ({ticker_count:3} stocks)")
            except Exception:
                print(f"{i:2}. {config_file.stem:30} (❌ corrupted)")
        
        try:
            choice = input(f"\nSelect config (1-{len(configs)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(configs):
                    self._load_config_file(configs[idx])
        except ValueError:
            print("❌ Invalid selection")
    
    def _load_config_file(self, config_path: Path):
        """Load a specific config file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.current_portfolio = PortfolioConfig.from_dict(config_data)
            print(f"\n✅ Loaded '{self.current_portfolio.name}'")
            self._show_portfolio_preview()
            
        except Exception as e:
            print(f"❌ Error loading config: {e}")
    
    def _save_portfolio(self):
        """Save the current portfolio configuration."""
        if not self.current_portfolio:
            print("❌ No current portfolio to save")
            return
        
        filename = input(f"Filename (default: {self.current_portfolio.name.lower().replace(' ', '_')}): ").strip()
        if not filename:
            filename = self.current_portfolio.name.lower().replace(' ', '_')
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.current_portfolio.to_dict(), f, indent=2)
            
            print(f"✅ Saved portfolio to {config_path}")
            
        except Exception as e:
            print(f"❌ Error saving portfolio: {e}")
    
    def _filter_portfolio(self):
        """Apply filters to current portfolio."""
        if not self.current_portfolio:
            print("❌ No current portfolio to filter")
            return
        
        print(f"\n🔧 FILTER PORTFOLIO")
        print(f"{'='*40}")
        print(f"Current portfolio: {len(self.current_portfolio.tickers)} stocks")
        
        print(f"\nFilter options:")
        print(f"1. Market cap range")
        print(f"2. Sector inclusion")
        print(f"3. Sector exclusion") 
        print(f"4. Beta range")
        print(f"5. Minimum dividend yield")
        
        filter_type = input("Select filter (1-5): ").strip()
        
        original_count = len(self.current_portfolio.tickers)
        
        if filter_type == '1':
            self._filter_by_market_cap()
        elif filter_type == '2':
            self._filter_by_sector_inclusion()
        elif filter_type == '3':
            self._filter_by_sector_exclusion()
        elif filter_type == '4':
            self._filter_by_beta_range()
        elif filter_type == '5':
            self._filter_by_dividend_yield()
        else:
            print("❌ Invalid filter option")
            return
        
        new_count = len(self.current_portfolio.tickers)
        print(f"\n✅ Filter applied: {original_count} → {new_count} stocks")
    
    def _filter_by_market_cap(self):
        """Filter by market cap."""
        print("Market cap ranges:")
        print("1. Mega-cap (>$200B)")
        print("2. Large-cap ($10B-$200B)")
        print("3. Mid-cap ($2B-$10B)")
        print("4. Custom range")
        
        choice = input("Select range (1-4): ").strip()
        
        if choice == '1':
            min_cap, max_cap = 200e9, float('inf')
        elif choice == '2':
            min_cap, max_cap = 10e9, 200e9
        elif choice == '3':
            min_cap, max_cap = 2e9, 10e9
        elif choice == '4':
            try:
                min_cap = float(input("Min market cap (billions): ")) * 1e9
                max_cap = float(input("Max market cap (billions): ")) * 1e9
            except ValueError:
                print("❌ Invalid input")
                return
        else:
            print("❌ Invalid choice")
            return
        
        filtered_tickers = self.filter.filter_by_market_cap(
            self.current_portfolio.tickers, min_cap, max_cap
        )
        self.current_portfolio.tickers = filtered_tickers
    
    def _filter_by_sector_inclusion(self):
        """Filter by sector inclusion."""
        sectors = self.universe_manager.list_sectors()
        
        print("Available sectors:")
        for i, sector in enumerate(sectors, 1):
            print(f"{i:2}. {sector}")
        
        selection = input("Select sectors to include (comma-separated numbers): ").strip()
        
        try:
            selected_sectors = []
            for num in selection.split(','):
                idx = int(num.strip()) - 1
                if 0 <= idx < len(sectors):
                    selected_sectors.append(sectors[idx])
            
            filtered_tickers = self.filter.filter_by_sector(
                self.current_portfolio.tickers, selected_sectors
            )
            self.current_portfolio.tickers = filtered_tickers
            
        except ValueError:
            print("❌ Invalid selection")
    
    def _filter_by_sector_exclusion(self):
        """Filter by sector exclusion."""
        sectors = self.universe_manager.list_sectors()
        
        print("Sectors to exclude:")
        for i, sector in enumerate(sectors, 1):
            print(f"{i:2}. {sector}")
        
        selection = input("Select sectors to exclude (comma-separated numbers): ").strip()
        
        try:
            exclude_sectors = []
            for num in selection.split(','):
                idx = int(num.strip()) - 1
                if 0 <= idx < len(sectors):
                    exclude_sectors.append(sectors[idx])
            
            filtered_tickers = self.filter.exclude_sectors(
                self.current_portfolio.tickers, exclude_sectors
            )
            self.current_portfolio.tickers = filtered_tickers
            
        except ValueError:
            print("❌ Invalid selection")
    
    def _filter_by_beta_range(self):
        """Filter by beta range."""
        try:
            min_beta = float(input("Minimum beta (0.0-2.0): "))
            max_beta = float(input("Maximum beta (0.0-2.0): "))
            
            filtered_tickers = self.filter.filter_by_beta(
                self.current_portfolio.tickers, min_beta, max_beta
            )
            self.current_portfolio.tickers = filtered_tickers
            
        except ValueError:
            print("❌ Invalid beta values")
    
    def _filter_by_dividend_yield(self):
        """Filter by dividend yield."""
        try:
            min_yield = float(input("Minimum dividend yield (%): "))
            
            filtered_tickers = self.filter.filter_by_dividend_yield(
                self.current_portfolio.tickers, min_yield
            )
            self.current_portfolio.tickers = filtered_tickers
            
        except ValueError:
            print("❌ Invalid yield value")
    
    def _validate_tickers(self):
        """Validate current portfolio tickers."""
        if not self.current_portfolio:
            print("❌ No current portfolio to validate")
            return
        
        print(f"\n✅ TICKER VALIDATION")
        print(f"{'='*40}")
        
        valid, invalid = self.stock_provider.validate_tickers(
            self.current_portfolio.tickers
        )
        
        print(f"\n📊 Validation Results:")
        print(f"  Valid tickers:   {len(valid)}")
        print(f"  Invalid tickers: {len(invalid)}")
        
        if invalid:
            print(f"\n❌ Invalid tickers:")
            for ticker in invalid:
                print(f"  {ticker}")
            
            remove = input("\nRemove invalid tickers? (y/n): ").strip().lower()
            if remove == 'y':
                self.current_portfolio.tickers = valid
                print(f"✅ Removed {len(invalid)} invalid tickers")
        else:
            print("✅ All tickers are valid!")
    
    def _analyze_current_portfolio(self):
        """Analyze the current portfolio."""
        if not self.current_portfolio:
            print("❌ No current portfolio to analyze")
            return
        
        print(f"\n📊 PORTFOLIO ANALYSIS")
        print(f"{'='*40}")
        print(f"Portfolio: {self.current_portfolio.name}")
        
        analysis = self.analyzer.analyze_portfolio(self.current_portfolio.tickers)
        self.analyzer.print_analysis(analysis)
        
        input("\nPress Enter to continue...")
    
    def _set_parameters(self):
        """Set portfolio parameters."""
        if not self.current_portfolio:
            print("❌ No current portfolio to configure")
            return
        
        print(f"\n⚙️  PORTFOLIO PARAMETERS")
        print(f"{'='*40}")
        
        try:
            print(f"Current rebalance threshold: {self.current_portfolio.rebalance_threshold:.1%}")
            new_threshold = input("New rebalance threshold (%, press Enter to keep): ").strip()
            if new_threshold:
                self.current_portfolio.rebalance_threshold = float(new_threshold) / 100
            
            print(f"Current max position size: {self.current_portfolio.max_position_size:.1%}")
            new_max = input("New max position size (%, press Enter to keep): ").strip()
            if new_max:
                self.current_portfolio.max_position_size = float(new_max) / 100
            
            print(f"Current min position size: {self.current_portfolio.min_position_size:.1%}")
            new_min = input("New min position size (%, press Enter to keep): ").strip()
            if new_min:
                self.current_portfolio.min_position_size = float(new_min) / 100
            
            print("✅ Parameters updated")
            
        except ValueError:
            print("❌ Invalid parameter values")
    
    def _generate_trading_config(self):
        """Generate configuration for live trading system."""
        if not self.current_portfolio:
            print("❌ No current portfolio to export")
            return
        
        print(f"\n🎯 GENERATE TRADING CONFIG")
        print(f"{'='*40}")
        
        # Generate Python code for the trading system
        config_code = f'''# Generated Portfolio Configuration
# Portfolio: {self.current_portfolio.name}
# Generated: {datetime.now().isoformat()}

from integration.live_trading_system import LiveTradingSystem

# Portfolio Configuration
tickers = {self.current_portfolio.tickers}

# Trading System Configuration
trading_system = LiveTradingSystem(
    broker_name='alpaca',
    tickers=tickers,
    rebalance_frequency='daily'
)

# Portfolio Parameters
trading_system.config.update({{
    'rebalance_threshold': {self.current_portfolio.rebalance_threshold},
    'max_position_size': {self.current_portfolio.max_position_size},
    'min_position_size': {self.current_portfolio.min_position_size},
    'execution_algorithm': 'VWAP',
    'risk_limit_enabled': True,
    'esg_constraints_enabled': True
}})

# Start Trading System
if __name__ == "__main__":
    print("Starting {self.current_portfolio.name}")
    print(f"Tickers: {{len(tickers)}} assets")
    
    if trading_system.start():
        try:
            import time
            while True:
                time.sleep(60)
                status = trading_system.get_system_status()
                print(f"Portfolio Value: ${{status.get('portfolio', {{}}).get('total_value', 0):,.2f}}")
        except KeyboardInterrupt:
            trading_system.stop()
    else:
        print("Failed to start trading system")
'''
        
        # Save configuration
        filename = f"trading_config_{self.current_portfolio.name.lower().replace(' ', '_')}.py"
        config_path = self.config_dir / filename
        
        with open(config_path, 'w') as f:
            f.write(config_code)
        
        print(f"✅ Generated trading configuration: {config_path}")
        print(f"\nTo use this configuration:")
        print(f"  cd 'Quantum Portfolio Hedging'")
        print(f"  python {config_path}")
    
    def _manage_configs(self):
        """Manage saved configurations."""
        print(f"\n📁 MANAGE CONFIGURATIONS")
        print(f"{'='*40}")
        
        configs = list(self.config_dir.glob("*.json"))
        
        if not configs:
            print("📁 No configurations found")
            return
        
        for i, config_file in enumerate(configs, 1):
            print(f"{i:2}. {config_file.name}")
        
        print(f"\nOptions:")
        print(f"1. Delete configuration")
        print(f"2. Rename configuration")
        print(f"3. View configuration details")
        
        action = input("Select action (1-3): ").strip()
        
        if action in ['1', '2', '3']:
            try:
                config_num = int(input(f"Select config (1-{len(configs)}): ")) - 1
                if 0 <= config_num < len(configs):
                    config_path = configs[config_num]
                    
                    if action == '1':
                        self._delete_config(config_path)
                    elif action == '2':
                        self._rename_config(config_path)
                    elif action == '3':
                        self._view_config_details(config_path)
                        
            except ValueError:
                print("❌ Invalid selection")
    
    def _delete_config(self, config_path: Path):
        """Delete a configuration."""
        confirm = input(f"Delete {config_path.name}? (y/n): ").strip().lower()
        if confirm == 'y':
            config_path.unlink()
            print(f"✅ Deleted {config_path.name}")
    
    def _rename_config(self, config_path: Path):
        """Rename a configuration."""
        new_name = input("New filename: ").strip()
        if new_name and not new_name.endswith('.json'):
            new_name += '.json'
        
        if new_name:
            new_path = config_path.parent / new_name
            config_path.rename(new_path)
            print(f"✅ Renamed to {new_name}")
    
    def _view_config_details(self, config_path: Path):
        """View configuration details."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            print(f"\n📄 Configuration Details: {config_path.name}")
            print(f"{'='*50}")
            print(f"Name: {config_data.get('name', 'N/A')}")
            print(f"Description: {config_data.get('description', 'N/A')}")
            print(f"Tickers: {len(config_data.get('tickers', []))}")
            print(f"Created: {config_data.get('created_at', 'N/A')}")
            print(f"Modified: {config_data.get('last_modified', 'N/A')}")
            
            print(f"\nTickers:")
            tickers = config_data.get('tickers', [])
            for i in range(0, len(tickers), 10):
                ticker_group = tickers[i:i+10]
                print(f"  {', '.join(ticker_group)}")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    
    def _quick_test_portfolios(self):
        """Generate quick test portfolios."""
        print(f"\n🧪 QUICK TEST PORTFOLIOS")
        print(f"{'='*40}")
        
        test_portfolios = {
            'Conservative': ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO'],
            'Growth': ['NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META'],
            'Value': ['BRK', 'JPM', 'XOM', 'CVX', 'WMT'],
            'Dividend': ['T', 'VZ', 'PM', 'MO', 'KMB'],
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'ADBE'],
            'Balanced': ['AAPL', 'JNJ', 'JPM', 'XOM', 'AMZN']
        }
        
        print("Available test portfolios:")
        portfolios = list(test_portfolios.keys())
        for i, name in enumerate(portfolios, 1):
            tickers = test_portfolios[name]
            print(f"{i}. {name:12} ({len(tickers)} stocks): {', '.join(tickers)}")
        
        try:
            choice = int(input(f"\\nSelect portfolio (1-{len(portfolios)}): ")) - 1
            if 0 <= choice < len(portfolios):
                name = portfolios[choice]
                tickers = test_portfolios[name]
                
                self.current_portfolio = PortfolioConfig(
                    name=f"{name} Test Portfolio",
                    description=f"Quick test portfolio for {name.lower()} strategy",
                    tickers=tickers
                )
                
                print(f"✅ Loaded {name} test portfolio")
                self._show_portfolio_preview()
        except ValueError:
            print("❌ Invalid selection")
    
    def _system_info(self):
        """Display system information."""
        print(f"\nℹ️  SYSTEM INFORMATION")
        print(f"{'='*40}")
        print(f"Available Features:")
        print(f"  yfinance (stock data): {'✅' if YF_AVAILABLE else '❌'}")
        print(f"  Broker integration:    {'✅' if BROKER_AVAILABLE else '❌'}")
        
        print(f"\nUniverse Statistics:")
        for universe in self.universe_manager.list_universes():
            count = len(self.universe_manager.get_universe(universe))
            print(f"  {universe.upper():15}: {count:3} stocks")
        
        print(f"\nSector Coverage:")
        for sector in self.universe_manager.list_sectors():
            count = len(self.universe_manager.get_sector_tickers(sector))
            print(f"  {sector:25}: {count:2} stocks")
        
        configs = list(self.config_dir.glob("*.json"))
        print(f"\nSaved Configurations: {len(configs)}")
        
        if self.current_portfolio:
            print(f"\nCurrent Portfolio:")
            print(f"  Name: {self.current_portfolio.name}")
            print(f"  Stocks: {len(self.current_portfolio.tickers)}")
            print(f"  Rebalance threshold: {self.current_portfolio.rebalance_threshold:.1%}")
        
        input("\nPress Enter to continue...")
    
    def _show_portfolio_preview(self):
        """Show a preview of the current portfolio."""
        if not self.current_portfolio:
            return
        
        print(f"\n📋 Portfolio Preview:")
        print(f"  Name: {self.current_portfolio.name}")
        print(f"  Tickers: {len(self.current_portfolio.tickers)}")
        
        # Show first 10 tickers
        preview_tickers = self.current_portfolio.tickers[:10]
        print(f"  Sample: {', '.join(preview_tickers)}")
        if len(self.current_portfolio.tickers) > 10:
            print(f"  ... and {len(self.current_portfolio.tickers) - 10} more")
    
    def _create_example_configs(self):
        """Create example configuration files."""
        examples = {
            'sp500_sample.json': PortfolioConfig(
                name="S&P 500 Sample",
                description="Sample of top S&P 500 stocks for testing",
                tickers=self.universe_manager.get_universe('sp500')[:20],
                rebalance_threshold=0.05,
                max_position_size=0.20
            ),
            'tech_growth.json': PortfolioConfig(
                name="Technology Growth",
                description="High-growth technology stocks",
                tickers=self.universe_manager.get_universe('growth_tech')[:15],
                rebalance_threshold=0.03,
                max_position_size=0.15
            ),
            'conservative_dividend.json': PortfolioConfig(
                name="Conservative Dividend",
                description="Conservative dividend-paying stocks",
                tickers=self.universe_manager.get_universe('dividend_aristocrats')[:15],
                rebalance_threshold=0.07,
                max_position_size=0.25
            )
        }
        
        for filename, config in examples.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)


def main():
    """Main function to run the portfolio configurator."""
    try:
        configurator = DynamicPortfolioConfigurator()
        configurator.run_interactive_menu()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
