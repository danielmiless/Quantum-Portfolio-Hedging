# src/integration/run_live_trading.py
"""
Production-Ready Quantum Portfolio Live Trading System
=====================================================

Enhanced version with improved cash management, position tracking,
configurable parameters, and dynamic portfolio support.

Features:
- Smart rebalancing with time and drift controls
- Dynamic portfolio loading and switching
- Enhanced error handling and logging
- Position reconciliation and caching
- Interactive configuration menu
- Performance tracking and reporting
- Support for custom and pre-built portfolios

Author: Daniel Miles
Date: October 2025
"""

import sys
import os
import json
from pathlib import Path
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings

# Fix imports by adding project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_dir = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import system components
from integration.live_trading_system import LiveTradingSystem

# Optional imports
try:
    from integration.dynamic_portfolio_configurator import DynamicPortfolioConfigurator, PortfolioConfig
    CONFIGURATOR_AVAILABLE = True
except ImportError:
    CONFIGURATOR_AVAILABLE = False
    warnings.warn("Dynamic Portfolio Configurator not available")

try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    warnings.warn("Config module not available")


class TradingSystemManager:
    """
    Enhanced trading system manager with improved configuration,
    error handling, and portfolio management capabilities.
    """
    
    def __init__(self):
        self.trading_system = None
        self.current_config = self._load_default_config()
        self.portfolio_config = None
        self.configurator = None
        
        # Performance tracking
        self.start_time = None
        self.performance_log = []
        
        # Initialize portfolio configurator if available
        if CONFIGURATOR_AVAILABLE:
            try:
                self.configurator = DynamicPortfolioConfigurator()
            except Exception as e:
                print(f"⚠️  Could not initialize portfolio configurator: {e}")
        
        # Enhanced logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup enhanced logging."""
        import logging
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_file = log_dir / f"trading_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('TradingSystemManager')
        self.logger.info("Trading System Manager initialized")
    
    def _load_default_config(self) -> Dict:
        """Load default trading configuration."""
        return {
            'broker_name': 'alpaca',
            'rebalance_frequency': 'daily',  # 'hourly', 'daily', 'weekly'
            'rebalance_threshold': 0.15,     # 15% drift threshold
            'min_rebalance_interval': 3600,  # Minimum 1 hour between rebalances
            'max_position_size': 0.20,       # 20% max per position
            'min_position_size': 0.01,       # 1% min per position
            'execution_algorithm': 'VWAP',
            'risk_limit_enabled': True,
            'esg_constraints_enabled': True,
            'cash_buffer': 0.05,             # Keep 5% cash buffer
            'max_orders_per_rebalance': 10,  # Limit simultaneous orders
            'optimization_frequency': 'daily', # Run optimization once per day
            'paper_trading': True,           # Use paper trading by default
            'enable_fractional_shares': True, # Enable fractional share trading
            'portfolio_name': 'default',     # Default portfolio name
            'auto_save_performance': True    # Auto-save performance metrics
        }
    
    def run_interactive_setup(self):
        """Run interactive setup menu."""
        print("\n" + "="*60)
        print("🚀 QUANTUM PORTFOLIO LIVE TRADING SYSTEM")
        print("="*60)
        print("Production-Ready Trading with Enhanced Features")
        
        while True:
            try:
                self._display_setup_menu()
                choice = input("\nSelect option (1-8, s to start trading, q to quit): ").strip().lower()
                
                if choice == 'q' or choice == 'quit':
                    print("\n👋 Goodbye!")
                    return False
                elif choice == 's' or choice == 'start':
                    return self._validate_and_start()
                else:
                    self._handle_setup_choice(choice)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return False
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\nPress Enter to continue...")
    
    def _display_setup_menu(self):
        """Display the setup menu."""
        print(f"\n📋 TRADING SYSTEM SETUP")
        print(f"{'='*40}")
        print(f"1. 📦 Select Portfolio")
        print(f"2. ⚙️  Configure Trading Parameters")
        print(f"3. 🔄 Set Rebalancing Rules")
        print(f"4. 🎯 Risk Management Settings")
        print(f"5. 📊 View Current Configuration")
        print(f"6. 💾 Save/Load Configuration")
        print(f"7. 🧪 Portfolio Analysis")
        print(f"8. ℹ️  System Information")
        
        # Show current status
        if self.portfolio_config:
            print(f"\n🎯 Current Portfolio: {self.portfolio_config.name}")
            print(f"   Assets: {len(self.portfolio_config.tickers)}")
        else:
            print(f"\n⚠️  No portfolio selected")
        
        print(f"\n🔧 Broker: {self.current_config['broker_name']}")
        print(f"🔄 Rebalance Threshold: {self.current_config['rebalance_threshold']:.1%}")
        print(f"📊 Paper Trading: {self.current_config['paper_trading']}")
    
    def _handle_setup_choice(self, choice: str):
        """Handle setup menu selection."""
        handlers = {
            '1': self._select_portfolio,
            '2': self._configure_trading_parameters,
            '3': self._set_rebalancing_rules,
            '4': self._configure_risk_management,
            '5': self._view_configuration,
            '6': self._save_load_configuration,
            '7': self._analyze_portfolio,
            '8': self._system_information
        }
        
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print("❌ Invalid choice. Please try again.")
    
    def _select_portfolio(self):
        """Select trading portfolio."""
        print(f"\n📦 SELECT PORTFOLIO")
        print(f"{'='*40}")
        
        if not CONFIGURATOR_AVAILABLE:
            print("❌ Portfolio configurator not available")
            print("Using default portfolio...")
            self._use_default_portfolio()
            return
        
        print(f"1. Load from saved configurations")
        print(f"2. Use predefined universe")
        print(f"3. Create custom portfolio")
        print(f"4. Use default test portfolio")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            self._load_saved_portfolio()
        elif choice == '2':
            self._select_predefined_universe()
        elif choice == '3':
            self._create_custom_portfolio()
        elif choice == '4':
            self._use_default_portfolio()
        else:
            print("❌ Invalid choice")
    
    def _load_saved_portfolio(self):
        """Load a saved portfolio configuration."""
        if not self.configurator:
            print("❌ Configurator not available")
            return
        
        config_dir = Path("portfolio_configs")
        if not config_dir.exists():
            print("❌ No saved configurations found")
            return
        
        configs = list(config_dir.glob("*.json"))
        if not configs:
            print("❌ No saved configurations found")
            return
        
        print(f"\n📂 Saved Portfolios:")
        for i, config_file in enumerate(configs, 1):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                name = config_data.get('name', config_file.stem)
                ticker_count = len(config_data.get('tickers', []))
                print(f"{i:2}. {name:30} ({ticker_count:3} assets)")
            except Exception:
                print(f"{i:2}. {config_file.stem:30} (❌ corrupted)")
        
        try:
            selection = int(input(f"\nSelect portfolio (1-{len(configs)}): ")) - 1
            if 0 <= selection < len(configs):
                with open(configs[selection], 'r') as f:
                    config_data = json.load(f)
                self.portfolio_config = PortfolioConfig.from_dict(config_data)
                print(f"✅ Loaded '{self.portfolio_config.name}' with {len(self.portfolio_config.tickers)} assets")
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    def _select_predefined_universe(self):
        """Select from predefined universes."""
        if not self.configurator:
            print("❌ Configurator not available")
            return
        
        universes = self.configurator.universe_manager.list_universes()
        
        print(f"\n🌌 Predefined Universes:")
        for i, universe in enumerate(universes, 1):
            tickers = self.configurator.universe_manager.get_universe(universe)
            print(f"{i:2}. {universe.upper():20} ({len(tickers):3} assets)")
        
        try:
            selection = int(input(f"\nSelect universe (1-{len(universes)}): ")) - 1
            if 0 <= selection < len(universes):
                universe_name = universes[selection]
                tickers = self.configurator.universe_manager.get_universe(universe_name)
                
                self.portfolio_config = PortfolioConfig(
                    name=f"{universe_name.upper()} Portfolio",
                    description=f"Portfolio based on {universe_name.upper()} universe",
                    tickers=tickers,
                    rebalance_threshold=self.current_config['rebalance_threshold'],
                    max_position_size=self.current_config['max_position_size']
                )
                
                print(f"✅ Selected {universe_name.upper()} with {len(tickers)} assets")
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    def _create_custom_portfolio(self):
        """Create a custom portfolio."""
        print(f"\n🔧 CREATE CUSTOM PORTFOLIO")
        print(f"{'='*40}")
        
        name = input("Portfolio name: ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return
        
        description = input("Description (optional): ").strip()
        
        print("\nEnter tickers separated by spaces:")
        print("Example: AAPL MSFT GOOGL AMZN TSLA NVDA META")
        
        ticker_input = input("Tickers: ").strip().upper()
        if not ticker_input:
            print("❌ No tickers provided")
            return
        
        tickers = [ticker.strip() for ticker in ticker_input.split() if ticker.strip()]
        
        if len(tickers) < 2:
            print("❌ Need at least 2 tickers")
            return
        
        self.portfolio_config = PortfolioConfig(
            name=name,
            description=description,
            tickers=tickers,
            rebalance_threshold=self.current_config['rebalance_threshold'],
            max_position_size=self.current_config['max_position_size']
        )
        
        print(f"✅ Created '{name}' with {len(tickers)} assets")
        
        # Optionally validate tickers
        validate = input("Validate tickers against broker? (y/n): ").strip().lower()
        if validate == 'y' and self.configurator:
            self._validate_portfolio_tickers()
    
    def _use_default_portfolio(self):
        """Use default test portfolio."""
        default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'UNH']
        
        self.portfolio_config = PortfolioConfig(
            name="Default Portfolio",
            description="Balanced portfolio of large-cap stocks for testing",
            tickers=default_tickers,
            rebalance_threshold=self.current_config['rebalance_threshold'],
            max_position_size=self.current_config['max_position_size']
        )
        
        print(f"✅ Using default portfolio with {len(default_tickers)} assets")
    
    def _validate_portfolio_tickers(self):
        """Validate portfolio tickers."""
        if not self.portfolio_config or not self.configurator:
            return
        
        print("\n🔍 Validating tickers...")
        
        stock_provider = self.configurator.stock_provider
        valid, invalid = stock_provider.validate_tickers(self.portfolio_config.tickers)
        
        if invalid:
            print(f"\n❌ Invalid tickers found: {', '.join(invalid)}")
            remove = input("Remove invalid tickers? (y/n): ").strip().lower()
            if remove == 'y':
                self.portfolio_config.tickers = valid
                print(f"✅ Portfolio updated: {len(valid)} valid tickers")
        else:
            print("✅ All tickers are valid!")
    
    def _configure_trading_parameters(self):
        """Configure trading parameters."""
        print(f"\n⚙️  TRADING PARAMETERS")
        print(f"{'='*40}")
        
        try:
            # Broker selection
            print(f"Current broker: {self.current_config['broker_name']}")
            new_broker = input("Broker (alpaca/ib) [Enter to keep]: ").strip().lower()
            if new_broker in ['alpaca', 'ib']:
                self.current_config['broker_name'] = new_broker
            
            # Paper vs live trading
            print(f"Paper trading: {self.current_config['paper_trading']}")
            paper = input("Use paper trading? (y/n) [Enter to keep]: ").strip().lower()
            if paper == 'y':
                self.current_config['paper_trading'] = True
            elif paper == 'n':
                self.current_config['paper_trading'] = False
            
            # Execution algorithm
            print(f"Execution algorithm: {self.current_config['execution_algorithm']}")
            algos = ['MARKET', 'VWAP', 'TWAP', 'POV']
            print(f"Available: {', '.join(algos)}")
            new_algo = input("Algorithm [Enter to keep]: ").strip().upper()
            if new_algo in algos:
                self.current_config['execution_algorithm'] = new_algo
            
            # Cash buffer
            print(f"Cash buffer: {self.current_config['cash_buffer']:.1%}")
            new_buffer = input("Cash buffer (% as decimal, e.g., 0.05 for 5%) [Enter to keep]: ").strip()
            if new_buffer:
                self.current_config['cash_buffer'] = float(new_buffer)
            
            print("✅ Trading parameters updated")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    def _set_rebalancing_rules(self):
        """Set rebalancing rules."""
        print(f"\n🔄 REBALANCING RULES")
        print(f"{'='*40}")
        
        try:
            # Rebalance frequency
            print(f"Current frequency: {self.current_config['rebalance_frequency']}")
            frequencies = ['hourly', 'daily', 'weekly']
            print(f"Available: {', '.join(frequencies)}")
            new_freq = input("Rebalance frequency [Enter to keep]: ").strip().lower()
            if new_freq in frequencies:
                self.current_config['rebalance_frequency'] = new_freq
            
            # Drift threshold
            print(f"Drift threshold: {self.current_config['rebalance_threshold']:.1%}")
            new_threshold = input("Drift threshold (% as decimal, e.g., 0.15 for 15%) [Enter to keep]: ").strip()
            if new_threshold:
                self.current_config['rebalance_threshold'] = float(new_threshold)
            
            # Minimum interval
            print(f"Min rebalance interval: {self.current_config['min_rebalance_interval']} seconds")
            new_interval = input("Min interval (seconds) [Enter to keep]: ").strip()
            if new_interval:
                self.current_config['min_rebalance_interval'] = int(new_interval)
            
            # Optimization frequency
            print(f"Optimization frequency: {self.current_config['optimization_frequency']}")
            opt_frequencies = ['hourly', 'daily', 'weekly']
            print(f"Available: {', '.join(opt_frequencies)}")
            new_opt_freq = input("Optimization frequency [Enter to keep]: ").strip().lower()
            if new_opt_freq in opt_frequencies:
                self.current_config['optimization_frequency'] = new_opt_freq
            
            print("✅ Rebalancing rules updated")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    def _configure_risk_management(self):
        """Configure risk management settings."""
        print(f"\n🎯 RISK MANAGEMENT")
        print(f"{'='*40}")
        
        try:
            # Position size limits
            print(f"Max position size: {self.current_config['max_position_size']:.1%}")
            new_max = input("Max position size (% as decimal) [Enter to keep]: ").strip()
            if new_max:
                self.current_config['max_position_size'] = float(new_max)
            
            print(f"Min position size: {self.current_config['min_position_size']:.1%}")
            new_min = input("Min position size (% as decimal) [Enter to keep]: ").strip()
            if new_min:
                self.current_config['min_position_size'] = float(new_min)
            
            # Risk controls
            print(f"Risk limits enabled: {self.current_config['risk_limit_enabled']}")
            risk_enabled = input("Enable risk limits? (y/n) [Enter to keep]: ").strip().lower()
            if risk_enabled == 'y':
                self.current_config['risk_limit_enabled'] = True
            elif risk_enabled == 'n':
                self.current_config['risk_limit_enabled'] = False
            
            print(f"ESG constraints: {self.current_config['esg_constraints_enabled']}")
            esg_enabled = input("Enable ESG constraints? (y/n) [Enter to keep]: ").strip().lower()
            if esg_enabled == 'y':
                self.current_config['esg_constraints_enabled'] = True
            elif esg_enabled == 'n':
                self.current_config['esg_constraints_enabled'] = False
            
            print("✅ Risk management settings updated")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    def _view_configuration(self):
        """View current configuration."""
        print(f"\n📊 CURRENT CONFIGURATION")
        print(f"{'='*50}")
        
        # Portfolio info
        if self.portfolio_config:
            print(f"Portfolio:")
            print(f"  Name: {self.portfolio_config.name}")
            print(f"  Assets: {len(self.portfolio_config.tickers)}")
            print(f"  Tickers: {', '.join(self.portfolio_config.tickers[:10])}")
            if len(self.portfolio_config.tickers) > 10:
                print(f"           ... and {len(self.portfolio_config.tickers) - 10} more")
        else:
            print(f"Portfolio: ❌ None selected")
        
        print(f"\nTrading Configuration:")
        for key, value in self.current_config.items():
            if isinstance(value, float) and 0 < value < 1:
                print(f"  {key}: {value:.1%}")
            else:
                print(f"  {key}: {value}")
        
        input("\nPress Enter to continue...")
    
    def _save_load_configuration(self):
        """Save or load trading configuration."""
        print(f"\n💾 CONFIGURATION MANAGEMENT")
        print(f"{'='*40}")
        print(f"1. Save current configuration")
        print(f"2. Load configuration")
        print(f"3. Export trading script")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            self._save_configuration()
        elif choice == '2':
            self._load_configuration()
        elif choice == '3':
            self._export_trading_script()
        else:
            print("❌ Invalid choice")
    
    def _save_configuration(self):
        """Save current configuration."""
        if not self.portfolio_config:
            print("❌ No portfolio to save")
            return
        
        config_dir = Path("trading_configs")
        config_dir.mkdir(exist_ok=True)
        
        filename = input("Configuration name: ").strip()
        if not filename:
            filename = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config_data = {
            'portfolio': self.portfolio_config.to_dict(),
            'trading': self.current_config,
            'created_at': datetime.now().isoformat()
        }
        
        config_path = config_dir / f"{filename}.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved to {config_path}")
    
    def _load_configuration(self):
        """Load saved configuration."""
        config_dir = Path("trading_configs")
        if not config_dir.exists():
            print("❌ No saved configurations")
            return
        
        configs = list(config_dir.glob("*.json"))
        if not configs:
            print("❌ No saved configurations")
            return
        
        print(f"\nSaved Configurations:")
        for i, config_file in enumerate(configs, 1):
            print(f"{i}. {config_file.stem}")
        
        try:
            selection = int(input(f"Select configuration (1-{len(configs)}): ")) - 1
            if 0 <= selection < len(configs):
                with open(configs[selection], 'r') as f:
                    config_data = json.load(f)
                
                self.portfolio_config = PortfolioConfig.from_dict(config_data['portfolio'])
                self.current_config.update(config_data['trading'])
                
                print(f"✅ Configuration loaded")
        except (ValueError, IndexError, KeyError) as e:
            print(f"❌ Error loading configuration: {e}")
    
    def _export_trading_script(self):
        """Export standalone trading script."""
        if not self.portfolio_config:
            print("❌ No portfolio to export")
            return
        
        script_content = self._generate_trading_script()
        
        filename = f"trading_script_{self.portfolio_config.name.lower().replace(' ', '_')}.py"
        script_path = Path("exported_scripts") / filename
        script_path.parent.mkdir(exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Trading script exported to {script_path}")
    
    def _generate_trading_script(self) -> str:
        """Generate standalone trading script."""
        return f'''#!/usr/bin/env python3
"""
Generated Trading Script: {self.portfolio_config.name}
Created: {datetime.now().isoformat()}
"""

from integration.live_trading_system import LiveTradingSystem
import time
import signal
import sys

def signal_handler(sig, frame):
    global trading_system
    print("\\nStopping trading system...")
    if trading_system:
        trading_system.stop()
    sys.exit(0)

# Portfolio Configuration
tickers = {self.portfolio_config.tickers}

# Initialize Trading System
trading_system = LiveTradingSystem(
    broker_name='{self.current_config['broker_name']}',
    tickers=tickers,
    rebalance_frequency='{self.current_config['rebalance_frequency']}'
)

# Configure Parameters
trading_system.config.update({json.dumps(self.current_config, indent=4)})

# Setup signal handler
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Starting {self.portfolio_config.name}")
    print(f"Assets: {{len(tickers)}}")
    
    if trading_system.start():
        try:
            while True:
                time.sleep(60)
                status = trading_system.get_system_status()
                portfolio = status.get('portfolio', {{}})
                print(f"Portfolio Value: ${{portfolio.get('total_value', 0):,.2f}}")
        except KeyboardInterrupt:
            pass
    else:
        print("Failed to start trading system")
'''
    
    def _analyze_portfolio(self):
        """Analyze current portfolio."""
        if not self.portfolio_config or not CONFIGURATOR_AVAILABLE:
            print("❌ Portfolio analysis not available")
            return
        
        print(f"\n📊 PORTFOLIO ANALYSIS")
        print(f"{'='*50}")
        
        analyzer = self.configurator.analyzer
        analysis = analyzer.analyze_portfolio(self.portfolio_config.tickers)
        
        if analysis:
            analyzer.print_analysis(analysis)
        else:
            print("❌ Could not analyze portfolio")
        
        input("\nPress Enter to continue...")
    
    def _system_information(self):
        """Display system information."""
        print(f"\nℹ️  SYSTEM INFORMATION")
        print(f"{'='*50}")
        
        print(f"System Status:")
        print(f"  Configuration Module: {'✅' if CONFIG_AVAILABLE else '❌'}")
        print(f"  Portfolio Configurator: {'✅' if CONFIGURATOR_AVAILABLE else '❌'}")
        print(f"  Live Trading System: ✅")
        
        if CONFIG_AVAILABLE:
            print(f"\nEnvironment:")
            print(f"  Environment: {getattr(Config, 'ENVIRONMENT', 'Unknown')}")
            print(f"  Alpaca API: {'✅' if getattr(Config, 'ALPACA_API_KEY', None) else '❌'}")
            print(f"  D-Wave API: {'✅' if getattr(Config, 'DWAVE_API_TOKEN', None) else '❌'}")
        
        print(f"\nFeatures:")
        print(f"  Quantum Optimization: ✅")
        print(f"  ESG Integration: ✅")
        print(f"  Risk Management: ✅")
        print(f"  Dynamic Portfolios: {'✅' if CONFIGURATOR_AVAILABLE else '❌'}")
        print(f"  Transaction Cost Analysis: ✅")
        print(f"  Order Management: ✅")
        
        input("\nPress Enter to continue...")
    
    def _validate_and_start(self) -> bool:
        """Validate configuration and start trading."""
        if not self.portfolio_config:
            print("❌ No portfolio selected. Please select a portfolio first.")
            return False
        
        if len(self.portfolio_config.tickers) < 2:
            print("❌ Portfolio must have at least 2 assets.")
            return False
        
        print(f"\n🔍 VALIDATION")
        print(f"{'='*40}")
        print(f"✅ Portfolio: {self.portfolio_config.name} ({len(self.portfolio_config.tickers)} assets)")
        print(f"✅ Broker: {self.current_config['broker_name']}")
        print(f"✅ Paper Trading: {self.current_config['paper_trading']}")
        print(f"✅ Rebalance Threshold: {self.current_config['rebalance_threshold']:.1%}")
        
        confirm = input("\nStart trading system? (y/n): ").strip().lower()
        return confirm == 'y'
    
    def start_trading(self):
        """Start the trading system with current configuration."""
        try:
            # Initialize trading system
            self.trading_system = LiveTradingSystem(
                broker_name=self.current_config['broker_name'],
                tickers=self.portfolio_config.tickers,
                rebalance_frequency=self.current_config['rebalance_frequency']
            )
            
            # Apply configuration
            enhanced_config = self.current_config.copy()
            enhanced_config.update({
                'rebalance_threshold': self.portfolio_config.rebalance_threshold,
                'max_position_size': self.portfolio_config.max_position_size,
                'min_position_size': self.portfolio_config.min_position_size
            })
            
            self.trading_system.config.update(enhanced_config)
            
            # Start system
            self.start_time = datetime.now()
            
            print(f"\n🚀 STARTING TRADING SYSTEM")
            print(f"{'='*50}")
            print(f"Portfolio: {self.portfolio_config.name}")
            print(f"Assets: {len(self.portfolio_config.tickers)}")
            print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            if self.trading_system.start():
                self.logger.info("Trading system started successfully")
                return self._run_trading_loop()
            else:
                print("❌ Failed to start trading system")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting trading system: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def _run_trading_loop(self):
        """Run the main trading loop."""
        print("✅ System running!")
        print("\n📊 Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                time.sleep(60)  # Check every minute
                iteration += 1
                
                status = self._get_enhanced_status()
                self._display_status_update(iteration, status)
                
                # Log performance
                if self.current_config['auto_save_performance']:
                    self._log_performance(status)
                
        except KeyboardInterrupt:
            return self._shutdown_system()
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
            print(f"\n❌ Trading loop error: {e}")
            return self._shutdown_system()
    
    def _get_enhanced_status(self) -> Dict:
        """Get enhanced system status."""
        if not self.trading_system:
            return {}
        
        base_status = self.trading_system.get_system_status()
        
        # Add performance metrics
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600  # hours
        
        enhanced_status = base_status.copy()
        enhanced_status.update({
            'runtime_hours': runtime,
            'start_time': self.start_time,
            'portfolio_name': self.portfolio_config.name,
            'total_assets': len(self.portfolio_config.tickers)
        })
        
        return enhanced_status
    
    def _display_status_update(self, iteration: int, status: Dict):
        """Display formatted status update."""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        print(f"\n{'='*50}")
        print(f"Status Update #{iteration} - {current_time}")
        print(f"{'='*50}")
        print(f"Portfolio: {status.get('portfolio_name', 'Unknown')}")
        print(f"Runtime: {status.get('runtime_hours', 0):.1f} hours")
        print(f"Connected: {status.get('connected', False)}")
        print(f"Running: {status.get('running', False)}")
        
        if 'portfolio' in status:
            portfolio = status['portfolio']
            print(f"\n💼 Portfolio:")
            print(f"  Value: ${portfolio.get('total_value', 0):,.2f}")
            print(f"  P&L: ${portfolio.get('total_pnl', 0):+,.2f} ({portfolio.get('total_pnl_pct', 0):+.2%})")
            print(f"  Cash: ${portfolio.get('cash', 0):,.2f}")
            print(f"  Positions: {portfolio.get('num_positions', 0)}")
            
            if portfolio.get('long_market_value', 0) > 0:
                print(f"  Long: ${portfolio.get('long_market_value', 0):,.2f}")
            if portfolio.get('short_market_value', 0) > 0:
                print(f"  Short: ${portfolio.get('short_market_value', 0):,.2f}")
        
        if 'orders' in status:
            orders = status['orders']
            print(f"\n📋 Orders:")
            print(f"  Active: {orders.get('active_orders', 0)}")
            print(f"  Submitted: {orders.get('orders_submitted', 0)}")
            print(f"  Filled: {orders.get('orders_filled', 0)}")
            if orders.get('orders_submitted', 0) > 0:
                fill_rate = orders.get('fill_rate', 0)
                print(f"  Fill Rate: {fill_rate:.1%}")
        
        if status.get('last_optimization'):
            print(f"\n🔬 Last Optimization: {status['last_optimization'].strftime('%H:%M:%S')}")
        if status.get('last_rebalance'):
            print(f"⚖️  Last Rebalance: {status['last_rebalance'].strftime('%H:%M:%S')}")
        
        print(f"\n{'='*50}\n")
    
    def _log_performance(self, status: Dict):
        """Log performance metrics."""
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': status.get('portfolio', {}).get('total_value', 0),
            'pnl': status.get('portfolio', {}).get('total_pnl', 0),
            'pnl_pct': status.get('portfolio', {}).get('total_pnl_pct', 0),
            'positions': status.get('portfolio', {}).get('num_positions', 0),
            'orders_filled': status.get('orders', {}).get('orders_filled', 0),
            'fill_rate': status.get('orders', {}).get('fill_rate', 0)
        }
        
        self.performance_log.append(performance_entry)
        
        # Save to file periodically
        if len(self.performance_log) % 10 == 0:  # Every 10 updates
            self._save_performance_log()
    
    def _save_performance_log(self):
        """Save performance log to file."""
        if not self.performance_log:
            return
        
        log_dir = Path("performance_logs")
        log_dir.mkdir(exist_ok=True)
        
        filename = f"performance_{self.portfolio_config.name.replace(' ', '_')}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        log_path = log_dir / filename
        
        try:
            with open(log_path, 'w') as f:
                json.dump(self.performance_log, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving performance log: {e}")
    
    def _shutdown_system(self) -> bool:
        """Shutdown the trading system."""
        print(f"\n🛑 Stopping system...")
        
        try:
            if self.trading_system:
                self.trading_system.stop()
            
            # Save final performance
            if self.performance_log and self.current_config['auto_save_performance']:
                self._save_performance_log()
            
            # Display final summary
            self._display_final_summary()
            
            print("✅ System stopped safely")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            print(f"❌ Shutdown error: {e}")
            return False
    
    def _display_final_summary(self):
        """Display final trading summary."""
        if not self.trading_system:
            return
        
        print(f"\n📊 FINAL SUMMARY")
        print(f"{'='*40}")
        
        try:
            final_status = self.trading_system.get_system_status()
            runtime = (datetime.now() - self.start_time).total_seconds() / 3600
            
            print(f"Portfolio: {self.portfolio_config.name}")
            print(f"Runtime: {runtime:.1f} hours")
            
            if 'portfolio' in final_status:
                portfolio = final_status['portfolio']
                print(f"Final Value: ${portfolio.get('total_value', 0):,.2f}")
                print(f"Total P&L: ${portfolio.get('total_pnl', 0):+,.2f}")
                print(f"P&L %: {portfolio.get('total_pnl_pct', 0):+.2%}")
            
            if 'orders' in final_status:
                orders = final_status['orders']
                print(f"Orders Filled: {orders.get('orders_filled', 0)}")
                print(f"Fill Rate: {orders.get('fill_rate', 0):.1%}")
            
        except Exception as e:
            self.logger.error(f"Error generating final summary: {e}")


def main():
    """Main function to run the trading system manager."""
    try:
        manager = TradingSystemManager()
        
        # Run interactive setup
        if manager.run_interactive_setup():
            # Start trading
            success = manager.start_trading()
            
            if not success:
                print("❌ Trading system failed to start properly")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())