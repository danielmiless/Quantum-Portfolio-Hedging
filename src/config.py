# src/config.py
"""
Centralized configuration management using environment variables.
Fixed to work from any directory.
"""

import os
import sys
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Add the src directory to the Python path so imports work
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = current_dir

# Add both project root and src to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Load environment variables - try multiple locations
env_loaded = False

# Try to find .env in project root first
env_paths = [
    project_root / '.env',  # Project root
    current_dir / '.env',   # src directory
    Path.cwd() / '.env',    # Current working directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
        print(f"✅ Loaded .env from: {env_path}")
        break

if not env_loaded:
    # Try find_dotenv as fallback
    try:
        dotenv_path = find_dotenv()
        if dotenv_path:
            load_dotenv(dotenv_path)
            print(f"✅ Loaded .env from: {dotenv_path}")
        else:
            print("⚠️  No .env file found")
    except Exception as e:
        print(f"⚠️  Could not load .env: {e}")


class Config:
    """Application configuration from environment variables."""
    
    # Broker API Keys
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')
    
    IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
    IB_PORT = int(os.getenv('IB_PORT', '7497'))
    IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', '1'))
    
    TD_API_KEY = os.getenv('TD_API_KEY')
    TD_REDIRECT_URI = os.getenv('TD_REDIRECT_URI')
    TD_ACCOUNT_ID = os.getenv('TD_ACCOUNT_ID')
    
    # D-Wave Quantum
    DWAVE_API_TOKEN = os.getenv('DWAVE_API_TOKEN')
    DWAVE_SOLVER = os.getenv('DWAVE_SOLVER', 'Advantage_system4.1')
    
    # ESG Data Providers
    MSCI_ESG_API_KEY = os.getenv('MSCI_ESG_API_KEY')
    SUSTAINALYTICS_API_KEY = os.getenv('SUSTAINALYTICS_API_KEY')
    REFINITIV_API_KEY = os.getenv('REFINITIV_API_KEY')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL')
    REDIS_URL = os.getenv('REDIS_URL')
    
    # Application Settings
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    
    # Risk Limits
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.25'))
    MAX_PORTFOLIO_LEVERAGE = float(os.getenv('MAX_PORTFOLIO_LEVERAGE', '2.0'))
    REBALANCE_THRESHOLD = float(os.getenv('REBALANCE_THRESHOLD', '0.05'))
    
    # Execution Settings
    DEFAULT_EXECUTION_ALGORITHM = os.getenv('DEFAULT_EXECUTION_ALGORITHM', 'VWAP')
    MAX_PARTICIPATION_RATE = float(os.getenv('MAX_PARTICIPATION_RATE', '0.20'))
    ENABLE_PAPER_TRADING = os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true'
    
    # Monitoring
    SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
    EMAIL_ALERTS_ENABLED = os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true'
    ALERT_EMAIL = os.getenv('ALERT_EMAIL')
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required environment variables are set."""
        # Warning for missing broker config, but don't fail
        if not cls.ALPACA_API_KEY and not cls.IB_HOST:
            import warnings
            warnings.warn(
                "No broker API keys found in environment. "
                "Set ALPACA_API_KEY/ALPACA_SECRET_KEY or IB_HOST in .env file"
            )
        return True
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production."""
        return cls.ENVIRONMENT.lower() == 'production'
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development."""
        return cls.ENVIRONMENT.lower() == 'development'
    
    @classmethod
    def debug_env_vars(cls):
        """Debug environment variables."""
        print("\n🔍 Environment Variables Debug:")
        vars_to_check = [
            'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL',
            'DWAVE_API_TOKEN', 'ENVIRONMENT'
        ]
        
        for var in vars_to_check:
            value = os.getenv(var)
            if value:
                # Don't print full API keys for security
                if 'KEY' in var or 'TOKEN' in var:
                    masked = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
                    print(f"  {var}: {masked}")
                else:
                    print(f"  {var}: {value}")
            else:
                print(f"  {var}: ❌ Not set")


# Validate configuration on import
Config.validate()

# Debug print (remove in production)
if Config.is_development():
    Config.debug_env_vars()
