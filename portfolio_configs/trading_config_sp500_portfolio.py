# Generated Portfolio Configuration
# Portfolio: SP500 Portfolio
# Generated: 2025-10-24T10:06:31.311153

from integration.live_trading_system import LiveTradingSystem

# Portfolio Configuration
tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'AVGO', 'KO', 'LLY', 'WMT', 'TMO', 'MRK', 'COST', 'ABT', 'CSCO', 'ACN', 'DHR', 'VZ', 'ADBE', 'TXN', 'NEE', 'NKE', 'PM', 'CRM', 'DIS', 'WFC', 'RTX', 'ORCL', 'NFLX', 'AMD', 'QCOM', 'INTC', 'T', 'COP', 'UPS', 'HON', 'IBM', 'GS', 'SPGI', 'LOW', 'SBUX', 'CAT', 'INTU', 'AXP', 'DE']

# Trading System Configuration
trading_system = LiveTradingSystem(
    broker_name='alpaca',
    tickers=tickers,
    rebalance_frequency='daily'
)

# Portfolio Parameters
trading_system.config.update({
    'rebalance_threshold': 0.05,
    'max_position_size': 0.25,
    'min_position_size': 0.01,
    'execution_algorithm': 'VWAP',
    'risk_limit_enabled': True,
    'esg_constraints_enabled': True
})

# Start Trading System
if __name__ == "__main__":
    print("Starting SP500 Portfolio")
    print(f"Tickers: {len(tickers)} assets")
    
    if trading_system.start():
        try:
            import time
            while True:
                time.sleep(60)
                status = trading_system.get_system_status()
                print(f"Portfolio Value: ${status.get('portfolio', {}).get('total_value', 0):,.2f}")
        except KeyboardInterrupt:
            trading_system.stop()
    else:
        print("Failed to start trading system")
