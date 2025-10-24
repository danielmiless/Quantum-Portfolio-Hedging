# dashboard.py
"""
Real-time portfolio monitoring dashboard
"""


import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


# Import our quantum portfolio modules
from quantum.data_preparation import PortfolioDataPreparer
from quantum.quantum_hardware_interface import DWaveQUBOSolver, QuantumPortfolioOptimizer
from alt_data.ml_return_forecasting import MLReturnForecaster, ForecastingConfig


# Move load_data outside the class as a standalone cached function
@st.cache_data
def load_data(tickers, start_date, end_date):
    """Load and cache market data."""
    try:
        preparer = PortfolioDataPreparer(
            list(tickers),
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        with st.spinner('Loading market data...'):
            data = preparer.download_data()
            stats = preparer.calculate_statistics()
        
        return data, stats, preparer
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


class QuantumPortfolioDashboard:
    """
    Real-time quantum portfolio optimization dashboard.
    """
    
    def __init__(self):
        self.setup_page_config()
        
    def setup_page_config(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="Quantum Portfolio Optimizer",
            page_icon="⚛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def run(self):
        """Main dashboard interface."""
        st.title("⚛️ Quantum Portfolio Optimization Dashboard")
        st.markdown("*Advanced portfolio management with quantum computing and machine learning*")
        
        # Sidebar controls
        self.render_sidebar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Portfolio Overview", 
            "⚛️ Quantum Optimization", 
            "🤖 ML Forecasting", 
            "📈 Performance Analytics"
        ])
        
        with tab1:
            self.render_portfolio_overview()
            
        with tab2:
            self.render_quantum_optimization()
            
        with tab3:
            self.render_ml_forecasting()
            
        with tab4:
            self.render_performance_analytics()
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("Configuration")
        
        # Portfolio settings
        st.sidebar.subheader("Portfolio Settings")
        tickers = st.sidebar.text_input(
            "Asset Tickers", 
            value="AAPL,GOOGL,MSFT,AMZN,TSLA",
            help="Comma-separated list of ticker symbols"
        ).split(',')
        
        start_date = st.sidebar.date_input(
            "Start Date", 
            value=datetime.now() - timedelta(days=365)
        )
        
        end_date = st.sidebar.date_input(
            "End Date", 
            value=datetime.now()
        )
        
        # Optimization settings
        st.sidebar.subheader("Optimization Settings")
        risk_aversion = st.sidebar.slider(
            "Risk Aversion", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1
        )
        
        use_quantum = st.sidebar.checkbox("Use Quantum Hardware", value=False)
        use_ml_forecasts = st.sidebar.checkbox("Use ML Forecasts", value=True)
        
        # Store settings in session state
        st.session_state.update({
            'tickers': [t.strip() for t in tickers],
            'start_date': start_date,
            'end_date': end_date,
            'risk_aversion': risk_aversion,
            'use_quantum': use_quantum,
            'use_ml_forecasts': use_ml_forecasts
        })
    
    def render_portfolio_overview(self):
        """Render portfolio overview tab."""
        st.header("Portfolio Overview")
        
        # Load data using standalone function
        data, stats, preparer = load_data(
            tuple(st.session_state.tickers),  # Convert to tuple for hashing
            st.session_state.start_date,
            st.session_state.end_date
        )
        
        if data is None:
            return
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Number of Assets", 
                len(st.session_state.tickers)
            )
        
        with col2:
            avg_return = stats['mean_returns'].mean()
            st.metric(
                "Average Return", 
                f"{avg_return:.2%}"
            )
        
        with col3:
            avg_volatility = stats['volatilities'].mean()
            st.metric(
                "Average Volatility", 
                f"{avg_volatility:.2%}"
            )
        
        with col4:
            avg_sharpe = stats['sharpe_ratios'].mean()
            st.metric(
                "Average Sharpe Ratio", 
                f"{avg_sharpe:.3f}"
            )
        
        # Price chart
        st.subheader("Price Evolution")
        fig = go.Figure()
        
        for ticker in st.session_state.tickers:
            if ticker in data['prices'].columns:
                prices = data['prices'][ticker]
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=prices.values,
                    mode='lines',
                    name=ticker
                ))
        
        fig.update_layout(
            title="Asset Prices Over Time",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Asset Correlations")
        corr_matrix = stats['returns'].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Return Correlations"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def render_quantum_optimization(self):
        """Render quantum optimization tab."""
        st.header("⚛️ Quantum Portfolio Optimization")
        
        # Load data
        data, stats, preparer = load_data(
            tuple(st.session_state.tickers),
            st.session_state.start_date,
            st.session_state.end_date
        )
        
        if data is None:
            return
        
        # Optimization button
        if st.button("🚀 Optimize Portfolio", type="primary"):
            
            with st.spinner('Running quantum optimization...'):
                # Get optimization inputs
                mu, sigma = preparer.get_optimization_inputs('ledoit_wolf')
                
                # Setup quantum solver
                quantum_solver = DWaveQUBOSolver(use_hardware=st.session_state.use_quantum)
                optimizer = QuantumPortfolioOptimizer(quantum_solver)
                
                # Run optimization
                result = optimizer.optimize_portfolio(
                    mu, sigma, 
                    risk_aversion=st.session_state.risk_aversion
                )
                
                # Store results
                st.session_state.quantum_result = result
        
        # Display results if available
        if hasattr(st.session_state, 'quantum_result'):
            result = st.session_state.quantum_result
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected Return", 
                    f"{result['return']:.2%}"
                )
            
            with col2:
                st.metric(
                    "Portfolio Risk", 
                    f"{result['risk']:.2%}"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio", 
                    f"{result['sharpe_ratio']:.3f}"
                )
            
            with col4:
                solver_type = "Quantum" if result['solver_info']['hardware'] else "Classical"
                st.metric(
                    "Solver Type", 
                    solver_type
                )
            
            # Portfolio weights visualization
            st.subheader("Optimal Portfolio Weights")
            
            weights_df = pd.DataFrame({
                'Asset': st.session_state.tickers,
                'Weight': result['weights']
            })
            
            fig_weights = px.pie(
                weights_df, 
                values='Weight', 
                names='Asset',
                title="Portfolio Allocation"
            )
            
            st.plotly_chart(fig_weights, use_container_width=True)
            
            # Weights table
            st.subheader("Portfolio Weights")
            weights_df['Weight %'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
            st.dataframe(weights_df[['Asset', 'Weight %']], hide_index=True)
    
    def render_ml_forecasting(self):
        """Render ML forecasting tab."""
        st.header("🤖 Machine Learning Forecasts")
        
        if not st.session_state.use_ml_forecasts:
            st.warning("ML forecasting is disabled. Enable it in the sidebar to view this section.")
            return
        
        # Load data
        data, stats, preparer = load_data(
            tuple(st.session_state.tickers),
            st.session_state.start_date,
            st.session_state.end_date
        )
        
        if data is None:
            return
        
        # ML configuration
        st.subheader("ML Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_horizon = st.selectbox(
                "Forecast Horizon (days)", 
                [1, 5, 10, 20], 
                index=1
            )
        
        with col2:
            lookback_window = st.selectbox(
                "Lookback Window (days)", 
                [30, 60, 90, 120], 
                index=1
            )
        
        # Generate forecasts button
        if st.button("🔮 Generate Forecasts", type="primary"):
            
            with st.spinner('Training ML models and generating forecasts...'):
                # Setup ML forecaster
                config = ForecastingConfig(
                    lookback_window=lookback_window,
                    forecast_horizon=forecast_horizon,
                    feature_engineering=True,
                    ensemble_models=True
                )
                
                forecaster = MLReturnForecaster(config)
                
                try:
                    # Prepare training data
                    X, y = forecaster.prepare_training_data(data['prices'])
                    
                    if len(X) < 100:
                        st.warning("Insufficient data for ML training. Need at least 100 observations.")
                        return
                    
                    # Split data
                    split_idx = int(0.8 * len(X))
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Train models
                    forecaster.train_models(X_train, y_train)
                    
                    # Generate predictions
                    predictions = forecaster.predict_returns(X_test)
                    
                    # Evaluate
                    metrics = forecaster.evaluate_forecasts(y_test, predictions)
                    
                    # Store results
                    st.session_state.ml_results = {
                        'predictions': predictions,
                        'metrics': metrics,
                        'y_test': y_test
                    }
                    
                except Exception as e:
                    st.error(f"ML forecasting error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        # Display ML results if available
        if hasattr(st.session_state, 'ml_results'):
            results = st.session_state.ml_results
            
            # Model performance
            st.subheader("Model Performance")
            
            metrics_df = pd.DataFrame(results['metrics']).T
            metrics_df = metrics_df.round(6)
            
            st.dataframe(metrics_df)
            
            # Best model
            best_model = min(results['metrics'].keys(), 
                           key=lambda k: results['metrics'][k]['rmse'])
            st.success(f"Best performing model: **{best_model}** (RMSE: {results['metrics'][best_model]['rmse']:.6f})")
            
            # Forecast visualization
            st.subheader("Return Forecasts")
            
            # Show forecasts for each asset
            for i, ticker in enumerate(st.session_state.tickers):
                if i < results['predictions']['ensemble'].shape[1]:
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Time series of actual vs predicted
                        actual = results['y_test'][:, i]
                        predicted = results['predictions']['ensemble'][:, i]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=actual,
                            mode='lines',
                            name=f'Actual {ticker}',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            y=predicted,
                            mode='lines',
                            name=f'Predicted {ticker}',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"{ticker} - Actual vs Predicted Returns",
                            yaxis_title="Return",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Latest forecast
                        latest_forecast = predicted[-1]
                        st.metric(
                            f"{ticker} Forecast",
                            f"{latest_forecast:.2%}",
                            delta=f"{latest_forecast - actual[-1]:.2%}"
                        )
    
    def render_performance_analytics(self):
        """Render performance analytics tab."""
        st.header("📈 Performance Analytics")
        
        st.info("Performance analytics would be implemented here with:")
        st.markdown("""
        - **Backtesting Results**: Historical performance simulation
        - **Risk Attribution**: Factor-based risk decomposition  
        - **Performance Attribution**: Return source analysis
        - **Stress Testing**: Scenario analysis and stress tests
        - **Real-time Monitoring**: Live portfolio tracking
        """)
        
        # Placeholder metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("YTD Return", "12.5%", "2.1%")
        
        with col2:
            st.metric("Max Drawdown", "-8.2%", "1.1%")
        
        with col3:
            st.metric("Information Ratio", "0.85", "0.05")


def main():
    """Main dashboard entry point."""
    dashboard = QuantumPortfolioDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

# cd "/Users/danielmiles/Documents/Quant Finance/Quantum Portfolio Hedging/src"
# streamlit run integration/portfolio_dashboard.py