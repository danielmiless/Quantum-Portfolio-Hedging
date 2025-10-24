# ml_return_forecasting.py
"""
Machine learning models for return forecasting and portfolio optimization
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


# Deep learning (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Using sklearn models only.")



@dataclass
class ForecastingConfig:
    """Configuration for ML forecasting models."""
    lookback_window: int = 60
    forecast_horizon: int = 5
    feature_engineering: bool = True
    cross_validation_splits: int = 5
    ensemble_models: bool = True



class FeatureEngineer:
    """
    Financial feature engineering for return forecasting.
    """
    
    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        
    def create_technical_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features."""
        features = pd.DataFrame(index=prices.index)
        
        for col in prices.columns:
            price = prices[col]
            
            # Moving averages
            features[f'{col}_MA_5'] = price.rolling(5).mean()
            features[f'{col}_MA_20'] = price.rolling(20).mean()
            features[f'{col}_MA_60'] = price.rolling(60).mean()
            
            # Moving average ratios (with safeguards)
            ma_20 = features[f'{col}_MA_20']
            ma_60 = features[f'{col}_MA_60']
            features[f'{col}_MA_ratio_5_20'] = features[f'{col}_MA_5'] / ma_20.replace(0, np.nan)
            features[f'{col}_MA_ratio_20_60'] = ma_20 / ma_60.replace(0, np.nan)
            
            # Volatility features
            returns = price.pct_change()
            features[f'{col}_volatility_5'] = returns.rolling(5).std()
            features[f'{col}_volatility_20'] = returns.rolling(20).std()
            
            # Momentum features (with safeguards)
            shifted_5 = price.shift(5).replace(0, np.nan)
            shifted_20 = price.shift(20).replace(0, np.nan)
            features[f'{col}_momentum_5'] = (price / shifted_5 - 1).clip(-1, 1)
            features[f'{col}_momentum_20'] = (price / shifted_20 - 1).clip(-1, 1)
            
            # RSI (simplified, with safeguards)
            delta = returns
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            features[f'{col}_RSI'] = 100 - (100 / (1 + rs))
            features[f'{col}_RSI'] = features[f'{col}_RSI'].fillna(50)  # Neutral RSI if undefined
            
            # Bollinger Bands (with safeguards)
            ma20 = features[f'{col}_MA_20']
            std20 = price.rolling(20).std()
            features[f'{col}_BB_upper'] = ma20 + (std20 * 2)
            features[f'{col}_BB_lower'] = ma20 - (std20 * 2)
            bb_range = (features[f'{col}_BB_upper'] - features[f'{col}_BB_lower']).replace(0, np.nan)
            features[f'{col}_BB_position'] = (price - features[f'{col}_BB_lower']) / bb_range
            features[f'{col}_BB_position'] = features[f'{col}_BB_position'].clip(0, 1)
        
        # Replace any remaining inf/nan
        features = features.replace([np.inf, -np.inf], np.nan)
        return features.dropna()
    
    def create_market_features(self, prices: pd.DataFrame, 
                              market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create market-wide features."""
        features = pd.DataFrame(index=prices.index)
        
        # Market volatility (using equal-weighted portfolio as proxy)
        market_returns = prices.pct_change().mean(axis=1)
        features['market_volatility'] = market_returns.rolling(20).std()
        
        # Market momentum (with safeguards)
        market_prices = prices.mean(axis=1)
        shifted_5 = market_prices.shift(5).replace(0, np.nan)
        shifted_20 = market_prices.shift(20).replace(0, np.nan)
        features['market_momentum_5'] = (market_prices / shifted_5 - 1).clip(-1, 1)
        features['market_momentum_20'] = (market_prices / shifted_20 - 1).clip(-1, 1)
        
        # Correlation features
        returns = prices.pct_change()
        rolling_corr = returns.rolling(60).corr()
        features['avg_correlation'] = rolling_corr.groupby(level=0).mean().mean(axis=1)
        
        # Add external market data if provided (VIX, interest rates, etc.)
        if market_data is not None:
            for col in market_data.columns:
                features[f'market_{col}'] = market_data[col]
        
        # Replace any remaining inf/nan
        features = features.replace([np.inf, -np.inf], np.nan)
        return features.dropna()



class MLReturnForecaster:
    """
    Machine learning ensemble for return forecasting.
    """
    
    def __init__(self, config: ForecastingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer(config.lookback_window)
        
    def prepare_training_data(self, prices: pd.DataFrame,
                             market_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for ML training.
        
        Args:
            prices: Asset price data
            market_data: Additional market data (VIX, rates, etc.)
            
        Returns:
            X: Features array
            y: Target returns array
        """
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Feature engineering
        if self.config.feature_engineering:
            tech_features = self.feature_engineer.create_technical_features(prices)
            market_features = self.feature_engineer.create_market_features(prices, market_data)
            
            # Combine features
            features = pd.concat([tech_features, market_features], axis=1)
        else:
            # Simple lagged returns
            features = pd.DataFrame()
            for lag in range(1, self.config.lookback_window + 1):
                for col in returns.columns:
                    features[f'{col}_lag_{lag}'] = returns[col].shift(lag)
        
        # Align features and returns
        common_dates = features.index.intersection(returns.index)
        features = features.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # Create targets (forward returns)
        targets = returns.shift(-self.config.forecast_horizon)
        
        # Remove NaN values
        valid_idx = features.dropna().index.intersection(targets.dropna().index)
        X = features.loc[valid_idx].values
        y = targets.loc[valid_idx].values
        
        # Additional safety: clip extreme values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train ensemble of ML models.
        
        Args:
            X: Features array (samples x features)
            y: Target returns (samples x assets)
            
        Returns:
            Training results and metrics
        """
        n_assets = y.shape[1]
        results = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_splits)
        
        # Model definitions with more regularization
        model_configs = {
            'ridge': Ridge(alpha=10.0),  # Increased regularization
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000),  # More iterations
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        # Train models for each asset
        for asset_idx in range(n_assets):
            asset_models = {}
            asset_scalers = {}
            
            y_asset = y[:, asset_idx]
            
            # Scale features with robust scaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Additional clipping after scaling
            X_scaled = np.clip(X_scaled, -10, 10)
            
            asset_scalers['features'] = scaler
            
            for model_name, model in model_configs.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y_asset, 
                                              cv=tscv, scoring='neg_mean_squared_error')
                    
                    # Train on full data
                    model.fit(X_scaled, y_asset)
                    asset_models[model_name] = model
                    
                    print(f"Asset {asset_idx}, {model_name}: CV Score = {-cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
                except Exception as e:
                    print(f"Asset {asset_idx}, {model_name}: Training failed - {e}")
                    continue
            
            self.models[asset_idx] = asset_models
            self.scalers[asset_idx] = asset_scalers
        
        return results
    
    def predict_returns(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate return forecasts using trained models.
        
        Args:
            X: Features array for prediction
            
        Returns:
            Dictionary of predictions by model
        """
        if not self.models:
            raise ValueError("Models must be trained first")
        
        n_assets = len(self.models)
        n_samples = X.shape[0]
        
        # Safety: clean input
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = {}
        
        # Get predictions from each model type
        for model_name in ['ridge', 'elastic_net', 'random_forest', 'gradient_boosting']:
            model_predictions = np.zeros((n_samples, n_assets))
            
            for asset_idx in range(n_assets):
                if model_name not in self.models[asset_idx]:
                    continue
                    
                # Scale features
                X_scaled = self.scalers[asset_idx]['features'].transform(X)
                
                # Clip extreme values after scaling
                X_scaled = np.clip(X_scaled, -10, 10)
                
                # Safety: ensure no inf/nan
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Predict
                model = self.models[asset_idx][model_name]
                model_predictions[:, asset_idx] = model.predict(X_scaled)
            
            predictions[model_name] = model_predictions
        
        # Ensemble prediction (equal weighting)
        if self.config.ensemble_models and len(predictions) > 0:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def evaluate_forecasts(self, y_true: np.ndarray, 
                          predictions: Dict[str, np.ndarray]) -> Dict:
        """
        Evaluate forecast accuracy.
        
        Args:
            y_true: Actual returns
            predictions: Dictionary of predictions
            
        Returns:
            Evaluation metrics
        """
        metrics = {}
        
        for model_name, y_pred in predictions.items():
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Directional accuracy
            direction_correct = np.mean(np.sign(y_true) == np.sign(y_pred))
            
            metrics[model_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'direction_accuracy': direction_correct
            }
        
        return metrics



class QuantumMLPortfolioOptimizer:
    """
    Integration of quantum optimization with ML forecasting.
    """
    
    def __init__(self, ml_forecaster: MLReturnForecaster,
                 quantum_optimizer: 'QuantumPortfolioOptimizer'):
        """
        Args:
            ml_forecaster: Trained ML forecasting model
            quantum_optimizer: Quantum portfolio optimizer
        """
        self.ml_forecaster = ml_forecaster
        self.quantum_optimizer = quantum_optimizer
    
    def optimize_with_ml_forecasts(self, X_features: np.ndarray,
                                  historical_returns: np.ndarray,
                                  risk_aversion: float = 1.0) -> Dict:
        """
        Optimize portfolio using ML-predicted returns and quantum optimization.
        
        Args:
            X_features: Features for return prediction
            historical_returns: Historical returns for covariance estimation
            risk_aversion: Risk aversion parameter
            
        Returns:
            Optimization results with ML forecasts
        """
        # Generate return forecasts
        predictions = self.ml_forecaster.predict_returns(X_features)
        
        # Use ensemble prediction as expected returns
        mu_predicted = predictions['ensemble'][-1]  # Latest prediction
        
        # Estimate covariance from historical data
        sigma = np.cov(historical_returns.T)
        
        # Quantum optimization with ML inputs
        quantum_result = self.quantum_optimizer.optimize_portfolio(
            mu_predicted, sigma, risk_aversion=risk_aversion
        )
        
        # Add ML information to results
        quantum_result['ml_predictions'] = predictions
        quantum_result['predicted_returns'] = mu_predicted
        
        return quantum_result



# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_assets = 5
    n_periods = 1000
    
    # Simulate price data with trends and patterns
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Create synthetic price data with some patterns
    prices_data = np.zeros((n_periods, n_assets))
    for i in range(n_assets):
        trend = 0.0005 * np.arange(n_periods)
        noise = np.random.normal(0, 0.02, n_periods)
        seasonal = 0.001 * np.sin(2 * np.pi * np.arange(n_periods) / 252)  # Annual cycle
        prices_data[:, i] = 100 * np.exp(np.cumsum(trend + seasonal + noise))
    
    prices_df = pd.DataFrame(prices_data, index=dates, 
                           columns=[f'Asset_{i}' for i in range(n_assets)])
    
    print("ML-Enhanced Portfolio Optimization")
    print("=" * 40)
    
    # Setup ML forecasting
    config = ForecastingConfig(
        lookback_window=60,
        forecast_horizon=5,
        feature_engineering=True,
        ensemble_models=True
    )
    
    forecaster = MLReturnForecaster(config)
    
    # Prepare training data
    print("Preparing training data...")
    X, y = forecaster.prepare_training_data(prices_df)
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Split data for training/testing
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train ML models
    print("\nTraining ML models...")
    forecaster.train_models(X_train, y_train)
    
    # Generate predictions
    print("\nGenerating forecasts...")
    predictions = forecaster.predict_returns(X_test)
    
    # Evaluate forecasts
    metrics = forecaster.evaluate_forecasts(y_test, predictions)
    
    print("\nForecast Evaluation:")
    for model_name, metric in metrics.items():
        print(f"{model_name:15} - RMSE: {metric['rmse']:.6f}, "
              f"Direction Acc: {metric['direction_accuracy']:.3f}")
    
    print(f"\nBest model by RMSE: {min(metrics.keys(), key=lambda k: metrics[k]['rmse'])}")
