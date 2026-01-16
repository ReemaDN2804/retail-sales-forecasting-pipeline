"""
Forecasting Module

This module implements multiple forecasting models for retail sales prediction:
1. Baseline Moving Average Model
2. SARIMA (Seasonal ARIMA) Model
3. Random Forest Regression Model

Each model is designed for time-series aware forecasting with proper train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Optional, Dict, Any, List
import warnings
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineMovingAverage:
    """
    Baseline forecasting model using moving average.
    
    This simple model predicts future sales as the average of the last N days.
    It serves as a baseline to compare more sophisticated models against.
    
    Attributes:
        window (int): Number of days to average
        predictions (np.ndarray): Stored predictions
    """
    
    def __init__(self, window: int = 7):
        """
        Initialize Baseline Moving Average model.
        
        Args:
            window: Number of days to use for moving average (default: 7)
        """
        self.window = window
        self.predictions = None
        self.model_name = f"Moving Average (window={window})"
        logger.info(f"Initialized {self.model_name}")
    
    def fit(self, y_train: np.ndarray) -> 'BaselineMovingAverage':
        """
        Fit the model (no actual training needed for moving average).
        
        Args:
            y_train: Training data
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting {self.model_name}...")
        # No actual training needed
        return self
    
    def predict(self, y_train: np.ndarray, steps: int) -> np.ndarray:
        """
        Generate predictions using moving average.
        
        Args:
            y_train: Historical sales data
            steps: Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions
        """
        logger.info(f"Generating {steps} predictions using {self.model_name}...")
        
        predictions = []
        history = list(y_train)
        
        for i in range(steps):
            # Calculate moving average of last 'window' values
            if len(history) >= self.window:
                pred = np.mean(history[-self.window:])
            else:
                pred = np.mean(history)
            
            predictions.append(pred)
            history.append(pred)  # Use prediction for next step
        
        self.predictions = np.array(predictions)
        return self.predictions


class SARIMAForecaster:
    """
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) model.
    
    SARIMA extends ARIMA by adding seasonal components, making it suitable
    for data with seasonal patterns (e.g., weekly, monthly seasonality).
    
    Model parameters:
    - (p, d, q): Non-seasonal parameters (AR order, differencing, MA order)
    - (P, D, Q, s): Seasonal parameters (seasonal AR, differencing, MA, period)
    
    Attributes:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        model: Fitted SARIMAX model
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)):
        """
        Initialize SARIMA model.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
                          Default (1,1,1,7) for weekly seasonality
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.model_name = f"SARIMA{order}x{seasonal_order}"
        logger.info(f"Initialized {self.model_name}")
    
    def fit(self, y_train: np.ndarray) -> 'SARIMAForecaster':
        """
        Fit SARIMA model to training data.
        
        Args:
            y_train: Training data (time series)
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting {self.model_name}...")
        
        try:
            # Create and fit SARIMAX model
            self.model = SARIMAX(
                y_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = self.model.fit(disp=False, maxiter=200)
            
            logger.info(f"{self.model_name} fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {str(e)}")
            raise
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts for specified number of steps.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"Generating {steps} predictions using {self.model_name}...")
        
        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(steps=steps)
            return np.asarray(forecast)
            
        except Exception as e:
            logger.error(f"Error generating SARIMA predictions: {str(e)}")
            raise
    
    def get_model_summary(self) -> str:
        """
        Get model summary statistics.
        
        Returns:
            str: Model summary
        """
        if self.fitted_model is None:
            return "Model not fitted yet"
        
        return str(self.fitted_model.summary())


class RandomForestForecaster:
    """
    Random Forest Regression model for sales forecasting.
    
    Random Forest is an ensemble learning method that uses multiple decision
    trees to make predictions. It's robust to overfitting and can capture
    complex non-linear relationships in the data.
    
    For time series forecasting, we use engineered features (lags, rolling stats)
    as inputs to predict future sales.
    
    Attributes:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        model: Fitted RandomForestRegressor
        feature_importance: Feature importance scores
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 10,
                 random_state: int = 42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        self.model_name = f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth})"
        logger.info(f"Initialized {self.model_name}")
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'RandomForestForecaster':
        """
        Fit Random Forest model to training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting {self.model_name}...")
        
        try:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            
            self.model.fit(X_train, y_train)
            
            # Store feature importance
            self.feature_names = X_train.columns.tolist()
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"{self.model_name} fitted successfully")
            logger.info(f"Top 5 important features: {self.feature_importance.head()['feature'].tolist()}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting Random Forest model: {str(e)}")
            raise
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using fitted model.
        
        Args:
            X_test: Test features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"Generating predictions using {self.model_name}...")
        
        try:
            predictions = self.model.predict(X_test)
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating Random Forest predictions: {str(e)}")
            raise
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.feature_importance is None:
            raise ValueError("Model not fitted yet")
        
        return self.feature_importance.head(top_n)


def prepare_time_series_split(df: pd.DataFrame, 
                              target_column: str = 'sales',
                              test_size: float = 0.2,
                              date_column: str = 'date') -> Tuple:
    """
    Prepare time-series aware train/test split.
    
    Unlike random splitting, time series split maintains temporal order:
    - Training set: earlier time periods
    - Test set: later time periods
    
    This prevents data leakage and mimics real-world forecasting scenarios.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        test_size: Proportion of data for testing (default: 0.2)
        date_column: Name of date column
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test, train_dates, test_dates)
    """
    logger.info("Preparing time-series aware train/test split...")
    
    try:
        # Remove rows with NaN in target
        df_clean = df.dropna(subset=[target_column])
        
        # Sort by date to ensure temporal order
        df_clean = df_clean.sort_values(date_column).reset_index(drop=True)
        
        # Calculate split index
        split_idx = int(len(df_clean) * (1 - test_size))
        
        # Get feature columns (exclude target and date)
        feature_cols = [col for col in df_clean.columns 
                       if col not in [target_column, date_column] 
                       and not df_clean[col].isnull().all()]
        
        # Remove columns with all NaN values
        feature_cols = [col for col in feature_cols 
                       if df_clean[col].notna().sum() > 0]
        
        # Split data
        train_data = df_clean.iloc[:split_idx]
        test_data = df_clean.iloc[split_idx:]
        
        # Prepare features and target
        X_train = train_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
        X_test = test_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
        y_train = train_data[target_column].values
        y_test = test_data[target_column].values
        
        train_dates = train_data[date_column]
        test_dates = test_data[date_column]
        
        logger.info(f"Train set: {len(X_train)} samples ({train_dates.min()} to {train_dates.max()})")
        logger.info(f"Test set: {len(X_test)} samples ({test_dates.min()} to {test_dates.max()})")
        logger.info(f"Number of features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
        
    except Exception as e:
        logger.error(f"Error preparing train/test split: {str(e)}")
        raise


def forecast_next_n_days(model: Any, 
                         last_known_data: pd.DataFrame,
                         n_days: int = 30,
                         model_type: str = 'rf') -> np.ndarray:
    """
    Forecast sales for the next N days.
    
    Args:
        model: Fitted forecasting model
        last_known_data: Most recent data with features
        n_days: Number of days to forecast
        model_type: Type of model ('rf', 'sarima', 'ma')
        
    Returns:
        np.ndarray: Forecasted sales for next N days
    """
    logger.info(f"Forecasting next {n_days} days using {model_type.upper()} model...")
    
    try:
        if model_type == 'sarima':
            # SARIMA can directly forecast multiple steps
            forecast = model.predict(steps=n_days)
            
        elif model_type == 'ma':
            # Moving average uses historical data
            y_history = last_known_data['sales'].values
            forecast = model.predict(y_history, steps=n_days)
            
        elif model_type == 'rf':
            # Random Forest requires feature engineering for future dates
            # For simplicity, use last known features repeated
            # In production, you'd engineer features for future dates
            last_features = last_known_data.iloc[-1:][model.feature_names]
            forecast = []
            
            for _ in range(n_days):
                pred = model.predict(last_features)[0]
                forecast.append(pred)
                # In production, update features based on prediction
            
            forecast = np.array(forecast)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Forecast generated: mean={forecast.mean():.2f}, std={forecast.std():.2f}")
        return forecast
        
    except Exception as e:
        logger.error(f"Error forecasting next {n_days} days: {str(e)}")
        raise


def train_all_models(df: pd.DataFrame, 
                     target_column: str = 'sales',
                     test_size: float = 0.2) -> Dict[str, Dict]:
    """
    Train all forecasting models and return results.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        test_size: Proportion of data for testing
        
    Returns:
        Dict: Dictionary containing all models and their predictions
    """
    logger.info("="*70)
    logger.info("TRAINING ALL FORECASTING MODELS")
    logger.info("="*70)
    
    results = {}
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, train_dates, test_dates = prepare_time_series_split(
            df, target_column, test_size
        )
        
        # 1. Baseline Moving Average
        logger.info("\n" + "-"*70)
        logger.info("Training Baseline Moving Average Model")
        logger.info("-"*70)
        ma_model = BaselineMovingAverage(window=7)
        ma_model.fit(y_train)
        ma_predictions = ma_model.predict(y_train, steps=len(y_test))
        
        results['moving_average'] = {
            'model': ma_model,
            'predictions': ma_predictions,
            'actuals': y_test,
            'dates': test_dates
        }
        logger.info("✓ Moving Average model trained")
        
        # 2. SARIMA Model
        logger.info("\n" + "-"*70)
        logger.info("Training SARIMA Model")
        logger.info("-"*70)
        try:
            sarima_model = SARIMAForecaster(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7)
            )
            sarima_model.fit(y_train)
            sarima_predictions = sarima_model.predict(steps=len(y_test))
            
            results['sarima'] = {
                'model': sarima_model,
                'predictions': sarima_predictions,
                'actuals': y_test,
                'dates': test_dates
            }
            logger.info("✓ SARIMA model trained")
        except Exception as e:
            logger.warning(f"SARIMA model failed: {str(e)}. Skipping...")
        
        # 3. Random Forest
        logger.info("\n" + "-"*70)
        logger.info("Training Random Forest Model")
        logger.info("-"*70)
        rf_model = RandomForestForecaster(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        # IMPORTANT: Random Forest accepts ONLY numeric features
        X_train_rf = X_train.select_dtypes(include=[np.number])
        X_test_rf = X_test.select_dtypes(include=[np.number])

        rf_model.fit(X_train_rf, y_train)
        rf_predictions = rf_model.predict(X_test_rf)

        
        results['random_forest'] = {
            'model': rf_model,
            'predictions': rf_predictions,
            'actuals': y_test,
            'dates': test_dates,
            'feature_importance': rf_model.get_feature_importance()
        }
        logger.info("✓ Random Forest model trained")
        
        logger.info("\n" + "="*70)
        logger.info(f"ALL MODELS TRAINED SUCCESSFULLY ({len(results)} models)")
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader, create_sample_data
    from feature_engineering import engineer_features
    from pathlib import Path
    
    # Create and load sample data
    sample_path = Path("data/raw/sample_sales_data.csv")
    if not sample_path.exists():
        create_sample_data(sample_path, n_records=730)
    
    loader = DataLoader(sample_path)
    df = loader.load_data()
    
    # Engineer features
    df_features = engineer_features(df, save_output=False)
    
    # Train all models
    results = train_all_models(df_features)
    
    print("\n" + "="*70)
    print("FORECASTING RESULTS")
    print("="*70)
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Predictions shape: {model_results['predictions'].shape}")
        print(f"  First 5 predictions: {model_results['predictions'][:5]}")