"""
Feature Engineering Module

This module creates comprehensive features for retail sales forecasting:
- Time-based features (day, week, month, quarter, year)
- Lag features (historical sales values)
- Rolling statistics (moving averages and standard deviations)
- Categorical encoding
- Interaction features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SalesFeatureEngineer:
    """
    Comprehensive feature engineering for retail sales forecasting.
    
    This class creates various types of features that capture temporal patterns,
    historical trends, and categorical information to improve forecasting accuracy.
    
    Key Features Created:
    1. Time-based: day, month, quarter, year, day_of_week, is_weekend
    2. Lag features: sales values from previous periods (7, 14, 30 days)
    3. Rolling statistics: moving averages and standard deviations
    4. Categorical encoding: one-hot encoding for categories
    
    Attributes:
        df (pd.DataFrame): The retail sales dataframe
        date_column (str): Name of the date column
        sales_column (str): Name of the sales/target column
    """
    
    def __init__(self, df: pd.DataFrame, date_column: str = 'date', 
                 sales_column: str = 'sales'):
        """
        Initialize SalesFeatureEngineer with sales data.
        
        Args:
            df: DataFrame containing retail sales data
            date_column: Name of the date column
            sales_column: Name of the sales column
        """
        self.df = df.copy()
        self.date_column = date_column
        self.sales_column = sales_column
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            self.df[date_column] = pd.to_datetime(self.df[date_column])
        
        # Sort by date to ensure proper lag and rolling calculations
        self.df = self.df.sort_values(date_column).reset_index(drop=True)
        
        self.feature_names = []  # Track created features
        
        logger.info(f"FeatureEngineer initialized with {len(self.df)} records")
    
    def create_time_features(self) -> pd.DataFrame:
        """
        Create comprehensive time-based features from the date column.
        
        Time features help capture cyclical patterns and seasonality:
        - year: captures long-term trends
        - month: captures monthly seasonality (1-12)
        - quarter: captures quarterly patterns (1-4)
        - day: day of month (1-31)
        - day_of_week: weekday patterns (0=Monday, 6=Sunday)
        - week_of_year: weekly patterns (1-52)
        - is_weekend: binary indicator for Saturday/Sunday
        - is_month_start/end: captures beginning/end of month effects
        - is_quarter_start/end: captures quarterly boundary effects
        
        Returns:
            pd.DataFrame: DataFrame with added time features
        """
        logger.info("Creating time-based features...")
        
        try:
            # Basic time components
            self.df['year'] = self.df[self.date_column].dt.year
            self.df['month'] = self.df[self.date_column].dt.month
            self.df['quarter'] = self.df[self.date_column].dt.quarter
            self.df['day'] = self.df[self.date_column].dt.day
            self.df['day_of_week'] = self.df[self.date_column].dt.dayofweek
            self.df['week_of_year'] = self.df[self.date_column].dt.isocalendar().week.astype(int)
            self.df['day_of_year'] = self.df[self.date_column].dt.dayofyear
            
            # Weekend indicator (Saturday=5, Sunday=6)
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
            
            # Month boundary indicators
            self.df['is_month_start'] = self.df[self.date_column].dt.is_month_start.astype(int)
            self.df['is_month_end'] = self.df[self.date_column].dt.is_month_end.astype(int)
            
            # Quarter boundary indicators
            self.df['is_quarter_start'] = self.df[self.date_column].dt.is_quarter_start.astype(int)
            self.df['is_quarter_end'] = self.df[self.date_column].dt.is_quarter_end.astype(int)
            
            # Cyclical encoding for month (captures circular nature: Dec -> Jan)
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
            
            # Cyclical encoding for day of week
            self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
            
            time_features = ['year', 'month', 'quarter', 'day', 'day_of_week', 
                           'week_of_year', 'day_of_year', 'is_weekend', 
                           'is_month_start', 'is_month_end', 'is_quarter_start', 
                           'is_quarter_end', 'month_sin', 'month_cos', 
                           'day_of_week_sin', 'day_of_week_cos']
            
            self.feature_names.extend(time_features)
            
            logger.info(f"Created {len(time_features)} time-based features")
            return self.df
            
        except Exception as e:
            logger.error(f"Error creating time features: {str(e)}")
            raise
    
    def create_lag_features(self, lags: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create lag features (previous sales values).
        
        Lag features capture historical sales patterns:
        - lag_7: sales from 1 week ago (captures weekly patterns)
        - lag_14: sales from 2 weeks ago (captures bi-weekly patterns)
        - lag_30: sales from ~1 month ago (captures monthly patterns)
        
        These features are crucial for time series forecasting as they
        allow the model to learn from historical values.
        
        Args:
            lags: List of lag periods (in days) to create
            
        Returns:
            pd.DataFrame: DataFrame with added lag features
        """
        logger.info(f"Creating lag features for periods: {lags}...")
        
        try:
            for lag in lags:
                feature_name = f'{self.sales_column}_lag_{lag}'
                self.df[feature_name] = self.df[self.sales_column].shift(lag)
                self.feature_names.append(feature_name)
            
            logger.info(f"Created {len(lags)} lag features")
            return self.df
            
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def create_rolling_features(self, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling window statistics features.
        
        Rolling features capture recent trends and volatility:
        - rolling_mean_X: average sales over last X days (trend indicator)
        - rolling_std_X: sales volatility over last X days (stability indicator)
        - rolling_min_X: minimum sales in last X days
        - rolling_max_X: maximum sales in last X days
        
        These features help the model understand recent sales momentum
        and variability patterns.
        
        Args:
            windows: List of window sizes (in days) for rolling calculations
            
        Returns:
            pd.DataFrame: DataFrame with added rolling features
        """
        logger.info(f"Creating rolling features for windows: {windows}...")
        
        try:
            for window in windows:
                # Rolling mean (trend indicator)
                feature_name = f'{self.sales_column}_rolling_mean_{window}'
                self.df[feature_name] = self.df[self.sales_column].rolling(
                    window=window, min_periods=1
                ).mean()
                self.feature_names.append(feature_name)
                
                # Rolling standard deviation (volatility indicator)
                feature_name = f'{self.sales_column}_rolling_std_{window}'
                self.df[feature_name] = self.df[self.sales_column].rolling(
                    window=window, min_periods=1
                ).std()
                self.feature_names.append(feature_name)
                
                # Rolling min
                feature_name = f'{self.sales_column}_rolling_min_{window}'
                self.df[feature_name] = self.df[self.sales_column].rolling(
                    window=window, min_periods=1
                ).min()
                self.feature_names.append(feature_name)
                
                # Rolling max
                feature_name = f'{self.sales_column}_rolling_max_{window}'
                self.df[feature_name] = self.df[self.sales_column].rolling(
                    window=window, min_periods=1
                ).max()
                self.feature_names.append(feature_name)
            
            logger.info(f"Created {len(windows) * 4} rolling features")
            return self.df
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {str(e)}")
            raise
    
    def create_exponential_moving_average(self, spans: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create exponential moving average features.
        
        EMA gives more weight to recent observations, making it more
        responsive to recent changes compared to simple moving average.
        
        Args:
            spans: List of span values for EMA calculation
            
        Returns:
            pd.DataFrame: DataFrame with added EMA features
        """
        logger.info(f"Creating EMA features for spans: {spans}...")
        
        try:
            for span in spans:
                feature_name = f'{self.sales_column}_ema_{span}'
                self.df[feature_name] = self.df[self.sales_column].ewm(
                    span=span, adjust=False
                ).mean()
                self.feature_names.append(feature_name)
            
            logger.info(f"Created {len(spans)} EMA features")
            return self.df
            
        except Exception as e:
            logger.error(f"Error creating EMA features: {str(e)}")
            raise
    
    def encode_categorical_features(self, categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Categorical features like store_id, category, etc. are converted
        to binary indicator variables. This allows the model to learn
        different patterns for different categories.
        
        Args:
            categorical_columns: List of categorical column names to encode
                                If None, automatically detect object/category columns
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        try:
            if categorical_columns is None:
                # Auto-detect categorical columns (excluding date column)
                categorical_columns = self.df.select_dtypes(
                    include=['object', 'category']
                ).columns.tolist()
                
                # Remove date column if present
                if self.date_column in categorical_columns:
                    categorical_columns.remove(self.date_column)
            
            if not categorical_columns:
                logger.info("No categorical columns found to encode")
                return self.df
            
            # One-hot encode categorical variables
            for col in categorical_columns:
                if col in self.df.columns:
                    # Create dummy variables
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    
                    # Add to dataframe
                    self.df = pd.concat([self.df, dummies], axis=1)
                    
                    # Track feature names
                    self.feature_names.extend(dummies.columns.tolist())
                    
                    logger.info(f"Encoded '{col}' into {len(dummies.columns)} features")
            
            logger.info(f"Categorical encoding completed for {len(categorical_columns)} columns")
            return self.df
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {str(e)}")
            raise
    
    def create_interaction_features(self) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Interaction features capture combined effects:
        - weekend * month: different weekend patterns across months
        - promotion * day_of_week: promotion effectiveness by weekday
        
        Returns:
            pd.DataFrame: DataFrame with added interaction features
        """
        logger.info("Creating interaction features...")
        
        try:
            # Weekend-Month interaction
            if 'is_weekend' in self.df.columns and 'month' in self.df.columns:
                self.df['weekend_month'] = self.df['is_weekend'] * self.df['month']
                self.feature_names.append('weekend_month')
            
            # Promotion-DayOfWeek interaction (if promotion column exists)
            if 'promotion' in self.df.columns and 'day_of_week' in self.df.columns:
                self.df['promotion_dow'] = self.df['promotion'] * self.df['day_of_week']
                self.feature_names.append('promotion_dow')
            
            # Weekend-Quarter interaction
            if 'is_weekend' in self.df.columns and 'quarter' in self.df.columns:
                self.df['weekend_quarter'] = self.df['is_weekend'] * self.df['quarter']
                self.feature_names.append('weekend_quarter')
            
            logger.info("Interaction features created")
            return self.df
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
            raise
    
    def create_all_features(self, 
                           lags: List[int] = [7, 14, 30],
                           windows: List[int] = [7, 14, 30],
                           ema_spans: List[int] = [7, 14, 30],
                           categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create all feature types in the proper sequence.
        
        Feature creation order matters:
        1. Time features (no dependencies)
        2. Lag features (depend on sorted data)
        3. Rolling features (depend on sorted data)
        4. EMA features (depend on sorted data)
        5. Categorical encoding
        6. Interaction features (depend on other features)
        
        Args:
            lags: List of lag periods for lag features
            windows: List of window sizes for rolling features
            ema_spans: List of spans for EMA features
            categorical_columns: List of categorical columns to encode
            
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        logger.info("="*70)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("="*70)
        
        try:
            initial_cols = len(self.df.columns)
            
            # Create features in sequence
            self.create_time_features()
            self.create_lag_features(lags)
            self.create_rolling_features(windows)
            self.create_exponential_moving_average(ema_spans)
            self.encode_categorical_features(categorical_columns)
            self.create_interaction_features()
            
            final_cols = len(self.df.columns)
            features_created = final_cols - initial_cols
            
            logger.info("="*70)
            logger.info("FEATURE ENGINEERING COMPLETED")
            logger.info(f"Initial columns: {initial_cols}")
            logger.info(f"Final columns: {final_cols}")
            logger.info(f"Features created: {features_created}")
            logger.info(f"Total feature names tracked: {len(self.feature_names)}")
            logger.info("="*70)
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """
        Get the dataframe with all engineered features.
        
        Returns:
            pd.DataFrame: DataFrame with features
        """
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all created feature names.
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names
    
    def save_features(self, output_path: str = 'data/processed/feature_data.csv') -> None:
        """
        Save the feature-engineered dataset to CSV.
        
        Args:
            output_path: Path where to save the processed data
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.df.to_csv(output_path, index=False)
            logger.info(f"Feature data saved to: {output_path}")
            logger.info(f"Saved {len(self.df)} rows and {len(self.df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Error saving feature data: {str(e)}")
            raise


def engineer_features(df: pd.DataFrame, 
                     date_column: str = 'date',
                     sales_column: str = 'sales',
                     save_output: bool = True,
                     output_path: str = 'data/processed/feature_data.csv') -> pd.DataFrame:
    """
    Convenience function to engineer all features from sales data.
    
    Args:
        df: DataFrame containing retail sales data
        date_column: Name of the date column
        sales_column: Name of the sales column
        save_output: Whether to save the processed data
        output_path: Path to save processed data
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    engineer = SalesFeatureEngineer(df, date_column, sales_column)
    df_features = engineer.create_all_features()
    
    if save_output:
        engineer.save_features(output_path)
    
    return df_features


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader, create_sample_data
    
    # Create sample data if needed
    sample_path = Path("data/raw/sample_sales_data.csv")
    if not sample_path.exists():
        create_sample_data(sample_path, n_records=730)
    
    # Load data
    loader = DataLoader(sample_path)
    df = loader.load_data()
    
    # Engineer features
    df_features = engineer_features(df)
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*70)
    print(f"Original shape: {df.shape}")
    print(f"Final shape: {df_features.shape}")
    print(f"\nFirst few rows:")
    print(df_features.head())
    print(f"\nFeature columns: {list(df_features.columns)}")