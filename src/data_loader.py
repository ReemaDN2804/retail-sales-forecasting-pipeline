"""
Data Loader Module

This module handles loading, validation, and initial preprocessing of retail sales data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class for loading and validating retail sales data.
    
    Attributes:
        data_path (Path): Path to the data file
        df (pd.DataFrame): Loaded dataframe
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize DataLoader with path to data file.
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, date_column: str = 'date', 
                  sales_column: str = 'sales') -> pd.DataFrame:
        """
        Load data from file and perform basic validation.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load data based on file extension
        if self.data_path.suffix == '.csv':
            try:
                self.df = pd.read_csv(self.data_path, encoding="utf-8")
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.data_path, encoding="latin1")
        elif self.data_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Clean non-breaking spaces
        self.df = self.df.applymap(
            lambda x: x.replace('\xa0', ' ') if isinstance(x, str) else x
        )

        # Normalize column names
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
        )

        logger.info(f"Loaded data with shape: {self.df.shape}")

        # Auto-detect date column
        if date_column not in self.df.columns:
            possible_date_cols = [
                col for col in self.df.columns
                if 'date' in col or 'time' in col
            ]
            if possible_date_cols:
                detected_col = possible_date_cols[0]
                logger.warning(
                    f"Date column '{date_column}' not found. "
                    f"Using '{detected_col}' instead."
                )
                # Rename detected column to 'date' for pipeline consistency
                self.df = self.df.rename(columns={detected_col: date_column})


        # Validate required columns
        if date_column not in self.df.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        if sales_column not in self.df.columns:
            raise ValueError(f"Sales column '{sales_column}' not found in data")

        # Convert date column to datetime
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors='coerce')

        # Drop rows with invalid dates
        self.df = self.df.dropna(subset=[date_column])

        # Sort by date
        self.df = self.df.sort_values(date_column).reset_index(drop=True)

        logger.info("Data loaded and validated successfully")
        return self.df

    def get_data_info(self) -> dict:
        """
        Get summary information about the loaded data.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,
        }
    
    def handle_missing_values(self, strategy: str = 'drop', 
                              fill_value: Optional[float] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'forward_fill':
            self.df = self.df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            self.df = self.df.fillna(method='bfill')
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(
                self.df[numeric_cols].mean()
            )
        elif strategy == 'custom' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        
        logger.info(f"Missing values handled using strategy: {strategy}")
        return self.df
    
    def save_processed_data(self, output_path: Union[str, Path]) -> None:
        """
        Save processed data to file.
        """
        if self.df is None:
            raise ValueError("No data to save. Load data first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        logger.info(f"Processed data saved to: {output_path}")


def create_sample_data(output_path: Union[str, Path], 
                       n_records: int = 365) -> pd.DataFrame:
    """
    Create sample retail sales data for testing purposes.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_records, freq='D')
    
    trend = np.linspace(100, 150, n_records)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_records) / 365)
    noise = np.random.normal(0, 10, n_records)
    sales = trend + seasonality + noise
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], n_records),
        'store_id': np.random.choice(['Store_A', 'Store_B', 'Store_C'], n_records),
        'promotion': np.random.choice([0, 1], n_records, p=[0.8, 0.2])
    })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data created and saved to: {output_path}")
    return df
