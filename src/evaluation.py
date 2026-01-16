"""
Evaluation Module

This module provides comprehensive model evaluation metrics and comparison tools:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Model comparison and ranking
- Results export to CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    Comprehensive evaluation of forecasting models.
    
    This class calculates multiple metrics to assess forecast accuracy:
    - MAE: Average absolute difference between predicted and actual
    - RMSE: Square root of average squared errors (penalizes large errors)
    - MAPE: Average percentage error (scale-independent)
    
    Attributes:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        metrics (Dict): Calculated metrics
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
        """
        Initialize evaluator with actual and predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model being evaluated
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.model_name = model_name
        self.metrics = {}
        
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        logger.info(f"Initialized evaluator for {model_name} with {len(y_true)} samples")
    
    def calculate_mae(self) -> float:
        """
        Calculate Mean Absolute Error.
        
        MAE measures the average magnitude of errors without considering direction.
        Lower values indicate better performance.
        
        Formula: MAE = mean(|actual - predicted|)
        
        Returns:
            float: MAE value
        """
        mae = np.mean(np.abs(self.y_true - self.y_pred))
        self.metrics['MAE'] = mae
        return mae
    
    def calculate_rmse(self) -> float:
        """
        Calculate Root Mean Squared Error.
        
        RMSE gives more weight to large errors due to squaring.
        It's in the same units as the target variable.
        
        Formula: RMSE = sqrt(mean((actual - predicted)^2))
        
        Returns:
            float: RMSE value
        """
        mse = np.mean((self.y_true - self.y_pred) ** 2)
        rmse = np.sqrt(mse)
        self.metrics['RMSE'] = rmse
        return rmse
    
    def calculate_mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        MAPE expresses error as a percentage, making it scale-independent
        and easy to interpret across different datasets.
        
        Formula: MAPE = mean(|actual - predicted| / |actual|) * 100
        
        Returns:
            float: MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = self.y_true != 0
        if mask.sum() == 0:
            logger.warning("All actual values are zero. MAPE cannot be calculated.")
            self.metrics['MAPE'] = np.inf
            return np.inf
        
        mape = np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
        self.metrics['MAPE'] = mape
        return mape
    
    def calculate_r2_score(self) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        R² indicates the proportion of variance in the dependent variable
        that is predictable from the independent variables.
        
        Values range from -∞ to 1:
        - 1: Perfect prediction
        - 0: Model performs as well as predicting the mean
        - <0: Model performs worse than predicting the mean
        
        Returns:
            float: R² value
        """
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        
        if ss_tot == 0:
            r2 = 0.0
        else:
            r2 = 1 - (ss_res / ss_tot)
        
        self.metrics['R2'] = r2
        return r2
    
    def calculate_mse(self) -> float:
        """
        Calculate Mean Squared Error.
        
        MSE is the average of squared errors. It's more sensitive to outliers
        than MAE due to squaring.
        
        Returns:
            float: MSE value
        """
        mse = np.mean((self.y_true - self.y_pred) ** 2)
        self.metrics['MSE'] = mse
        return mse
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all evaluation metrics at once.
        
        Returns:
            Dict: Dictionary containing all metrics
        """
        logger.info(f"Calculating all metrics for {self.model_name}...")
        
        try:
            self.calculate_mae()
            self.calculate_rmse()
            self.calculate_mape()
            self.calculate_r2_score()
            self.calculate_mse()
            
            logger.info(f"Metrics calculated for {self.model_name}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get metrics as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Metrics in DataFrame format
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        df = pd.DataFrame([self.metrics])
        df.insert(0, 'Model', self.model_name)
        return df
    
    def print_metrics(self) -> None:
        """
        Print metrics in a formatted table.
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        print("\n" + "="*60)
        print(f"EVALUATION METRICS: {self.model_name}")
        print("="*60)
        
        for metric_name, value in self.metrics.items():
            if metric_name == 'MAPE':
                print(f"{metric_name:.<20} {value:.2f}%")
            else:
                print(f"{metric_name:.<20} {value:.4f}")
        
        print("="*60 + "\n")


class ModelComparator:
    """
    Compare multiple forecasting models based on their performance metrics.
    
    This class aggregates results from multiple models and provides
    ranking and comparison functionality.
    
    Attributes:
        results (Dict): Dictionary of model results
        comparison_df (pd.DataFrame): Comparison table
    """
    
    def __init__(self):
        """Initialize ModelComparator."""
        self.results = {}
        self.comparison_df = None
        logger.info("ModelComparator initialized")
    
    def add_model_results(self, model_name: str, y_true: np.ndarray, 
                         y_pred: np.ndarray) -> None:
        """
        Add a model's results for comparison.
        
        Args:
            model_name: Name of the model
            y_true: Actual values
            y_pred: Predicted values
        """
        logger.info(f"Adding results for {model_name}...")
        
        evaluator = ForecastEvaluator(y_true, y_pred, model_name)
        metrics = evaluator.calculate_all_metrics()
        
        self.results[model_name] = {
            'evaluator': evaluator,
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table of all models.
        
        Returns:
            pd.DataFrame: Comparison table with all metrics
        """
        logger.info("Generating model comparison table...")
        
        if not self.results:
            raise ValueError("No model results added. Use add_model_results() first.")
        
        comparison_data = []
        
        for model_name, data in self.results.items():
            row = {'Model': model_name}
            row.update(data['metrics'])
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        self.comparison_df = self.comparison_df.sort_values('RMSE')
        
        logger.info("Comparison table generated")
        return self.comparison_df
    
    def get_best_model(self, metric: str = 'RMSE') -> Tuple[str, float]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for ranking ('MAE', 'RMSE', 'MAPE')
                   Lower values are better
            
        Returns:
            Tuple: (model_name, metric_value)
        """
        if self.comparison_df is None:
            self.generate_comparison_table()
        
        if metric not in self.comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        best_idx = self.comparison_df[metric].idxmin()
        best_model = self.comparison_df.loc[best_idx, 'Model']
        best_value = self.comparison_df.loc[best_idx, metric]
        
        logger.info(f"Best model by {metric}: {best_model} ({metric}={best_value:.4f})")
        return best_model, best_value
    
    def print_comparison(self) -> None:
        """
        Print formatted comparison table.
        """
        if self.comparison_df is None:
            self.generate_comparison_table()
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(self.comparison_df.to_string(index=False))
        print("="*80)
        
        # Print best models
        print("\nBEST MODELS BY METRIC:")
        print("-"*80)
        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in self.comparison_df.columns:
                best_model, best_value = self.get_best_model(metric)
                if metric == 'MAPE':
                    print(f"{metric:.<20} {best_model} ({best_value:.2f}%)")
                else:
                    print(f"{metric:.<20} {best_model} ({best_value:.4f})")
        print("="*80 + "\n")
    
    def save_comparison(self, output_path: str = 'outputs/reports/model_comparison.csv') -> None:
        """
        Save comparison table to CSV file.
        
        Args:
            output_path: Path to save the comparison table
        """
        if self.comparison_df is None:
            self.generate_comparison_table()
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.comparison_df.to_csv(output_path, index=False)
            logger.info(f"Comparison table saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving comparison table: {str(e)}")
            raise


def evaluate_models(model_results: Dict[str, Dict], 
                   save_results: bool = True,
                   output_dir: str = 'outputs/reports') -> pd.DataFrame:
    """
    Convenience function to evaluate multiple models and save results.
    
    Args:
        model_results: Dictionary with model names as keys and results as values
                      Each result should contain 'predictions' and 'actuals'
        save_results: Whether to save results to CSV
        output_dir: Directory to save results
        
    Returns:
        pd.DataFrame: Comparison table
    """
    logger.info("="*70)
    logger.info("EVALUATING ALL MODELS")
    logger.info("="*70)
    
    comparator = ModelComparator()
    
    # Add all model results
    for model_name, results in model_results.items():
        comparator.add_model_results(
            model_name,
            results['actuals'],
            results['predictions']
        )
    
    # Generate and print comparison
    comparison_df = comparator.generate_comparison_table()
    comparator.print_comparison()
    
    # Save results
    if save_results:
        output_path = Path(output_dir) / 'model_comparison.csv'
        comparator.save_comparison(output_path)
        
        # Save detailed metrics for each model
        for model_name, data in comparator.results.items():
            metrics_df = data['evaluator'].get_metrics_dataframe()
            metrics_path = Path(output_dir) / f'{model_name}_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Detailed metrics for {model_name} saved to: {metrics_path}")
    
    logger.info("="*70)
    logger.info("MODEL EVALUATION COMPLETED")
    logger.info("="*70)
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample predictions
    n_samples = 100
    y_true = np.random.randn(n_samples) * 10 + 100
    
    # Simulate different model predictions
    model_results = {
        'Model_A': {
            'predictions': y_true + np.random.randn(n_samples) * 5,
            'actuals': y_true
        },
        'Model_B': {
            'predictions': y_true + np.random.randn(n_samples) * 8,
            'actuals': y_true
        },
        'Model_C': {
            'predictions': y_true + np.random.randn(n_samples) * 3,
            'actuals': y_true
        }
    }
    
    # Evaluate models
    comparison = evaluate_models(model_results, save_results=False)
    
    print("\nComparison DataFrame:")
    print(comparison)