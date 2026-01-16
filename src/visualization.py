"""
Visualization Module

This module provides comprehensive visualization functions for:
- Actual vs Predicted sales comparison
- Forecast trends with confidence intervals
- Residual analysis
- Feature importance plots
- Model comparison charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


class ForecastVisualizer:
    """
    Comprehensive visualization tools for forecasting results.
    
    This class provides methods to create various plots for analyzing
    and presenting forecasting model results.
    
    Attributes:
        output_dir (Path): Directory to save visualization outputs
        figsize (Tuple): Default figure size for plots
    """
    
    def __init__(self, output_dir: str = 'outputs/figures', figsize: Tuple = (14, 6)):
        """
        Initialize ForecastVisualizer.
        
        Args:
            output_dir: Directory to save output figures
            figsize: Default figure size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        logger.info(f"ForecastVisualizer initialized. Outputs will be saved to: {self.output_dir}")
    
    def plot_actual_vs_predicted(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 dates: Optional[pd.Series] = None,
                                 model_name: str = "Model",
                                 save_filename: Optional[str] = None) -> None:
        """
        Plot actual vs predicted sales over time.
        
        This visualization shows how well the model's predictions align
        with actual sales values over the test period.
        
        Args:
            y_true: Actual sales values
            y_pred: Predicted sales values
            dates: Date index for x-axis (optional)
            model_name: Name of the model
            save_filename: Filename to save plot (optional)
        """
        logger.info(f"Creating actual vs predicted plot for {model_name}...")
        
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Determine x-axis values
            if dates is not None:
                x = dates.reset_index(drop=True)
                xlabel = 'Date'
            else:
                x = np.arange(len(y_true))
                xlabel = 'Time Period'
            
            # Plot actual and predicted
            ax.plot(x, y_true, label='Actual Sales', 
                   linewidth=2, alpha=0.8, color='blue', marker='o', markersize=3)
            ax.plot(x, y_pred, label='Predicted Sales', 
                   linewidth=2, alpha=0.8, color='red', linestyle='--', marker='s', markersize=3)
            
            # Formatting
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
            ax.set_ylabel('Sales', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name}: Actual vs Predicted Sales', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if dates
            if dates is not None:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save plot
            if save_filename:
                save_path = self.output_dir / save_filename
            else:
                save_path = self.output_dir / f'{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating actual vs predicted plot: {str(e)}")
            raise
    
    def plot_forecast_trend(self,
                           historical_data: np.ndarray,
                           forecast: np.ndarray,
                           historical_dates: Optional[pd.Series] = None,
                           forecast_dates: Optional[pd.Series] = None,
                           model_name: str = "Model",
                           save_filename: Optional[str] = None) -> None:
        """
        Plot historical data with future forecast.
        
        This shows the continuation of the sales trend into the future,
        helping visualize the forecasted trajectory.
        
        Args:
            historical_data: Historical sales values
            forecast: Forecasted sales values
            historical_dates: Dates for historical data (optional)
            forecast_dates: Dates for forecast period (optional)
            model_name: Name of the model
            save_filename: Filename to save plot (optional)
        """
        logger.info(f"Creating forecast trend plot for {model_name}...")
        
        try:
            fig, ax = plt.subplots(figsize=(16, 6))
            
            # Determine x-axis values
            if historical_dates is not None and forecast_dates is not None:
                x_hist = historical_dates
                x_forecast = forecast_dates
                xlabel = 'Date'
            else:
                x_hist = np.arange(len(historical_data))
                x_forecast = np.arange(len(historical_data), len(historical_data) + len(forecast))
                xlabel = 'Time Period'
            
            # Plot historical data
            ax.plot(x_hist, historical_data, label='Historical Sales',
                   linewidth=2, alpha=0.8, color='blue')
            
            # Plot forecast
            ax.plot(x_forecast, forecast, label='Forecast',
                   linewidth=2, alpha=0.8, color='red', linestyle='--', marker='o', markersize=4)
            
            # Add vertical line at forecast start
            if historical_dates is not None:
                ax.axvline(x=x_hist.iloc[-1], color='green', linestyle=':', 
                          linewidth=2, label='Forecast Start')
            else:
                ax.axvline(x=len(historical_data)-1, color='green', linestyle=':', 
                          linewidth=2, label='Forecast Start')
            
            # Formatting
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
            ax.set_ylabel('Sales', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name}: Sales Forecast', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            if historical_dates is not None:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save plot
            if save_filename:
                save_path = self.output_dir / save_filename
            else:
                save_path = self.output_dir / f'{model_name.lower().replace(" ", "_")}_forecast_trend.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating forecast trend plot: {str(e)}")
            raise
    
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      dates: Optional[pd.Series] = None,
                      model_name: str = "Model",
                      save_filename: Optional[str] = None) -> None:
        """
        Plot residual analysis (errors).
        
        Residuals help identify patterns in prediction errors:
        - Random scatter: good model
        - Patterns: model missing something
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Date index (optional)
            model_name: Name of the model
            save_filename: Filename to save plot (optional)
        """
        logger.info(f"Creating residuals plot for {model_name}...")
        
        try:
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Residuals over time
            if dates is not None:
                x = dates.reset_index(drop=True)
                axes[0, 0].set_xlabel('Date')
            else:
                x = np.arange(len(residuals))
                axes[0, 0].set_xlabel('Time Period')
            
            axes[0, 0].scatter(x, residuals, alpha=0.6, s=30, color='steelblue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[0, 0].set_ylabel('Residuals', fontweight='bold')
            axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Residuals distribution
            axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Residuals', fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontweight='bold')
            axes[0, 1].set_title('Residuals Distribution', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 3. Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Residuals vs Predicted
            axes[1, 1].scatter(y_pred, residuals, alpha=0.6, s=30, color='steelblue')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Predicted Values', fontweight='bold')
            axes[1, 1].set_ylabel('Residuals', fontweight='bold')
            axes[1, 1].set_title('Residuals vs Predicted', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            fig.suptitle(f'{model_name}: Residual Analysis', 
                        fontsize=16, fontweight='bold', y=1.00)
            
            plt.tight_layout()
            
            # Save plot
            if save_filename:
                save_path = self.output_dir / save_filename
            else:
                save_path = self.output_dir / f'{model_name.lower().replace(" ", "_")}_residuals.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating residuals plot: {str(e)}")
            raise
    
    def plot_model_comparison(self,
                             comparison_df: pd.DataFrame,
                             metrics: List[str] = ['MAE', 'RMSE', 'MAPE'],
                             save_filename: str = 'model_comparison.png') -> None:
        """
        Plot comparison of multiple models across different metrics.
        
        Args:
            comparison_df: DataFrame with model comparison results
            metrics: List of metrics to plot
            save_filename: Filename to save plot
        """
        logger.info("Creating model comparison plot...")
        
        try:
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
            
            if n_metrics == 1:
                axes = [axes]
            
            for idx, metric in enumerate(metrics):
                if metric not in comparison_df.columns:
                    logger.warning(f"Metric '{metric}' not found in comparison data")
                    continue
                
                ax = axes[idx]
                models = comparison_df['Model']
                values = comparison_df[metric]
                
                # Create bar plot
                bars = ax.bar(range(len(models)), values, 
                             color='steelblue', alpha=0.7, edgecolor='black')
                
                # Highlight best model (lowest value)
                best_idx = values.idxmin()
                bars[best_idx].set_color('lightgreen')
                
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylabel(metric, fontsize=12, fontweight='bold')
                ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    if metric == 'MAPE':
                        label = f'{height:.1f}%'
                    else:
                        label = f'{height:.2f}'
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            
            save_path = self.output_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {str(e)}")
            raise
    
    def plot_feature_importance(self,
                               feature_importance: pd.DataFrame,
                               top_n: int = 15,
                               model_name: str = "Model",
                               save_filename: Optional[str] = None) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            model_name: Name of the model
            save_filename: Filename to save plot (optional)
        """
        logger.info(f"Creating feature importance plot for {model_name}...")
        
        try:
            # Get top N features
            top_features = feature_importance.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'], 
                   color='steelblue', alpha=0.7, edgecolor='black')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'])
            ax.invert_yaxis()  # Highest importance at top
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name}: Top {top_n} Feature Importance', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            # Save plot
            if save_filename:
                save_path = self.output_dir / save_filename
            else:
                save_path = self.output_dir / f'{model_name.lower().replace(" ", "_")}_feature_importance.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise


def visualize_all_results(model_results: Dict[str, Dict],
                         comparison_df: pd.DataFrame,
                         output_dir: str = 'outputs/figures') -> None:
    """
    Create all visualizations for model results.
    
    Args:
        model_results: Dictionary with model results
        comparison_df: DataFrame with model comparison
        output_dir: Directory to save outputs
    """
    logger.info("="*70)
    logger.info("CREATING ALL VISUALIZATIONS")
    logger.info("="*70)
    
    visualizer = ForecastVisualizer(output_dir)
    
    try:
        # 1. Model comparison
        visualizer.plot_model_comparison(comparison_df)
        logger.info("✓ Model comparison plot created")
        
        # 2. Individual model plots
        for model_name, results in model_results.items():
            logger.info(f"\nCreating plots for {model_name}...")
            
            # Actual vs Predicted
            visualizer.plot_actual_vs_predicted(
                results['actuals'],
                results['predictions'],
                results.get('dates'),
                model_name
            )
            
            # Residuals
            visualizer.plot_residuals(
                results['actuals'],
                results['predictions'],
                results.get('dates'),
                model_name
            )
            
            # Feature importance (if available)
            if 'feature_importance' in results:
                visualizer.plot_feature_importance(
                    results['feature_importance'],
                    top_n=15,
                    model_name=model_name
                )
            
            logger.info(f"✓ All plots created for {model_name}")
        
        logger.info("\n" + "="*70)
        logger.info("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
        logger.info(f"Saved to: {output_dir}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    y_true = np.random.randn(100) * 10 + 100
    y_pred = y_true + np.random.randn(100) * 5
    
    # Create visualizer
    visualizer = ForecastVisualizer(output_dir='outputs/figures/test')
    
    # Create plots
    visualizer.plot_actual_vs_predicted(y_true, y_pred, dates, "Test Model")
    visualizer.plot_residuals(y_true, y_pred, dates, "Test Model")
    
    print("Test visualizations created successfully!")