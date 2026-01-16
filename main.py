"""
Main Pipeline for Retail Sales Analysis & Forecasting

This script orchestrates the complete workflow:
1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (Moving Average, SARIMA, Random Forest)
5. Model Evaluation
6. Visualization

Usage:
    python main.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
BASE_DIR = Path(__file__).resolve().parent
outputs_dir = BASE_DIR / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import DataLoader, create_sample_data
from src.eda import perform_eda
from src.feature_engineering import engineer_features
from src.forecasting import train_all_models, forecast_next_n_days
from src.evaluation import evaluate_models
from src.visualization import visualize_all_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(outputs_dir / "pipeline.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main execution function for the retail sales forecasting pipeline.
    
    Pipeline Steps:
    1. Load data from data/raw/
    2. Perform EDA and save visualizations
    3. Engineer features and save processed data
    4. Train multiple forecasting models
    5. Evaluate and compare models
    6. Create comprehensive visualizations
    7. Generate future forecasts
    """
    
    print("\n" + "="*80)
    print(" "*20 + "RETAIL SALES ANALYSIS & FORECASTING")
    print(" "*30 + "Complete Pipeline")
    print("="*80 + "\n")
    
    try:
        # ====================================================================
        # SETUP: Define paths and create directories
        # ====================================================================
        logger.info("Setting up directories...")
        
        data_dir = BASE_DIR / "data"
        raw_data_dir = data_dir / "raw"
        processed_data_dir = data_dir / "processed"

        outputs_dir = BASE_DIR / "outputs"
        figures_dir = outputs_dir / "figures"
        reports_dir = outputs_dir / "reports"   
        
        # Create directories
        for directory in [raw_data_dir, processed_data_dir, figures_dir, reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ Directories created/verified")
        
        # ====================================================================
        # STEP 1: DATA LOADING
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING")
        print("="*80)
        
        sample_data_path = raw_data_dir / "sales_data.csv"
        
        # Create sample data if it doesn't exist
        if not sample_data_path.exists():
            logger.info("Sample data not found. Creating sample dataset...")
            create_sample_data(sample_data_path, n_records=730)  # 2 years of data
            logger.info("✓ Sample data created")
        
        # Load data
        logger.info(f"Loading data from: {sample_data_path}")
        loader = DataLoader(sample_data_path)
        df = loader.load_data(date_column='date', sales_column='sales')
        
        print(f"\n✓ Data loaded successfully")
        print(f"  - Shape: {df.shape}")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Missing values: {df.isnull().sum().sum()}")
        
        # ====================================================================
        # STEP 2: EXPLORATORY DATA ANALYSIS
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        eda_output_dir = figures_dir / "eda"
        logger.info("Performing comprehensive EDA...")
        
        eda_results = perform_eda(df, output_dir=str(eda_output_dir))
        
        print(f"\n✓ EDA completed")
        print(f"  - Visualizations saved to: {eda_output_dir}")
        print(f"  - Summary statistics calculated")
        print(f"  - Seasonality patterns analyzed")
        
        # Save summary statistics
        if 'summary_stats' in eda_results:
            stats_path = reports_dir / "summary_statistics.csv"
            eda_results['summary_stats'].to_csv(stats_path)
            logger.info(f"Summary statistics saved to: {stats_path}")
        
        # ====================================================================
        # STEP 3: FEATURE ENGINEERING
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*80)
        
        logger.info("Engineering features...")
        
        processed_data_path = processed_data_dir / "feature_data.csv"
        df_features = engineer_features(
            df,
            date_column='date',
            sales_column='sales',
            save_output=True,
            output_path=str(processed_data_path)
        )
        
        print(f"\n✓ Feature engineering completed")
        print(f"  - Original features: {df.shape[1]}")
        print(f"  - Engineered features: {df_features.shape[1]}")
        print(f"  - Total features created: {df_features.shape[1] - df.shape[1]}")
        print(f"  - Processed data saved to: {processed_data_path}")
        
        # ====================================================================
        # STEP 4: MODEL TRAINING & FORECASTING
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 4: MODEL TRAINING & FORECASTING")
        print("="*80)
        
        logger.info("Training all forecasting models...")
        
        model_results = train_all_models(
            df_features,
            target_column='sales',
            test_size=0.2
        )
        
        print(f"\n✓ Model training completed")
        print(f"  - Models trained: {len(model_results)}")
        print(f"  - Models: {', '.join(model_results.keys())}")
        
        # ====================================================================
        # STEP 5: MODEL EVALUATION
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 5: MODEL EVALUATION")
        print("="*80)
        
        logger.info("Evaluating all models...")
        
        comparison_df = evaluate_models(
            model_results,
            save_results=True,
            output_dir=str(reports_dir)
        )
        
        print(f"\n✓ Model evaluation completed")
        print(f"  - Evaluation metrics saved to: {reports_dir}")
        
        # Display comparison table
        print("\n" + "-"*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("-"*80)
        print(comparison_df.to_string(index=False))
        print("-"*80)
        
        # Identify best model
        best_model_name = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
        best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'RMSE']
        
        print(f"\n🏆 Best Model: {best_model_name} (RMSE: {best_rmse:.4f})")
        
        # ====================================================================
        # STEP 6: VISUALIZATION
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 6: VISUALIZATION")
        print("="*80)
        
        logger.info("Creating comprehensive visualizations...")
        
        visualize_all_results(
            model_results,
            comparison_df,
            output_dir=str(figures_dir)
        )
        
        print(f"\n✓ Visualizations created")
        print(f"  - All plots saved to: {figures_dir}")
        print(f"  - Actual vs Predicted plots")
        print(f"  - Residual analysis plots")
        print(f"  - Model comparison charts")
        print(f"  - Feature importance plots")
        
        # ====================================================================
        # STEP 7: FUTURE FORECASTING (OPTIONAL)
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 7: FUTURE FORECASTING (Next 30 Days)")
        print("="*80)
        
        logger.info("Generating future forecasts...")
        
        # Use best model for future forecasting
        best_model = model_results[best_model_name]['model']
        
        # Get last known data for forecasting
        df_clean = df_features.dropna(subset=['sales'])
        last_data = df_clean.tail(100)  # Use last 100 days for context
        
        # Determine model type
        if 'moving_average' in best_model_name.lower():
            model_type = 'ma'
        elif 'sarima' in best_model_name.lower():
            model_type = 'sarima'
        else:
            model_type = 'rf'
        
        try:
            future_forecast = forecast_next_n_days(
                best_model,
                last_data,
                n_days=30,
                model_type=model_type
            )
            
            # Save forecast
            forecast_dates = pd.date_range(
                start=df['date'].max() + pd.Timedelta(days=1),
                periods=30,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecasted_sales': future_forecast
            })
            
            forecast_path = reports_dir / 'future_forecast_30_days.csv'
            forecast_df.to_csv(forecast_path, index=False)
            
            print(f"\n✓ Future forecast generated")
            print(f"  - Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
            print(f"  - Average forecasted sales: {future_forecast.mean():.2f}")
            print(f"  - Forecast saved to: {forecast_path}")
            
            # Display first few forecasts
            print(f"\n  First 5 days forecast:")
            for i in range(min(5, len(future_forecast))):
                print(f"    {forecast_dates[i].strftime('%Y-%m-%d')}: {future_forecast[i]:.2f}")
        
        except Exception as e:
            logger.warning(f"Future forecasting failed: {str(e)}")
            print(f"\n⚠ Future forecasting skipped due to error")
        
        # ====================================================================
        # PIPELINE COMPLETION
        # ====================================================================
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉")
        print("="*80)
        
        print("\n📊 SUMMARY OF OUTPUTS:")
        print("-"*80)
        print(f"1. EDA Visualizations    : {eda_output_dir}")
        print(f"2. Processed Data        : {processed_data_path}")
        print(f"3. Model Comparison      : {reports_dir / 'model_comparison.csv'}")
        print(f"4. Forecast Plots        : {figures_dir}")
        print(f"5. Future Forecast       : {reports_dir / 'future_forecast_30_days.csv'}")
        print("-"*80)
        
        print(f"\n🏆 Best Performing Model : {best_model_name}")
        print(f"📈 RMSE                  : {best_rmse:.4f}")
        
        print("\n" + "="*80)
        print("Thank you for using Retail Sales Analysis & Forecasting!")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        print(f"\n❌ ERROR: Pipeline failed")
        print(f"Error details: {str(e)}")
        print(f"Check logs in outputs/pipeline.log for more information")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)