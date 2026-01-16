"""
Exploratory Data Analysis Module

This module performs comprehensive EDA on retail sales data including:
- Advanced statistical tests (normality, correlation)
- Outlier detection and analysis
- Distribution analysis with KDE plots
- Time series decomposition (trend, seasonality, residuals)
- Professional visualizations with custom styling
- Executive summary generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set professional visualization style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10


class RetailEDA:
    """
    Comprehensive Exploratory Data Analysis for Retail Sales Data.
    
    This class provides advanced methods to analyze sales trends, seasonality patterns,
    promotional effects, and statistical properties of retail sales data.
    
    Attributes:
        df (pd.DataFrame): The retail sales dataframe
        date_column (str): Name of the date column
        sales_column (str): Name of the sales column
        output_dir (Path): Directory to save visualization outputs
        summary_stats (Dict): Comprehensive summary statistics
    """
    
    def __init__(self, df: pd.DataFrame, date_column: str = 'date', 
                 sales_column: str = 'sales', output_dir: str = 'outputs/figures'):
        """
        Initialize RetailEDA with sales data.
        
        Args:
            df: DataFrame containing retail sales data
            date_column: Name of the date column
            sales_column: Name of the sales column
            output_dir: Directory path to save output figures
        """
        self.df = df.copy()
        self.date_column = date_column
        self.sales_column = sales_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_stats = {}
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            self.df[date_column] = pd.to_datetime(self.df[date_column])
        
        # Sort by date
        self.df = self.df.sort_values(date_column).reset_index(drop=True)
        
        logger.info(f"RetailEDA initialized with {len(self.df)} records")
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics for sales data.
        
        Returns:
            pd.DataFrame: Summary statistics including mean, median, std, min, max, etc.
        """
        logger.info("Generating comprehensive summary statistics...")
        
        try:
            sales_data = self.df[self.sales_column]
            
            # Basic statistics
            stats_dict = {
                'count': len(sales_data),
                'mean': sales_data.mean(),
                'median': sales_data.median(),
                'std': sales_data.std(),
                'min': sales_data.min(),
                'max': sales_data.max(),
                'range': sales_data.max() - sales_data.min(),
                'q1': sales_data.quantile(0.25),
                'q3': sales_data.quantile(0.75),
                'iqr': sales_data.quantile(0.75) - sales_data.quantile(0.25),
                'variance': sales_data.var(),
                'skewness': sales_data.skew(),
                'kurtosis': sales_data.kurtosis(),
                'cv': (sales_data.std() / sales_data.mean()) * 100,  # Coefficient of variation
                'missing_count': sales_data.isnull().sum(),
                'missing_percent': (sales_data.isnull().sum() / len(sales_data)) * 100
            }
            
            self.summary_stats = stats_dict
            
            summary_df = pd.DataFrame([stats_dict]).T
            summary_df.columns = ['Value']
            
            logger.info("Summary statistics generated successfully")
            return summary_df
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {str(e)}")
            raise
    
    def perform_normality_tests(self) -> Dict[str, Tuple[float, float]]:
        """
        Perform statistical tests for normality of sales distribution.
        
        Tests performed:
        - Shapiro-Wilk test
        - Kolmogorov-Smirnov test
        - Anderson-Darling test
        
        Returns:
            Dict: Test results with statistics and p-values
        """
        logger.info("Performing normality tests...")
        
        try:
            sales_data = self.df[self.sales_column].dropna()
            results = {}
            
            # Shapiro-Wilk test
            if len(sales_data) <= 5000:  # Shapiro-Wilk works best for smaller samples
                shapiro_stat, shapiro_p = stats.shapiro(sales_data)
                results['Shapiro-Wilk'] = (shapiro_stat, shapiro_p)
                logger.info(f"Shapiro-Wilk: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(sales_data, 'norm', 
                                         args=(sales_data.mean(), sales_data.std()))
            results['Kolmogorov-Smirnov'] = (ks_stat, ks_p)
            logger.info(f"K-S test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
            
            # Anderson-Darling test
            anderson_result = stats.anderson(sales_data, dist='norm')
            results['Anderson-Darling'] = (anderson_result.statistic, anderson_result.critical_values[2])
            logger.info(f"Anderson-Darling: statistic={anderson_result.statistic:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing normality tests: {str(e)}")
            raise
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in sales data using multiple methods.
        
        Args:
            method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe containing outliers
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        try:
            sales_data = self.df[self.sales_column]
            
            if method == 'iqr':
                Q1 = sales_data.quantile(0.25)
                Q3 = sales_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = self.df[
                    (sales_data < lower_bound) | (sales_data > upper_bound)
                ]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(sales_data))
                outliers = self.df[z_scores > threshold]
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outlier_pct = (len(outliers) / len(self.df)) * 100
            logger.info(f"Found {len(outliers)} outliers ({outlier_pct:.2f}% of data)")
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            raise
    
    def plot_distribution_analysis(self, save_plot: bool = True) -> None:
        """
        Create comprehensive distribution analysis with histogram and KDE.
        
        Args:
            save_plot: Whether to save the plot to disk
        """
        logger.info("Creating distribution analysis plots...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            sales_data = self.df[self.sales_column].dropna()
            
            # 1. Histogram with KDE
            axes[0, 0].hist(sales_data, bins=50, density=True, alpha=0.7, 
                           color='steelblue', edgecolor='black', label='Histogram')
            
            # Add KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(sales_data)
            x_range = np.linspace(sales_data.min(), sales_data.max(), 100)
            axes[0, 0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            # Add normal distribution overlay
            mu, sigma = sales_data.mean(), sales_data.std()
            normal_dist = stats.norm.pdf(x_range, mu, sigma)
            axes[0, 0].plot(x_range, normal_dist, 'g--', linewidth=2, label='Normal Distribution')
            
            axes[0, 0].set_xlabel('Sales', fontweight='bold')
            axes[0, 0].set_ylabel('Density', fontweight='bold')
            axes[0, 0].set_title('Sales Distribution with KDE', fontweight='bold', fontsize=13)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Box Plot
            box_plot = axes[0, 1].boxplot(sales_data, vert=True, patch_artist=True,
                                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                                          medianprops=dict(color='red', linewidth=2),
                                          whiskerprops=dict(linewidth=1.5),
                                          capprops=dict(linewidth=1.5))
            axes[0, 1].set_ylabel('Sales', fontweight='bold')
            axes[0, 1].set_title('Box Plot (Outlier Detection)', fontweight='bold', fontsize=13)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Add statistics text
            stats_text = f'Mean: {sales_data.mean():.2f}\nMedian: {sales_data.median():.2f}\nStd: {sales_data.std():.2f}'
            axes[0, 1].text(1.15, sales_data.mean(), stats_text, fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 3. Q-Q Plot
            stats.probplot(sales_data, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontweight='bold', fontsize=13)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Violin Plot
            parts = axes[1, 1].violinplot([sales_data], positions=[1], showmeans=True, 
                                         showmedians=True, widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.7)
            
            axes[1, 1].set_ylabel('Sales', fontweight='bold')
            axes[1, 1].set_title('Violin Plot (Distribution Shape)', fontweight='bold', fontsize=13)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].set_xticks([1])
            axes[1, 1].set_xticklabels(['Sales'])
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / 'distribution_analysis.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Distribution analysis plot saved to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating distribution analysis: {str(e)}")
            raise
    
    def perform_time_series_decomposition(self, period: int = 7, save_plot: bool = True) -> Dict:
        """
        Perform time series decomposition to separate trend, seasonality, and residuals.
        
        Args:
            period: Seasonal period (7 for weekly, 30 for monthly)
            save_plot: Whether to save the plot
            
        Returns:
            Dict: Decomposition components
        """
        logger.info(f"Performing time series decomposition (period={period})...")
        
        try:
            # Set date as index
            df_temp = self.df.set_index(self.date_column)
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                df_temp[self.sales_column],
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            
            # Create visualization
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            
            # Original
            axes[0].plot(df_temp.index, df_temp[self.sales_column], color='steelblue', linewidth=1.5)
            axes[0].set_ylabel('Original', fontweight='bold')
            axes[0].set_title('Time Series Decomposition', fontweight='bold', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(df_temp.index, decomposition.trend, color='red', linewidth=2)
            axes[1].set_ylabel('Trend', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonality
            axes[2].plot(df_temp.index, decomposition.seasonal, color='green', linewidth=1.5)
            axes[2].set_ylabel('Seasonality', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            # Residuals
            axes[3].plot(df_temp.index, decomposition.resid, color='purple', linewidth=1, alpha=0.7)
            axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
            axes[3].set_ylabel('Residuals', fontweight='bold')
            axes[3].set_xlabel('Date', fontweight='bold')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / 'time_series_decomposition.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Decomposition plot saved to {plot_path}")
            
            plt.close()
            
            # Return components
            results = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'resid': decomposition.resid,
                'observed': decomposition.observed
            }
            
            logger.info("Time series decomposition completed")
            return results
            
        except Exception as e:
            logger.error(f"Error performing decomposition: {str(e)}")
            raise
    
    def analyze_correlation(self, save_plot: bool = True) -> pd.DataFrame:
        """
        Perform comprehensive correlation analysis on numeric features.
        
        Args:
            save_plot: Whether to save the correlation heatmap
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        logger.info("Performing correlation analysis...")
        
        try:
            # Select numeric columns
            numeric_df = self.df.select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create professional heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, ax=ax,
                       linewidths=0.5, cbar_kws={"shrink": 0.8},
                       vmin=-1, vmax=1)
            
            ax.set_title('Correlation Matrix Heatmap', fontweight='bold', fontsize=14, pad=20)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / 'correlation_heatmap.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation heatmap saved to {plot_path}")
            
            plt.close()
            
            logger.info("Correlation analysis completed")
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Error performing correlation analysis: {str(e)}")
            raise
    
    def analyze_sales_trends(self, save_plot: bool = True) -> None:
        """
        Analyze and visualize overall sales trends over time with professional styling.
        
        Args:
            save_plot: Whether to save the plot to disk
        """
        logger.info("Analyzing sales trends over time...")
        
        try:
            fig, ax = plt.subplots(figsize=(16, 7))
            
            # Plot daily sales with transparency
            ax.plot(self.df[self.date_column], self.df[self.sales_column], 
                   label='Daily Sales', linewidth=1, alpha=0.5, color='steelblue')
            
            # Add 7-day moving average
            ma_7 = self.df[self.sales_column].rolling(window=7, center=True).mean()
            ax.plot(self.df[self.date_column], ma_7, 
                   label='7-Day MA', linewidth=2.5, color='orange')
            
            # Add 30-day moving average
            ma_30 = self.df[self.sales_column].rolling(window=30, center=True).mean()
            ax.plot(self.df[self.date_column], ma_30, 
                   label='30-Day MA', linewidth=2.5, color='red')
            
            # Add trend line
            from scipy.stats import linregress
            x_numeric = np.arange(len(self.df))
            slope, intercept, r_value, p_value, std_err = linregress(x_numeric, self.df[self.sales_column])
            trend_line = slope * x_numeric + intercept
            ax.plot(self.df[self.date_column], trend_line, 
                   label=f'Trend Line (R²={r_value**2:.3f})', 
                   linewidth=2, color='green', linestyle='--', alpha=0.8)
            
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Sales', fontsize=12, fontweight='bold')
            ax.set_title('Sales Trends Over Time with Moving Averages', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # Add annotations for min and max
            max_idx = self.df[self.sales_column].idxmax()
            min_idx = self.df[self.sales_column].idxmin()
            
            ax.annotate(f'Max: {self.df.loc[max_idx, self.sales_column]:.1f}',
                       xy=(self.df.loc[max_idx, self.date_column], 
                           self.df.loc[max_idx, self.sales_column]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / 'sales_trends_advanced.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Sales trends plot saved to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error analyzing sales trends: {str(e)}")
            raise
    
    def generate_executive_summary(self) -> str:
        """
        Generate an executive summary of the EDA findings.
        
        Returns:
            str: Executive summary text
        """
        logger.info("Generating executive summary...")
        
        try:
            if not self.summary_stats:
                self.generate_summary_statistics()
            
            sales_data = self.df[self.sales_column]
            
            summary = f"""
EXECUTIVE SUMMARY - RETAIL SALES ANALYSIS
{'='*70}

DATA OVERVIEW:
- Total Records: {len(self.df):,}
- Date Range: {self.df[self.date_column].min().strftime('%Y-%m-%d')} to {self.df[self.date_column].max().strftime('%Y-%m-%d')}
- Duration: {(self.df[self.date_column].max() - self.df[self.date_column].min()).days} days

SALES STATISTICS:
- Average Daily Sales: {sales_data.mean():.2f} units
- Median Daily Sales: {sales_data.median():.2f} units
- Standard Deviation: {sales_data.std():.2f} units
- Coefficient of Variation: {self.summary_stats.get('cv', 0):.2f}%
- Sales Range: {sales_data.min():.2f} to {sales_data.max():.2f} units

DISTRIBUTION CHARACTERISTICS:
- Skewness: {sales_data.skew():.3f} ({'Right-skewed' if sales_data.skew() > 0 else 'Left-skewed' if sales_data.skew() < 0 else 'Symmetric'})
- Kurtosis: {sales_data.kurtosis():.3f} ({'Heavy-tailed' if sales_data.kurtosis() > 0 else 'Light-tailed'})

DATA QUALITY:
- Missing Values: {sales_data.isnull().sum()} ({(sales_data.isnull().sum()/len(sales_data)*100):.2f}%)
- Outliers (IQR method): {len(self.detect_outliers())} records ({(len(self.detect_outliers())/len(self.df)*100):.2f}%)

KEY INSIGHTS:
1. Sales show {'increasing' if sales_data.iloc[-30:].mean() > sales_data.iloc[:30].mean() else 'decreasing'} trend
2. Volatility: {'High' if self.summary_stats.get('cv', 0) > 20 else 'Moderate' if self.summary_stats.get('cv', 0) > 10 else 'Low'} (CV: {self.summary_stats.get('cv', 0):.1f}%)
3. Distribution: {'Approximately normal' if abs(sales_data.skew()) < 0.5 else 'Skewed'}

{'='*70}
            """
            
            logger.info("Executive summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            raise
    
    def analyze_monthly_seasonality(self, save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze monthly seasonality patterns with enhanced visualization.
        
        Returns:
            pd.DataFrame: Monthly aggregated sales statistics
        """
        logger.info("Analyzing monthly seasonality...")
        
        try:
            # Extract month and year
            self.df['month'] = self.df[self.date_column].dt.month
            self.df['year'] = self.df[self.date_column].dt.year
            self.df['month_name'] = self.df[self.date_column].dt.strftime('%b')
            
            # Aggregate by month
            monthly_sales = self.df.groupby('month').agg({
                self.sales_column: ['mean', 'sum', 'std', 'count']
            }).round(2)
            monthly_sales.columns = ['Mean Sales', 'Total Sales', 'Std Dev', 'Count']
            
            # Create enhanced visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_avg = self.df.groupby('month_name')[self.sales_column].mean().reindex(month_order)
            
            # Average sales by month with gradient colors
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, 12))
            bars = axes[0].bar(range(12), monthly_avg.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Highlight max and min
            max_idx = monthly_avg.values.argmax()
            min_idx = monthly_avg.values.argmin()
            bars[max_idx].set_color('gold')
            bars[min_idx].set_color('lightcoral')
            
            axes[0].set_xticks(range(12))
            axes[0].set_xticklabels(month_order, rotation=45, ha='right')
            axes[0].set_xlabel('Month', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Average Sales', fontsize=12, fontweight='bold')
            axes[0].set_title('Average Sales by Month', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, monthly_avg.values)):
                axes[0].text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Total sales by month
            monthly_total = self.df.groupby('month_name')[self.sales_column].sum().reindex(month_order)
            
            axes[1].plot(range(12), monthly_total.values, marker='o', linewidth=2.5, 
                        markersize=8, color='steelblue', markerfacecolor='orange')
            axes[1].fill_between(range(12), monthly_total.values, alpha=0.3, color='steelblue')
            axes[1].set_xticks(range(12))
            axes[1].set_xticklabels(month_order, rotation=45, ha='right')
            axes[1].set_xlabel('Month', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Total Sales', fontsize=12, fontweight='bold')
            axes[1].set_title('Total Sales by Month', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / 'monthly_seasonality_advanced.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Monthly seasonality plot saved to {plot_path}")
            
            plt.close()
            
            logger.info("Monthly seasonality analysis completed")
            return monthly_sales
            
        except Exception as e:
            logger.error(f"Error analyzing monthly seasonality: {str(e)}")
            raise
    
    def analyze_yearly_seasonality(self, save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze yearly patterns in sales data.
        
        Returns:
            pd.DataFrame: Yearly aggregated sales statistics
        """
        logger.info("Analyzing yearly seasonality...")
        
        try:
            if 'year' not in self.df.columns:
                self.df['year'] = self.df[self.date_column].dt.year
            
            yearly_sales = self.df.groupby('year').agg({
                self.sales_column: ['mean', 'sum', 'std', 'count']
            }).round(2)
            yearly_sales.columns = ['Mean Sales', 'Total Sales', 'Std Dev', 'Count']
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            years = sorted(self.df['year'].unique())
            yearly_avg = self.df.groupby('year')[self.sales_column].mean()
            yearly_total = self.df.groupby('year')[self.sales_column].sum()
            
            # Average sales by year
            axes[0].plot(years, yearly_avg.values, marker='o', linewidth=3, 
                        markersize=10, color='steelblue', markerfacecolor='red')
            axes[0].fill_between(years, yearly_avg.values, alpha=0.2, color='steelblue')
            axes[0].set_xlabel('Year', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Average Sales', fontsize=12, fontweight='bold')
            axes[0].set_title('Average Sales by Year', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Add growth rate annotations
            if len(years) > 1:
                growth_rate = ((yearly_avg.values[-1] - yearly_avg.values[0]) / yearly_avg.values[0]) * 100
                axes[0].text(0.5, 0.95, f'Growth Rate: {growth_rate:+.1f}%',
                           transform=axes[0].transAxes, fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           verticalalignment='top', horizontalalignment='center')
            
            # Total sales by year
            bars = axes[1].bar(years, yearly_total.values, color='coral', alpha=0.7, 
                              edgecolor='black', linewidth=1.5)
            axes[1].set_xlabel('Year', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Total Sales', fontsize=12, fontweight='bold')
            axes[1].set_title('Total Sales by Year', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / 'yearly_seasonality_advanced.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Yearly seasonality plot saved to {plot_path}")
            
            plt.close()
            
            logger.info("Yearly seasonality analysis completed")
            return yearly_sales
            
        except Exception as e:
            logger.error(f"Error analyzing yearly seasonality: {str(e)}")
            raise
    
    def analyze_promotion_impact(self, promotion_column: str = 'promotion', 
                                 save_plot: bool = True) -> Dict:
        """
        Analyze the impact of promotions on sales with statistical testing.
        
        Args:
            promotion_column: Name of the promotion indicator column
            save_plot: Whether to save the plot to disk
            
        Returns:
            Dict: Statistics comparing promotion vs non-promotion sales
        """
        logger.info("Analyzing promotion impact on sales...")
        
        try:
            if promotion_column not in self.df.columns:
                logger.warning(f"Column '{promotion_column}' not found. Skipping promotion analysis.")
                return {}
            
            promo_sales = self.df[self.df[promotion_column] == 1][self.sales_column]
            non_promo_sales = self.df[self.df[promotion_column] == 0][self.sales_column]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(promo_sales, non_promo_sales)
            
            # Calculate uplift
            uplift_pct = ((promo_sales.mean() - non_promo_sales.mean()) / non_promo_sales.mean()) * 100
            
            stats_dict = {
                'promotion_avg_sales': promo_sales.mean(),
                'non_promotion_avg_sales': non_promo_sales.mean(),
                'uplift_percentage': uplift_pct,
                'promotion_days': len(promo_sales),
                'non_promotion_days': len(non_promo_sales),
                't_statistic': t_stat,
                'p_value': p_value,
                'statistically_significant': p_value < 0.05
            }
            
            # Create enhanced visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Box plot comparison
            box_data = [non_promo_sales, promo_sales]
            bp = axes[0].boxplot(box_data, labels=['No Promotion', 'Promotion'],
                                patch_artist=True, widths=0.6)
            
            colors = ['lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[0].set_ylabel('Sales', fontsize=12, fontweight='bold')
            axes[0].set_title('Sales Distribution: Promotion vs Non-Promotion', 
                            fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # 2. Bar chart comparison
            categories = ['Non-Promotion', 'Promotion']
            means = [non_promo_sales.mean(), promo_sales.mean()]
            colors_bar = ['lightcoral', 'lightgreen']
            
            bars = axes[1].bar(categories, means, color=colors_bar, alpha=0.8, 
                              edgecolor='black', linewidth=2)
            axes[1].set_ylabel('Average Sales', fontsize=12, fontweight='bold')
            axes[1].set_title(f'Average Sales Comparison\nUplift: {uplift_pct:.1f}% (p={p_value:.4f})', 
                            fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, means):
                axes[1].text(bar.get_x() + bar.get_width()/2., val, f'{val:.2f}',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Add significance indicator
            if p_value < 0.05:
                axes[1].text(0.5, max(means) * 1.1, '***Statistically Significant***',
                           ha='center', fontsize=10, fontweight='bold', color='green',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # 3. Distribution comparison
            axes[2].hist(non_promo_sales, bins=30, alpha=0.6, label='No Promotion', 
                        color='lightcoral', edgecolor='black')
            axes[2].hist(promo_sales, bins=30, alpha=0.6, label='Promotion', 
                        color='lightgreen', edgecolor='black')
            axes[2].axvline(non_promo_sales.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean (No Promo): {non_promo_sales.mean():.1f}')
            axes[2].axvline(promo_sales.mean(), color='green', linestyle='--', 
                          linewidth=2, label=f'Mean (Promo): {promo_sales.mean():.1f}')
            axes[2].set_xlabel('Sales', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
            axes[2].set_title('Sales Distribution Overlay', fontsize=13, fontweight='bold')
            axes[2].legend(fontsize=9)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / 'promotion_impact_advanced.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Promotion impact plot saved to {plot_path}")
            
            plt.close()
            
            logger.info(f"Promotion analysis completed. Uplift: {uplift_pct:.2f}%, p-value: {p_value:.4f}")
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error analyzing promotion impact: {str(e)}")
            raise
    
    def analyze_day_of_week_patterns(self, save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze sales patterns by day of week.
        
        Returns:
            pd.DataFrame: Sales statistics by day of week
        """
        logger.info("Analyzing day of week patterns...")
        
        try:
            self.df['day_of_week'] = self.df[self.date_column].dt.dayofweek
            self.df['day_name'] = self.df[self.date_column].dt.strftime('%A')
            
            dow_sales = self.df.groupby(['day_of_week', 'day_name'])[self.sales_column].agg([
                'mean', 'sum', 'std', 'count'
            ]).round(2)
            dow_sales.columns = ['Mean Sales', 'Total Sales', 'Std Dev', 'Count']
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_avg = self.df.groupby('day_name')[self.sales_column].mean().reindex(day_order)
            
            colors = ['steelblue'] * 5 + ['coral', 'coral']  # Highlight weekends
            bars = ax.bar(range(7), dow_avg.values, color=colors, alpha=0.8, 
                         edgecolor='black', linewidth=1.5)
            
            ax.set_xticks(range(7))
            ax.set_xticklabels(day_order, rotation=45, ha='right')
            ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Sales', fontsize=12, fontweight='bold')
            ax.set_title('Average Sales by Day of Week (Weekends Highlighted)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, dow_avg.values):
                ax.text(bar.get_x() + bar.get_width()/2., val, f'{val:.1f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add weekend vs weekday comparison
            weekday_avg = dow_avg[:5].mean()
            weekend_avg = dow_avg[5:].mean()
            weekend_uplift = ((weekend_avg - weekday_avg) / weekday_avg) * 100
            
            ax.text(0.5, 0.95, f'Weekend Uplift: {weekend_uplift:+.1f}%',
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                   verticalalignment='top', horizontalalignment='center')
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.output_dir / 'day_of_week_patterns_advanced.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Day of week patterns plot saved to {plot_path}")
            
            plt.close()
            
            logger.info("Day of week analysis completed")
            return dow_sales
            
        except Exception as e:
            logger.error(f"Error analyzing day of week patterns: {str(e)}")
            raise
    
    def run_complete_eda(self) -> Dict:
        """
        Run complete EDA pipeline with all analyses and visualizations.
        
        Returns:
            Dict: Dictionary containing all analysis results
        """
        logger.info("="*70)
        logger.info("STARTING COMPREHENSIVE EDA ANALYSIS")
        logger.info("="*70)
        
        results = {}
        
        try:
            # 1. Summary statistics
            results['summary_stats'] = self.generate_summary_statistics()
            logger.info("✓ Summary statistics generated")
            
            # 2. Normality tests
            results['normality_tests'] = self.perform_normality_tests()
            logger.info("✓ Normality tests completed")
            
            # 3. Distribution analysis
            self.plot_distribution_analysis(save_plot=True)
            logger.info("✓ Distribution analysis completed")
            
            # 4. Sales trends
            self.analyze_sales_trends(save_plot=True)
            logger.info("✓ Sales trends analyzed")
            
            # 5. Time series decomposition
            results['decomposition'] = self.perform_time_series_decomposition(period=7, save_plot=True)
            logger.info("✓ Time series decomposition completed")
            
            # 6. Correlation analysis
            results['correlation_matrix'] = self.analyze_correlation(save_plot=True)
            logger.info("✓ Correlation analysis completed")
            
            # 7. Monthly seasonality
            results['monthly_stats'] = self.analyze_monthly_seasonality(save_plot=True)
            logger.info("✓ Monthly seasonality analyzed")
            
            # 8. Yearly seasonality
            results['yearly_stats'] = self.analyze_yearly_seasonality(save_plot=True)
            logger.info("✓ Yearly seasonality analyzed")
            
            # 9. Day of week patterns
            results['dow_stats'] = self.analyze_day_of_week_patterns(save_plot=True)
            logger.info("✓ Day of week patterns analyzed")
            
            # 10. Promotion impact (if column exists)
            if 'promotion' in self.df.columns:
                results['promotion_stats'] = self.analyze_promotion_impact(save_plot=True)
                logger.info("✓ Promotion impact analyzed")
            
            # 11. Outlier detection
            results['outliers'] = self.detect_outliers()
            logger.info("✓ Outliers detected")
            
            # 12. Executive summary
            results['executive_summary'] = self.generate_executive_summary()
            logger.info("✓ Executive summary generated")
            
            logger.info("="*70)
            logger.info("COMPREHENSIVE EDA ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info(f"Total visualizations created: 10+")
            logger.info(f"All outputs saved to: {self.output_dir}")
            logger.info("="*70)
            
            # Print executive summary
            print(results['executive_summary'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error during complete EDA: {str(e)}")
            raise


def perform_eda(df: pd.DataFrame, date_column: str = 'date', 
                sales_column: str = 'sales', 
                output_dir: str = 'outputs/figures') -> Dict:
    """
    Convenience function to perform complete EDA on retail sales data.
    
    Args:
        df: DataFrame containing retail sales data
        date_column: Name of the date column
        sales_column: Name of the sales column
        output_dir: Directory to save output figures
        
    Returns:
        Dict: Dictionary containing all analysis results
    """
    eda = RetailEDA(df, date_column, sales_column, output_dir)
    return eda.run_complete_eda()


if __name__ == "__main__":
    from data_loader import DataLoader, create_sample_data
    
    sample_path = Path("data/raw/sample_sales_data.csv")
    if not sample_path.exists():
        create_sample_data(sample_path, n_records=730)
    
    loader = DataLoader(sample_path)
    df = loader.load_data()
    
    results = perform_eda(df, output_dir='outputs/figures/eda')
    
    print("\n" + "="*70)
    print("EDA COMPLETED - Check outputs/figures/eda for visualizations")
    print("="*70)