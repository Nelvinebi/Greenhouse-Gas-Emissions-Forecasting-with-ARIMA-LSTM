"""Plotting functions for time series analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path


class TimeSeriesVisualizer:
    """Create publication-quality time series plots."""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        plt.style.use(style)
        self.colors = {
            'primary': '#2E7D32',
            'secondary': '#1565C0',
            'accent': '#C62828',
            'neutral': '#424242'
        }
    
    def plot_forecast(
        self,
        train_data: pd.Series,
        test_data: pd.Series,
        predictions: Dict[str, np.ndarray],
        title: str = "Forecast vs Actual",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot forecast with confidence intervals."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training data
        ax.plot(train_data.index, train_data.values, 
                color=self.colors['neutral'], label='Training', alpha=0.7)
        
        # Plot actual test data
        ax.plot(test_data.index, test_data.values,
                color=self.colors['primary'], label='Actual', linewidth=2)
        
        # Plot predictions
        pred_index = test_data.index[:len(predictions['mean'])]
        ax.plot(pred_index, predictions['mean'],
                color=self.colors['accent'], label='Forecast', 
                linestyle='--', linewidth=2)
        
        # Confidence intervals
        if 'lower' in predictions and 'upper' in predictions:
            ax.fill_between(
                pred_index,
                predictions['lower'],
                predictions['upper'],
                alpha=0.2, color=self.colors['accent'],
                label='95% Confidence Interval'
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Emissions (MTCO2e)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(
        self,
        residuals: np.ndarray,
        title: str = "Residual Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot residual diagnostics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residuals over time
        axes[0, 0].plot(residuals, color=self.colors['primary'])
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        
        # Histogram
        axes[0, 1].hist(residuals, bins=20, color=self.colors['secondary'], 
                       edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual Value')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # ACF of residuals
        from statsmodels.tsa.stattools import acf
        lags = min(40, len(residuals)//2)
        acf_values = acf(residuals, nlags=lags, fft=True)
        axes[1, 1].bar(range(lags+1), acf_values, color=self.colors['primary'])
        axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[1, 1].axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('ACF of Residuals')
        axes[1, 1].set_xlabel('Lag')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "rmse",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Compare multiple models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(results.keys())
        values = [results[m][metric] for m in models]
        
        colors = [self.colors['primary'] if v == min(values) 
                 else self.colors['neutral'] for v in values]
        
        bars = ax.bar(models, values, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(f'Model Comparison: {metric.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig