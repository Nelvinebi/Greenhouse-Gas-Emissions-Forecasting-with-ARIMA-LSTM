"""Evaluation metrics for time series forecasting."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


class EvaluationMetrics:
    """Calculate and store forecasting metrics."""
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    @staticmethod
    def mpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Percentage Error."""
        return np.mean((y_true - y_pred) / y_true) * 100
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @classmethod
    def calculate_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all metrics."""
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": cls.rmse(y_true, y_pred),
            "mape": cls.mape(y_true, y_pred),
            "smape": cls.smape(y_true, y_pred),
            "mpe": cls.mpe(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred)
        }
        return metrics
    
    @classmethod
    def evaluate(
        cls,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray]
    ) -> "MetricsResult":
        """Evaluate predictions."""
        y_true = np.array(actual).flatten()
        y_pred = np.array(predicted).flatten()
        
        metrics = cls.calculate_all(y_true, y_pred)
        
        return MetricsResult(metrics, y_true, y_pred)


class MetricsResult:
    """Container for evaluation results."""
    
    def __init__(
        self,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        self.metrics = metrics
        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = y_true - y_pred
    
    def summary(self) -> str:
        """Return formatted summary."""
        lines = ["Evaluation Metrics:", "=" * 40]
        for metric, value in self.metrics.items():
            lines.append(f"{metric.upper():10s}: {value:10.4f}")
        return "\\n".join(lines)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return self.metrics.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame([self.metrics])
    
    def get_residuals(self) -> np.ndarray:
        """Get prediction residuals."""
        return self.residuals