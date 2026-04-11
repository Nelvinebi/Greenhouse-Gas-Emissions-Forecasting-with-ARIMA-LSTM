"""General helper utilities."""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import joblib
from pathlib import Path


def save_object(obj, filepath: str):
    """Save Python object using joblib."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)


def load_object(filepath: str):
    """Load Python object using joblib."""
    return joblib.load(filepath)


def create_sequences(
    data: np.ndarray,
    seq_length: int,
    target_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series forecasting."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler,
    target_idx: int = 0,
    n_features: int = 1
) -> np.ndarray:
    """Inverse transform scaled predictions."""
    if n_features == 1:
        return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, target_idx] = predictions.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]


def add_time_features(df: pd.DataFrame, datetime_col: str = None) -> pd.DataFrame:
    """Add time-based features to dataframe."""
    if datetime_col:
        df = df.copy()
        df["year"] = df[datetime_col].dt.year
        df["month"] = df[datetime_col].dt.month
        df["quarter"] = df[datetime_col].dt.quarter
        df["day_of_year"] = df[datetime_col].dt.dayofyear
    else:
        df = df.copy()
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["day_of_year"] = df.index.dayofyear
    
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    
    return df


def detect_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """Detect stationarity using ADF and KPSS tests."""
    from statsmodels.tsa.stattools import adfuller, kpss
    
    adf_result = adfuller(series.dropna())
    adf_stationary = adf_result[1] < significance
    
    kpss_result = kpss(series.dropna(), regression="c")
    kpss_stationary = kpss_result[1] > significance
    
    return {
        "adf_statistic": adf_result[0],
        "adf_pvalue": adf_result[1],
        "adf_stationary": adf_stationary,
        "kpss_statistic": kpss_result[0],
        "kpss_pvalue": kpss_result[1],
        "kpss_stationary": kpss_stationary,
        "is_stationary": adf_stationary and kpss_stationary
    }


def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series data maintaining temporal order."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test