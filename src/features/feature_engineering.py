"""Feature engineering for time series."""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.feature_selection import mutual_info_regression
import logging

try:
    from src.utils.config import Config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import Config

class FeatureEngineer:
    """Engineer features for GHG emissions forecasting."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_importance = {}
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create complete feature set."""
        self.logger.info("Starting feature engineering...")
        df = df.copy()
        
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_cyclical_features(df)
        df = self.create_difference_features(df)
        df = self.create_interaction_features(df)
        
        df = df.dropna()
        
        self.logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features."""
        target = self.config.get("data.target_column")
        lags = self.config.get("features.lag_features", [1, 2, 3, 6, 12])
        
        for lag in lags:
            df[f"{target}_lag_{lag}"] = df[target].shift(lag)
        
        self.logger.info(f"Created {len(lags)} lag features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics."""
        target = self.config.get("data.target_column")
        windows = self.config.get("features.rolling_windows", [3, 6, 12])
        
        for window in windows:
            df[f"{target}_rolling_mean_{window}"] = df[target].rolling(window).mean()
            df[f"{target}_rolling_std_{window}"] = df[target].rolling(window).std()
            df[f"{target}_rolling_min_{window}"] = df[target].rolling(window).min()
            df[f"{target}_rolling_max_{window}"] = df[target].rolling(window).max()
        
        df[f"{target}_ema_12"] = df[target].ewm(span=12).mean()
        
        self.logger.info(f"Created rolling features for windows: {windows}")
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical time features."""
        if not self.config.get("features.cyclical_encoding", True):
            return df
        
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        
        quarter = df.index.quarter
        df["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)
        
        self.logger.info("Created cyclical time features")
        return df
    
    def create_difference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create differencing features for stationarity."""
        target = self.config.get("data.target_column")
        orders = self.config.get("features.difference_order", [1, 12])
        
        for order in orders:
            df[f"{target}_diff_{order}"] = df[target].diff(order)
        
        df["yoy_change"] = df[target].pct_change(12) * 100
        
        self.logger.info(f"Created difference features for orders: {orders}")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between exogenous variables."""
        exog = self.config.get("features.exogenous", [])
        
        if len(exog) >= 2:
            df[f"{exog[0]}_x_{exog[1]}"] = df[exog[0]] * df[exog[1]]
            df[f"{exog[0]}_div_{exog[1]}"] = df[exog[0]] / (df[exog[1]] + 1e-8)
        
        if len(exog) >= 3:
            for var in exog:
                df[f"{var}_ma_3"] = df[var].rolling(3).mean()
        
        self.logger.info("Created interaction features")
        return df
    
    def select_features(
        self,
        df: pd.DataFrame,
        method: str = "correlation",
        threshold: float = 0.1
    ) -> List[str]:
        """Select most important features."""
        target = self.config.get("data.target_column")
        feature_cols = [c for c in df.columns if c != target]
        
        if method == "correlation":
            correlations = df[feature_cols].corrwith(df[target]).abs()
            selected = correlations[correlations > threshold].index.tolist()
        
        elif method == "mutual_info":
            X = df[feature_cols].fillna(0)
            y = df[target]
            
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_series = pd.Series(mi_scores, index=feature_cols)
            selected = mi_series[mi_series > threshold].index.tolist()
        
        else:
            selected = feature_cols
        
        self.logger.info(f"Selected {len(selected)} features using {method}")
        return selected + [target]
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importance using correlation."""
        target = self.config.get("data.target_column")
        feature_cols = [c for c in df.columns if c != target]
        
        importance = df[feature_cols].corrwith(df[target]).abs().sort_values(ascending=False)
        
        return pd.DataFrame({
            "feature": importance.index,
            "correlation": importance.values
        })