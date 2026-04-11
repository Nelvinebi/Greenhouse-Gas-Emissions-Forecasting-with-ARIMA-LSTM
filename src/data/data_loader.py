"""Data loading and preprocessing."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

try:
    from src.utils.config import Config
    from src.utils.helpers import add_time_features, detect_stationarity
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import Config
    from utils.helpers import add_time_features, detect_stationarity


class DataLoader:
    """Load and preprocess GHG emissions data."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scaler = None
        self.target_scaler = None
        
    def load_raw_data(self, filepath: str = None) -> pd.DataFrame:
        """Load raw data from CSV."""
        if filepath is None:
            filepath = self.config.get("data.raw_data_path")
        
        self.logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(
            filepath,
            index_col=self.config.get("data.date_column", "Date"),
            parse_dates=True
        )
        
        df = df.asfreq("M")
        
        self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality."""
        target_col = self.config.get("data.target_column")
        
        checks = {
            "missing_values": df[target_col].isnull().sum() == 0,
            "negative_values": (df[target_col] < 0).sum() == 0,
            "constant_values": df[target_col].nunique() > 1,
            "frequency": pd.infer_freq(df.index) is not None
        }
        
        for check, passed in checks.items():
            if not passed:
                self.logger.error(f"Data validation failed: {check}")
                return False
        
        self.logger.info("Data validation passed")
        return True
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        train_end: str = None,
        test_start: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        if train_end is None:
            train_end = self.config.get("data.train_end")
        if test_start is None:
            test_start = self.config.get("data.test_start")
        
        train = df[df.index <= train_end].copy()
        test = df[df.index >= test_start].copy()
        
        self.logger.info(f"Train set: {len(train)} rows ({train.index.min()} to {train.index.max()})")
        self.logger.info(f"Test set: {len(test)} rows ({test.index.min()} to {test.index.max()})")
        
        return train, test
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        scale: bool = True,
        scaler_type: str = "minmax"
    ) -> Tuple[pd.DataFrame, Optional[object]]:
        """Prepare data for modeling."""
        target_col = self.config.get("data.target_column")
        feature_cols = [c for c in df.columns if c != target_col]
        
        if scale:
            if scaler_type == "minmax":
                self.scaler = MinMaxScaler()
                self.target_scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
                self.target_scaler = StandardScaler()
            
            if feature_cols:
                df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            
            df[target_col] = self.target_scaler.fit_transform(df[[target_col]])
            
            self.logger.info(f"Data scaled using {scaler_type}")
        
        return df, self.scaler
    
    def inverse_scale_target(self, values: np.ndarray) -> np.ndarray:
        """Inverse transform scaled target values."""
        if self.target_scaler is None:
            return values
        return self.target_scaler.inverse_transform(values.reshape(-1, 1)).flatten()
    
    def get_stationarity_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate stationarity report for all numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        reports = []
        
        for col in numeric_cols:
            result = detect_stationarity(df[col])
            result["column"] = col
            reports.append(result)
        
        return pd.DataFrame(reports)


class TimeSeriesCV:
    """Time series cross-validation with expanding window."""
    
    def __init__(self, n_splits: int = 5, test_size: int = 12):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, df: pd.DataFrame):
        """Generate train/test indices for time series CV."""
        n_samples = len(df)
        indices = np.arange(n_samples)
        
        min_train_size = n_samples - (self.n_splits * self.test_size)
        
        if min_train_size < self.test_size:
            raise ValueError("Not enough data for specified splits")
        
        for i in range(self.n_splits):
            train_end = min_train_size + (i * self.test_size)
            test_end = train_end + self.test_size
            
            train_idx = indices[:train_end]
            test_idx = indices[train_end:test_end]
            
            yield train_idx, test_idx