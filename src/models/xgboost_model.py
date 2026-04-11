"""XGBoost model for time series forecasting."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import joblib
from pathlib import Path

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from src.models.base_model import BaseModel
    from src.utils.config import Config
except ImportError:
    from base_model import BaseModel
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import Config


class XGBoostModel(BaseModel):
    """XGBoost for time series with feature engineering."""
    
    def __init__(self, config: Config = None, name: str = "xgboost"):
        super().__init__(config, name)
        
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required")
        
        self.model = None
        self.feature_cols = None
        self.target_col = self.config.get("data.target_column")
        
    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        **kwargs
    ) -> "XGBoostModel":
        """Train XGBoost model."""
        self.logger.info("Training XGBoost model...")
        
        # Prepare features
        self.feature_cols = [c for c in train_data.columns if c != self.target_col]
        
        X_train = train_data[self.feature_cols]
        y_train = train_data[self.target_col]
        
        # Get parameters from config
        params = self.config.get("xgboost.params", {})
        
        # Prepare validation set
        eval_set = [(X_train, y_train)]
        if validation_data is not None:
            X_val = validation_data[0][self.feature_cols]
            y_val = validation_data[0][self.target_col]
            eval_set.append((X_val, y_val))
        
        self.model = xgb.XGBRegressor(**params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        self.is_fitted = True
        self.logger.info("XGBoost training complete")
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_cols, importance))
        
        return self
    
    def predict(
        self,
        steps: int,
        future_features: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if future_features is None:
            raise ValueError("XGBoost requires future_features for prediction")
        
        X_pred = future_features[self.feature_cols]
        predictions = self.model.predict(X_pred)
        
        return {
            "mean": predictions,
            "lower": predictions * 0.95,
            "upper": predictions * 1.05,
            "std": np.zeros_like(predictions)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def summary(self) -> str:
        """Return model summary."""
        if self.model is None:
            return "Model not fitted"
        
        importance = self.get_feature_importance().head(10)
        return f"XGBoost Model\\nTop 10 Features:\\n{importance.to_string()}"
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance
        }
        
        joblib.dump(save_dict, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> "XGBoostModel":
        """Load model from disk."""
        save_dict = joblib.load(filepath)
        
        self.model = save_dict["model"]
        self.feature_cols = save_dict["feature_cols"]
        self.target_col = save_dict["target_col"]
        self.metrics = save_dict["metrics"]
        self.feature_importance = save_dict["feature_importance"]
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from {filepath}")
        return self