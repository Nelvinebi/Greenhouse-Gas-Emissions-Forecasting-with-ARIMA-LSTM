"""SARIMAX model implementation with auto_arima."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import joblib
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
import pmdarima as pm

try:
    from src.models.base_model import BaseModel
    from src.utils.config import Config
except ImportError:
    from base_model import BaseModel
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import Config


class SARIMAXModel(BaseModel):
    """SARIMAX model with automated parameter selection."""
    
    def __init__(self, config: Config = None, name: str = "sarimax"):
        super().__init__(config, name)
        self.model = None
        self.auto_arima_model = None
        self.order = None
        self.seasonal_order = None
        self.exog_columns = None
        self.target_col = self.config.get("data.target_column")
    
    def fit(
        self,
        train_data: pd.DataFrame,
        exogenous: Optional[pd.DataFrame] = None,
        auto_select: bool = True,
        **kwargs
    ) -> "SARIMAXModel":
        """Train SARIMAX model."""
        self.logger.info("Training SARIMAX model...")
        
        y = train_data[self.target_col]
        
        if self.config.get("arima.use_exogenous") and exogenous is not None:
            self.exog_columns = exogenous.columns.tolist()
            X = exogenous.values
            self.logger.info(f"Using exogenous variables: {self.exog_columns}")
        else:
            X = None
        
        if auto_select:
            self.logger.info("Running auto_arima for parameter selection...")
            self.auto_arima_model = pm.auto_arima(
                y,
                exogenous=X,
                seasonal=self.config.get("arima.auto_arima.seasonal", True),
                m=self.config.get("arima.auto_arima.m", 12),
                start_p=self.config.get("arima.auto_arima.start_p", 0),
                max_p=self.config.get("arima.auto_arima.max_p", 5),
                start_q=self.config.get("arima.auto_arima.start_q", 0),
                max_q=self.config.get("arima.auto_arima.max_q", 5),
                d=self.config.get("arima.auto_arima.d"),
                start_P=self.config.get("arima.auto_arima.start_P", 0),
                max_P=self.config.get("arima.auto_arima.max_P", 2),
                start_Q=self.config.get("arima.auto_arima.start_Q", 0),
                max_Q=self.config.get("arima.auto_arima.max_Q", 2),
                D=self.config.get("arima.auto_arima.D"),
                trace=self.config.get("arima.auto_arima.trace", True),
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                information_criterion="aic",
                n_jobs=-1
            )
            
            self.order = self.auto_arima_model.order
            self.seasonal_order = self.auto_arima_model.seasonal_order
            
            self.logger.info(f"Best ARIMA order: {self.order}")
            self.logger.info(f"Best seasonal order: {self.seasonal_order}")
            
            self.model = SARIMAX(
                y,
                exog=X,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
        
        else:
            self.order = tuple(self.config.get("arima.order", [1, 1, 1]))
            self.seasonal_order = tuple(self.config.get(
                "arima.seasonal_order", [1, 1, 1, 12]
            ))
            
            self.model = SARIMAX(
                y,
                exog=X,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
        
        self.is_fitted = True
        self.logger.info("SARIMAX training complete")
        
        self.metrics["aic"] = self.model.aic
        self.metrics["bic"] = self.model.bic
        self.metrics["loglik"] = self.model.llf
        
        return self
    
    def predict(
        self,
        steps: int,
        exogenous: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts with confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.exog_columns is not None and exogenous is not None:
            X = exogenous[self.exog_columns].values
        else:
            X = None
        
        forecast = self.model.get_forecast(steps=steps, exog=X)
        
        mean_pred = forecast.predicted_mean.values
        conf_int = forecast.conf_int(alpha=alpha)
        
        return {
            "mean": mean_pred,
            "lower": conf_int.iloc[:, 0].values,
            "upper": conf_int.iloc[:, 1].values,
            "std": np.sqrt(forecast.var_pred_mean.values)
        }
    
    def diagnose(self) -> Dict[str, Any]:
        """Generate diagnostic statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        import statsmodels.api as sm
        
        residuals = self.model.resid
        
        diagnostics = {
            "ljung_box": self.model.test_serial_correlation(method="ljungbox"),
            "jarque_bera": self.model.test_normality(method="jarquebera"),
            "heteroscedasticity": self.model.test_heteroscedasticity(method="breakvar"),
            "durbin_watson": sm.stats.durbin_watson(residuals),
            "residual_mean": residuals.mean(),
            "residual_std": residuals.std()
        }
        
        return diagnostics
    
    def summary(self) -> str:
        """Return model summary."""
        if self.model is None:
            return "Model not fitted"
        return str(self.model.summary())
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model": self.model,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "exog_columns": self.exog_columns,
            "metrics": self.metrics,
            "config": self.config.to_dict()
        }
        
        joblib.dump(save_dict, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> "SARIMAXModel":
        """Load model from disk."""
        save_dict = joblib.load(filepath)
        
        self.model = save_dict["model"]
        self.order = save_dict["order"]
        self.seasonal_order = save_dict["seasonal_order"]
        self.exog_columns = save_dict["exog_columns"]
        self.metrics = save_dict["metrics"]
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from {filepath}")
        return self