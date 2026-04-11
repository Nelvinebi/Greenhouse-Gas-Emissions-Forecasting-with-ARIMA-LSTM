"""Hybrid model combining ARIMA and LSTM."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import joblib
from pathlib import Path

try:
    from src.models.base_model import BaseModel
    from src.models.sarimax_model import SARIMAXModel
    from src.models.lstm_model import LSTMModel
    from src.utils.config import Config
except ImportError:
    from base_model import BaseModel
    from sarimax_model import SARIMAXModel
    from lstm_model import LSTMModel
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import Config


class HybridModel(BaseModel):
    """Hybrid model: ARIMA for linear patterns, LSTM for residuals."""
    
    def __init__(self, config: Config = None, name: str = "hybrid"):
        super().__init__(config, name)
        
        self.arima_model = SARIMAXModel(config, name="arima_component")
        self.lstm_model = LSTMModel(config, name="lstm_component")
        
        self.method = self.config.get("hybrid.method", "residual_correction")
        self.arima_weight = self.config.get("hybrid.arima_weight", 0.4)
        self.lstm_weight = self.config.get("hybrid.lstm_weight", 0.6)
        
        self.residuals_train = None
        
    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        **kwargs
    ) -> "HybridModel":
        """Train hybrid model."""
        self.logger.info("Training Hybrid Model...")
        
        # Step 1: Train ARIMA
        self.logger.info("Step 1/3: Training ARIMA component...")
        
        exog_cols = self.config.get("features.exogenous", [])
        if exog_cols:
            exog_train = train_data[exog_cols]
        else:
            exog_train = None
        
        self.arima_model.fit(train_data, exogenous=exog_train, auto_select=True)
        
        # Step 2: Get ARIMA predictions and calculate residuals
        self.logger.info("Step 2/3: Calculating residuals...")
        
        arima_pred_train = self.arima_model.model.fittedvalues
        y_train = train_data[self.config.get("data.target_column")]
        
        # Calculate residuals (actual - ARIMA prediction)
        self.residuals_train = y_train - arima_pred_train
        
        # Step 3: Train LSTM on residuals
        self.logger.info("Step 3/3: Training LSTM on residuals...")
        
        # Prepare data with residuals as target
        residual_data = train_data.copy()
        residual_data[self.config.get("data.target_column")] = self.residuals_train
        
        self.lstm_model.fit(residual_data, validation_data=validation_data)
        
        self.is_fitted = True
        self.logger.info("Hybrid model training complete")
        
        return self
    
    def predict(
        self,
        steps: int,
        exogenous: Optional[pd.DataFrame] = None,
        last_sequence: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate hybrid forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # ARIMA prediction
        arima_pred = self.arima_model.predict(steps, exogenous=exogenous)
        arima_mean = arima_pred["mean"]
        
        # LSTM prediction on residuals
        lstm_pred = self.lstm_model.predict(steps, last_sequence=last_sequence)
        lstm_mean = lstm_pred["mean"]
        
        # Combine predictions
        if self.method == "residual_correction":
            # Add LSTM residual correction to ARIMA
            final_pred = arima_mean + lstm_mean
        else:
            # Weighted ensemble
            final_pred = (self.arima_weight * arima_mean + 
                         self.lstm_weight * (arima_mean + lstm_mean))
        
        # Uncertainty combination (simplified)
        combined_std = np.sqrt(arima_pred["std"]**2 + lstm_pred["std"]**2)
        
        return {
            "mean": final_pred,
            "lower": final_pred - 1.96 * combined_std,
            "upper": final_pred + 1.96 * combined_std,
            "std": combined_std,
            "arima_component": arima_mean,
            "lstm_component": lstm_mean
        }
    
    def summary(self) -> str:
        """Return model summary."""
        return f"""Hybrid Model Summary:
- Method: {self.method}
- ARIMA Weight: {self.arima_weight}
- LSTM Weight: {self.lstm_weight}
- ARIMA Order: {self.arima_model.order if self.arima_model.is_fitted else 'Not fitted'}
- LSTM Sequence Length: {self.lstm_model.sequence_length if self.lstm_model.is_fitted else 'Not fitted'}
"""
    
    def save(self, filepath: str):
        """Save hybrid model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save components separately
        base_path = filepath.replace('.pkl', '')
        
        self.arima_model.save(f"{base_path}_arima.pkl")
        self.lstm_model.save(f"{base_path}_lstm.pkl")
        
        metadata = {
            "method": self.method,
            "arima_weight": self.arima_weight,
            "lstm_weight": self.lstm_weight,
            "residuals_train": self.residuals_train,
            "arima_path": f"{base_path}_arima.pkl",
            "lstm_path": f"{base_path}_lstm.pkl"
        }
        
        joblib.dump(metadata, filepath)
        self.logger.info(f"Hybrid model saved to {filepath}")
    
    def load(self, filepath: str) -> "HybridModel":
        """Load hybrid model."""
        metadata = joblib.load(filepath)
        
        self.method = metadata["method"]
        self.arima_weight = metadata["arima_weight"]
        self.lstm_weight = metadata["lstm_weight"]
        self.residuals_train = metadata["residuals_train"]
        
        self.arima_model.load(metadata["arima_path"])
        self.lstm_model.load(metadata["lstm_path"])
        
        self.is_fitted = True
        self.logger.info(f"Hybrid model loaded from {filepath}")
        return self