"""Abstract base class for all models."""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json
from pathlib import Path

from ..utils.config import Config
from ..utils.logger import LoggerMixin


class BaseModel(ABC, LoggerMixin):
    """Abstract base class for forecasting models."""
    
    def __init__(self, config: Config = None, name: str = "base_model"):
        self.config = config or Config()
        self.name = name
        self.is_fitted = False
        self.metrics = {}
        self.history = None
    
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> "BaseModel":
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(
        self,
        steps: int,
        exogenous: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate forecasts."""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> "BaseModel":
        """Load model from disk."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}
    
    def set_params(self, **params):
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def summary(self) -> str:
        """Return model summary."""
        return f"{self.name}: Not implemented"