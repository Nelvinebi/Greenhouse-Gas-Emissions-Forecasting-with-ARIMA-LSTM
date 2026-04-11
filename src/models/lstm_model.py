"""LSTM model with attention mechanism."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import joblib
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Bidirectional, Input, Concatenate,
        Attention, LayerNormalization, GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from src.models.base_model import BaseModel
    from src.utils.config import Config
    from src.utils.helpers import create_sequences
except ImportError:
    from base_model import BaseModel
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import Config
    from utils.helpers import create_sequences


class LSTMModel(BaseModel):
    """Bidirectional LSTM with attention for time series forecasting."""
    
    def __init__(self, config: Config = None, name: str = "lstm"):
        super().__init__(config, name)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.model = None
        self.sequence_length = self.config.get("lstm.training.sequence_length", 12)
        self.target_col = self.config.get("data.target_column")
        self.feature_cols = None
        self.history = None
        
    def build_model(self, n_features: int) -> Model:
        """Build LSTM architecture from config."""
        arch_config = self.config.get("lstm.architecture", {})
        
        inputs = Input(shape=(self.sequence_length, n_features))
        
        # First LSTM layer
        x = Bidirectional(
            LSTM(
                128,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1
            )
        )(inputs)
        x = LayerNormalization()(x)
        
        # Second LSTM layer
        x = LSTM(
            64,
            return_sequences=True,
            dropout=0.2
        )(x)
        x = LayerNormalization()(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        x = GlobalAveragePooling1D()(attention)
        
        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer_config = self.config.get("lstm.optimizer", {})
        optimizer = Adam(
            learning_rate=optimizer_config.get("learning_rate", 0.001),
            clipnorm=optimizer_config.get("clipnorm", 1.0)
        )
        
        model.compile(
            optimizer=optimizer,
            loss=self.config.get("lstm.loss", "huber"),
            metrics=self.config.get("lstm.metrics", ["mae", "mse"])
        )
        
        return model
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM."""
        self.feature_cols = [c for c in df.columns if c != self.target_col]
        
        data = df[[self.target_col] + self.feature_cols].values
        
        X, y = create_sequences(
            data,
            seq_length=self.sequence_length,
            target_idx=0
        )
        
        return X, y
    
    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        **kwargs
    ) -> "LSTMModel":
        """Train LSTM model."""
        self.logger.info("Preparing data for LSTM...")
        
        X_train, y_train = self.prepare_data(train_data)
        
        self.logger.info(f"Training sequences: {X_train.shape}")
        
        # Build model
        if self.model is None:
            n_features = X_train.shape[2]
            self.model = self.build_model(n_features)
            self.logger.info(self.model.summary())
        
        # Validation data
        validation_split = self.config.get("lstm.training.validation_split", 0.2)
        val_data = None
        
        if validation_data is not None:
            X_val, y_val = self.prepare_data(validation_data[0])
            val_data = (X_val, y_val)
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        es_config = self.config.get("lstm.callbacks.early_stopping", {})
        callbacks.append(EarlyStopping(
            monitor=es_config.get("monitor", "val_loss"),
            patience=es_config.get("patience", 20),
            restore_best_weights=es_config.get("restore_best_weights", True),
            min_delta=es_config.get("min_delta", 0.0001)
        ))
        
        # Learning rate reduction
        lr_config = self.config.get("lstm.callbacks.reduce_lr", {})
        callbacks.append(ReduceLROnPlateau(
            monitor=lr_config.get("monitor", "val_loss"),
            factor=lr_config.get("factor", 0.5),
            patience=lr_config.get("patience", 10),
            min_lr=lr_config.get("min_lr", 0.00001)
        ))
        
        # Training
        self.logger.info("Training LSTM model...")
        
        batch_size = self.config.get("lstm.training.batch_size", 32)
        epochs = self.config.get("lstm.training.epochs", 200)
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split if val_data is None else 0.0,
            validation_data=val_data,
            callbacks=callbacks,
            shuffle=self.config.get("lstm.training.shuffle", False),
            verbose=1
        )
        
        self.is_fitted = True
        self.logger.info("LSTM training complete")
        
        # Store metrics
        self.metrics["final_loss"] = self.history.history['loss'][-1]
        self.metrics["final_val_loss"] = self.history.history.get('val_loss', [None])[-1]
        
        return self
    
    def predict(
        self,
        steps: int,
        last_sequence: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts using recursive prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        current_sequence = last_sequence.copy() if last_sequence is not None else None
        
        for _ in range(steps):
            if current_sequence is None:
                raise ValueError("Must provide last_sequence for prediction")
            
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Predict
            pred = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence (rolling window)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred  # Update target position
        
        predictions = np.array(predictions)
        
        return {
            "mean": predictions,
            "lower": predictions * 0.95,  # Approximate intervals
            "upper": predictions * 1.05,
            "std": np.zeros_like(predictions)
        }
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict on batch of sequences."""
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        keras_path = filepath.replace('.pkl', '_keras')
        self.model.save(keras_path)
        
        # Save metadata
        metadata = {
            "sequence_length": self.sequence_length,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "metrics": self.metrics,
            "history": self.history.history if self.history else None,
            "keras_model_path": keras_path
        }
        
        joblib.dump(metadata, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> "LSTMModel":
        """Load model from disk."""
        metadata = joblib.load(filepath)
        
        self.sequence_length = metadata["sequence_length"]
        self.feature_cols = metadata["feature_cols"]
        self.target_col = metadata["target_col"]
        self.metrics = metadata["metrics"]
        
        # Load Keras model
        self.model = tf.keras.models.load_model(metadata["keras_model_path"])
        
        self.is_fitted = True
        self.logger.info(f"Model loaded from {filepath}")
        return self