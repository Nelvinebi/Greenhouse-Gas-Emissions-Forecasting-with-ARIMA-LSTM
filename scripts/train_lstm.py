#!/usr/bin/env python
"""LSTM training script."""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("=" * 60)
print("GHG EMISSIONS FORECASTING - LSTM TRAINING")
print("=" * 60)

# Load data
print("\n1. Loading data...")
data_path = "data/raw/enhanced_ghg_emissions_dataset.csv"
df = pd.read_csv(data_path, index_col='Date', parse_dates=True)

# Feature engineering
print("2. Preparing features...")
target_col = 'GHG_Emissions_MTCO2e'
feature_cols = ['Industrial_Production_Index', 'Temperature_Anomaly_C', 
                'Energy_Price_Index', 'month_sin', 'month_cos']

# Add time features
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

# Select features
df_model = df[[target_col] + feature_cols].copy()

# Scale data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_model),
    columns=df_model.columns,
    index=df_model.index
)

# Create sequences
def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Target is first column
    return np.array(X), np.array(y)

seq_length = 12
X, y = create_sequences(df_scaled.values, seq_length)

# Split data (maintain temporal order)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training sequences: {X_train.shape}")
print(f"Test sequences: {X_test.shape}")

# Build LSTM model
print("\n3. Building LSTM model...")
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(seq_length, len(feature_cols)+1))),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='huber', metrics=['mae'])

print(model.summary())

# Train
print("\n4. Training...")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Predict
print("\n5. Generating predictions...")
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform
# Create dummy array for inverse scaling
dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)+1))
dummy_pred[:, 0] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test), len(feature_cols)+1))
dummy_actual[:, 0] = y_test
y_actual = scaler.inverse_transform(dummy_actual)[:, 0]

# Metrics
print("\n6. Evaluation Metrics:")
mae = np.mean(np.abs(y_actual - y_pred))
rmse = np.sqrt(np.mean((y_actual - y_pred)**2))
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
r2 = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - np.mean(y_actual))**2)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

# Plot
print("\n7. Creating visualization...")
plt.figure(figsize=(12, 6))

# Get dates for test period
test_dates = df_model.index[train_size+seq_length:train_size+seq_length+len(y_actual)]

plt.plot(df_model.index[:train_size+seq_length], 
         df_model[target_col].iloc[:train_size+seq_length], 
         label='Training', color='blue', alpha=0.7)
plt.plot(test_dates, y_actual, label='Actual', color='green', linewidth=2)
plt.plot(test_dates, y_pred, label='LSTM Forecast', color='red', linestyle='--', linewidth=2)

plt.title('GHG Emissions Forecast - LSTM')
plt.xlabel('Date')
plt.ylabel('Emissions (MTCO2e)')
plt.legend()
plt.grid(True, alpha=0.3)

os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/lstm_forecast.png', dpi=150, bbox_inches='tight')
print("Saved: reports/figures/lstm_forecast.png")
plt.close()

# Save model
os.makedirs('models/saved', exist_ok=True)
model.save('models/saved/lstm_model.keras')
joblib.dump(scaler, 'models/saved/lstm_scaler.pkl')
print("Saved: models/saved/lstm_model.keras")

print("\n" + "=" * 60)
print("LSTM TRAINING COMPLETE")
print("=" * 60)