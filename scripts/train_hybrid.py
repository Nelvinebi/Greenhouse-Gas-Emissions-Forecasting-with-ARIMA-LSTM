#!/usr/bin/env python
"""Hybrid model: SARIMA + LSTM on residuals."""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

print("=" * 60)
print("GHG EMISSIONS FORECASTING - HYBRID MODEL")
print("=" * 60)

# Load data
print("\n1. Loading data...")
data_path = "data/raw/enhanced_ghg_emissions_dataset.csv"
df = pd.read_csv(data_path, index_col='Date', parse_dates=True)

# Feature engineering
print("2. Preparing features...")
target_col = 'GHG_Emissions_MTCO2e'
exog_cols = ['Industrial_Production_Index', 'Temperature_Anomaly_C', 'Energy_Price_Index']

# Add time features
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

# Split data
train_data = df[df.index <= '2021-12-31']
test_data = df[df.index >= '2022-01-01']

y_train = train_data[target_col]
y_test = test_data[target_col]
exog_train = train_data[exog_cols]
exog_test = test_data[exog_cols]

print(f"Train: {len(train_data)} rows")
print(f"Test: {len(test_data)} rows")

# Step 1: Train SARIMAX
print("\n3. Training SARIMAX component...")
auto_model = pm.auto_arima(
    y_train,
    exogenous=exog_train,
    seasonal=True, m=12,
    start_p=0, max_p=3, start_q=0, max_q=3,
    start_P=0, max_P=2, start_Q=0, max_Q=2,
    trace=False, error_action='ignore',
    suppress_warnings=True, stepwise=True
)

sarimax_model = SARIMAX(
    y_train, exog=exog_train,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

print(f"SARIMAX order: {auto_model.order}")
print(f"SARIMAX seasonal order: {auto_model.seasonal_order}")

# Get SARIMA fitted values and residuals
sarima_fitted = sarimax_model.fittedvalues
residuals_train = y_train - sarima_fitted

print(f"\nSARIMA residual std: {residuals_train.std():.4f}")

# Step 2: Train LSTM on residuals
print("\n4. Training LSTM on residuals...")

# Prepare residual data for LSTM
residual_df = pd.DataFrame({
    'residuals': residuals_train,
    'month_sin': train_data['month_sin'],
    'month_cos': train_data['month_cos']
})

# Scale residuals
scaler = MinMaxScaler()
residual_scaled = scaler.fit_transform(residual_df)

# Create sequences
seq_length = 12
X_resid, y_resid = [], []
for i in range(seq_length, len(residual_scaled)):
    X_resid.append(residual_scaled[i-seq_length:i])
    y_resid.append(residual_scaled[i, 0])

X_resid, y_resid = np.array(X_resid), np.array(y_resid)

print(f"Residual sequences: {X_resid.shape}")

# Build LSTM for residuals
resid_model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(seq_length, 3))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

resid_model.compile(optimizer='adam', loss='huber', metrics=['mae'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001)

resid_model.fit(
    X_resid, y_resid,
    epochs=100, batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

print("LSTM residual training complete")

# Step 3: Generate hybrid predictions
print("\n5. Generating hybrid predictions...")

# SARIMA forecast
sarima_forecast = sarimax_model.get_forecast(steps=len(test_data), exog=exog_test)
sarima_mean = sarima_forecast.predicted_mean.values

# LSTM residual forecast (recursive)
residual_predictions = []
last_sequence = residual_scaled[-seq_length:]

for i in range(len(test_data)):
    X_pred = last_sequence.reshape(1, seq_length, 3)
    pred_scaled = resid_model.predict(X_pred, verbose=0)[0, 0]
    
    # Inverse transform
    dummy = np.zeros((1, 3))
    dummy[0, 0] = pred_scaled
    pred = scaler.inverse_transform(dummy)[0, 0]
    residual_predictions.append(pred)
    
    # Update sequence
    last_sequence = np.roll(last_sequence, -1, axis=0)
    last_sequence[-1, 0] = pred_scaled

residual_predictions = np.array(residual_predictions)

# Combine: SARIMA + LSTM residuals
hybrid_pred = sarima_mean + residual_predictions

# Evaluate
y_true = y_test.values

print("\n6. Evaluation Metrics:")

# SARIMA only
mae_sarima = np.mean(np.abs(y_true - sarima_mean))
rmse_sarima = np.sqrt(np.mean((y_true - sarima_mean)**2))
mape_sarima = np.mean(np.abs((y_true - sarima_mean) / y_true)) * 100

print(f"\nSARIMA Only:")
print(f"  MAE:  {mae_sarima:.4f}")
print(f"  RMSE: {rmse_sarima:.4f}")
print(f"  MAPE: {mape_sarima:.2f}%")

# Hybrid
mae_hybrid = np.mean(np.abs(y_true - hybrid_pred))
rmse_hybrid = np.sqrt(np.mean((y_true - hybrid_pred)**2))
mape_hybrid = np.mean(np.abs((y_true - hybrid_pred) / y_true)) * 100
r2_hybrid = 1 - np.sum((y_true - hybrid_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

print(f"\nHybrid (SARIMA + LSTM):")
print(f"  MAE:  {mae_hybrid:.4f}")
print(f"  RMSE: {rmse_hybrid:.4f}")
print(f"  MAPE: {mape_hybrid:.2f}%")
print(f"  R²:   {r2_hybrid:.4f}")

# Improvement
improvement = ((mape_sarima - mape_hybrid) / mape_sarima) * 100
print(f"\nImprovement over SARIMA: {improvement:.2f}%")

# Plot
print("\n7. Creating visualization...")
plt.figure(figsize=(12, 6))

plt.plot(train_data.index, y_train, label='Training', color='blue', alpha=0.7)
plt.plot(test_data.index, y_true, label='Actual', color='green', linewidth=2)
plt.plot(test_data.index, sarima_mean, label='SARIMA Only', color='orange', linestyle='--', alpha=0.8)
plt.plot(test_data.index, hybrid_pred, label='Hybrid (SARIMA+LSTM)', color='red', linestyle='--', linewidth=2)

plt.title('GHG Emissions Forecast - Hybrid Model')
plt.xlabel('Date')
plt.ylabel('Emissions (MTCO2e)')
plt.legend()
plt.grid(True, alpha=0.3)

os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/hybrid_forecast.png', dpi=150, bbox_inches='tight')
print("Saved: reports/figures/hybrid_forecast.png")
plt.close()

# Save models
os.makedirs('models/saved', exist_ok=True)
joblib.dump({
    'sarimax': sarimax_model,
    'resid_lstm': resid_model,
    'scaler': scaler,
    'order': auto_model.order,
    'seasonal_order': auto_model.seasonal_order
}, 'models/saved/hybrid_model.pkl')
print("Saved: models/saved/hybrid_model.pkl")

print("\n" + "=" * 60)
print("HYBRID TRAINING COMPLETE")
print("=" * 60)