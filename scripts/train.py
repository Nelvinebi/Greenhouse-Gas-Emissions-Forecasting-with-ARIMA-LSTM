#!/usr/bin/env python
"""Training script for GHG forecasting models."""
import sys
import os
from pathlib import Path

# Add project root to path BEFORE any other imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import after path is set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

# Import statsmodels directly (no package imports)
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

print("=" * 60)
print("GHG EMISSIONS FORECASTING - MODEL TRAINING")
print("=" * 60)

# Load your data
print("\n1. Loading data...")
data_path = "data/raw/enhanced_ghg_emissions_dataset.csv"

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found!")
    print("Please place your CSV file in data/raw/ folder")
    sys.exit(1)

df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Feature engineering
print("\n2. Engineering features...")
df_features = df.copy()

# Add lag features
target_col = 'GHG_Emissions_MTCO2e'
for lag in [1, 2, 3, 6, 12]:
    df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)

# Add rolling features
for window in [3, 6, 12]:
    df_features[f'{target_col}_rolling_mean_{window}'] = df_features[target_col].rolling(window).mean()
    df_features[f'{target_col}_rolling_std_{window}'] = df_features[target_col].rolling(window).std()

# Add cyclical features
df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12)
df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12)

# Drop NaN
df_features = df_features.dropna()
print(f"Features shape: {df_features.shape}")

# Split data
print("\n3. Splitting data...")
train_data = df_features[df_features.index <= '2021-12-31']
test_data = df_features[df_features.index >= '2022-01-01']
print(f"Train: {len(train_data)} rows")
print(f"Test: {len(test_data)} rows")

# Train SARIMAX with auto_arima
print("\n4. Training SARIMAX with auto_arima...")
y_train = train_data[target_col]
exog_cols = ['Industrial_Production_Index', 'Temperature_Anomaly_C', 'Energy_Price_Index']
exog_train = train_data[exog_cols] if all(c in train_data.columns for c in exog_cols) else None

if exog_train is not None:
    print(f"Using exogenous variables: {exog_cols}")
    
    # Auto ARIMA
    auto_model = pm.auto_arima(
        y_train,
        exogenous=exog_train,
        seasonal=True,
        m=12,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=None,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        D=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print(f"Best order: {auto_model.order}")
    print(f"Best seasonal order: {auto_model.seasonal_order}")
    
    # Fit final SARIMAX
    final_model = SARIMAX(
        y_train,
        exog=exog_train,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
else:
    # Univariate
    auto_model = pm.auto_arima(
        y_train,
        seasonal=True,
        m=12,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    final_model = SARIMAX(
        y_train,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

print("SARIMAX training complete")

# Predict
print("\n5. Generating predictions...")
exog_test = test_data[exog_cols] if exog_train is not None else None

forecast = final_model.get_forecast(steps=len(test_data), exog=exog_test)
y_pred = forecast.predicted_mean.values
conf_int = forecast.conf_int()

y_true = test_data[target_col].values

# Calculate metrics
print("\n6. Evaluation Metrics:")
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

# Plot
print("\n7. Creating visualization...")
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data[target_col], label='Training', color='blue', alpha=0.7)
plt.plot(test_data.index, y_true, label='Actual', color='green', linewidth=2)
plt.plot(test_data.index, y_pred, label='SARIMAX Forecast', color='red', linestyle='--', linewidth=2)
plt.fill_between(test_data.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                 alpha=0.2, color='red', label='95% CI')
plt.title('GHG Emissions Forecast - SARIMAX')
plt.xlabel('Date')
plt.ylabel('Emissions (MTCO2e)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/sarimax_forecast.png', dpi=150, bbox_inches='tight')
print("Saved: reports/figures/sarimax_forecast.png")
plt.close()

# Save model
os.makedirs('models/saved', exist_ok=True)
model_data = {
    'model': final_model,
    'order': auto_model.order,
    'seasonal_order': auto_model.seasonal_order,
    'exog_columns': exog_cols if exog_train is not None else None
}
joblib.dump(model_data, 'models/saved/sarimax_model.pkl')
print("Saved: models/saved/sarimax_model.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)