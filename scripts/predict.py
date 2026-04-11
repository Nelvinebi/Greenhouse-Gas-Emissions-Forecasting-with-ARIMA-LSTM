#!/usr/bin/env python
"""Prediction script for making forecasts with trained model."""
import sys
import os
from pathlib import Path
import argparse

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def predict_future(steps=12, model_path='models/saved/sarimax_model.pkl'):
    """
    Make future predictions using trained SARIMAX model.
    
    Args:
        steps: Number of months to forecast (default: 12)
        model_path: Path to saved model
    """
    print("=" * 60)
    print("GHG EMISSIONS FORECASTING - PREDICTION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please train the model first: python scripts/train.py")
        return
    
    # Load model
    print(f"\n1. Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    order = model_data.get('order', 'N/A')
    seasonal_order = model_data.get('seasonal_order', 'N/A')
    exog_columns = model_data.get('exog_columns', None)
    
    print(f"   Model order: {order}")
    print(f"   Seasonal order: {seasonal_order}")
    
    # Generate future dates
    last_date = pd.Timestamp('2024-12-31')  # Last date in training data
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq='M'
    )
    
    print(f"\n2. Forecasting {steps} months ahead...")
    print(f"   From: {future_dates[0].strftime('%Y-%m')}")
    print(f"   To: {future_dates[-1].strftime('%Y-%m')}")
    
    # Prepare exogenous variables (you need to provide these)
    if exog_columns:
        print(f"\n3. Exogenous variables required: {exog_columns}")
        print("   ⚠️  You need to provide future values for these variables!")
        
        # Example: Using last known values (REPLACE WITH ACTUAL FORECASTS)
        print("\n   Using placeholder values (REPLACE WITH REAL DATA):")
        
        # In production, load these from economic forecasts
        future_exog = pd.DataFrame({
            'Industrial_Production_Index': [115 + i*0.5 for i in range(steps)],
            'Temperature_Anomaly_C': [2 + 8*np.sin(2*np.pi*i/12) for i in range(steps)],
            'Energy_Price_Index': [108 + i*0.3 for i in range(steps)]
        }, index=future_dates)
        
        print(future_exog.head())
        
        # Generate forecast
        forecast = model.get_forecast(steps=steps, exog=future_exog)
    else:
        # Univariate forecast
        forecast = model.get_forecast(steps=steps)
    
    # Extract results
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    # Create results dataframe
    results = pd.DataFrame({
        'Date': future_dates,
        'Forecast': pred_mean.values,
        'Lower_95': conf_int.iloc[:, 0].values,
        'Upper_95': conf_int.iloc[:, 1].values
    })
    
    print("\n4. Forecast Results:")
    print(results.to_string(index=False))
    
    # Save predictions
    os.makedirs('reports', exist_ok=True)
    output_file = f'reports/forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results.to_csv(output_file, index=False)
    print(f"\n5. Saved predictions to: {output_file}")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        # Load historical data for context
        hist_data = pd.read_csv(
            'data/raw/enhanced_ghg_emissions_dataset.csv',
            index_col='Date',
            parse_dates=True
        )
        
        plt.figure(figsize=(12, 6))
        
        # Historical data (last 24 months)
        hist_plot = hist_data.last('24M')
        plt.plot(hist_plot.index, hist_plot['GHG_Emissions_MTCO2e'], 
                label='Historical', color='blue', alpha=0.7)
        
        # Forecast
        plt.plot(future_dates, pred_mean, 
                label='Forecast', color='red', linewidth=2)
        plt.fill_between(future_dates, 
                        conf_int.iloc[:, 0], 
                        conf_int.iloc[:, 1],
                        alpha=0.2, color='red', label='95% CI')
        
        plt.title('GHG Emissions Forecast')
        plt.xlabel('Date')
        plt.ylabel('Emissions (MTCO2e)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = f'reports/forecast_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Make GHG emissions forecasts')
    parser.add_argument('--steps', type=int, default=12,
                       help='Number of months to forecast (default: 12)')
    parser.add_argument('--model', type=str, default='models/saved/sarimax_model.pkl',
                       help='Path to saved model')
    
    args = parser.parse_args()
    
    predict_future(steps=args.steps, model_path=args.model)


if __name__ == "__main__":
    main()