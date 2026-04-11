"""Setup script for GHG Emissions Forecasting package."""
from setuptools import setup, find_packages

setup(
    name="ghg-forecast",
    version="2.0.0",
    author="Data Science Team",
    description="Advanced Greenhouse Gas Emissions Forecasting with ARIMA, LSTM, and Hybrid Models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "tensorflow>=2.13.0",
        "statsmodels>=0.14.0",
        "pmdarima>=2.0.0",
        "prophet>=1.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
    ],
)