
# Greenhouse Gas Emissions Forecasting with ARIMA & LSTM

## 📌 Project Overview

This project focuses on forecasting **Greenhouse Gas (GHG) emissions** using two powerful time series modeling techniques:

1. **ARIMA (AutoRegressive Integrated Moving Average)** – for statistical forecasting.
2. **LSTM (Long Short-Term Memory)** – for deep learning-based sequential modeling.

Synthetic data representing **monthly GHG emissions over several years** was generated to simulate real-world patterns, trends, and seasonality. The goal is to demonstrate **hybrid forecasting approaches** to tackle environmental monitoring challenges, particularly in the context of **climate change mitigation and policy decision-making**.

---

## 🎯 Objectives

* Generate synthetic GHG emissions data for experimentation.
* Implement **ARIMA** to capture statistical patterns in time series data.
* Implement **LSTM** to learn complex temporal dependencies.
* Compare model performances for improved forecasting accuracy.
* Provide a reproducible framework for environmental data science research.

---

## 📂 Project Structure

```
Greenhouse-Gas-Emissions-Forecasting-with-ARIMA-LSTM/
│── data/
│   └── greenhouse_gas_emissions_synthetic.xlsx  # Synthetic dataset
│── notebooks/
│   └── ARIMA_LSTM_Forecasting.ipynb             # Main analysis notebook
│── src/
│   └── data_generation.py                       # Code to create synthetic data
│   └── arima_model.py                           # ARIMA model training
│   └── lstm_model.py                            # LSTM model training
│── results/
│   └── arima_forecast.png                       # ARIMA prediction plot
│   └── lstm_forecast.png                        # LSTM prediction plot
│── README.md
│── requirements.txt
```

---

## 📊 Dataset Description

The synthetic dataset contains **120+ months of GHG emissions data** with realistic trends and seasonality.

* **Columns**:

  * `Date` – Monthly timestamp (YYYY-MM)
  * `Emissions` – GHG emissions in million metric tons CO₂ equivalent (MMT CO₂e)

Example:

| Date    | Emissions |
| ------- | --------- |
| 2010-01 | 450.23    |
| 2010-02 | 455.67    |
| ...     | ...       |

---

## 🛠️ Installation & Requirements

To run this project, install the dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
pandas
matplotlib
scikit-learn
statsmodels
tensorflow
```

---

## 🚀 Usage

1. Clone the repository:

```bash
git clone https://github.com/Nelvinebi/Greenhouse-Gas-Emissions-Forecasting-with-ARIMA-LSTM.git
```

2. Navigate to the project folder:

```bash
cd Greenhouse-Gas-Emissions-Forecasting-with-ARIMA-LSTM
```

3. Run the notebook:

```bash
jupyter notebook notebooks/ARIMA_LSTM_Forecasting.ipynb
```

---

## 📈 Results

* **ARIMA** captures the linear patterns well but struggles with complex seasonality.
* **LSTM** learns long-term dependencies and adapts better to nonlinear trends.
* Combining insights from both can improve forecasting accuracy for climate-related decision-making.

---

## 🌍 Relevance to the Niger Delta

The Niger Delta faces **rising environmental concerns** due to industrial activities, oil and gas extraction, and urban expansion. Forecasting GHG emissions can help policymakers implement **targeted climate action** and **environmental protection strategies**.

---

## 👨‍💻 Author

**Name:** Agbozu Ebingiye Nelvin
**Email:** [nelvinebingiye@gmail.com](mailto:nelvinebingiye@gmail.com)
**GitHub:** [Nelvinebi](https://github.com/Nelvinebi)
