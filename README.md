# 📊 Time Series Analysis - Electric Production

> A complete guide to analyzing and forecasting electric production data using Python

---

## 🎯 Overview

This project performs time series analysis on the **Electric Production dataset** to identify patterns and forecast future values.

---

## 📋 Tasks Completed

- ✅ Load and explore data
- ✅ Find null values
- ✅ Identify seasonal/cyclical patterns
- ✅ Check stationarity
- ✅ Make data stationary (differencing)
- ✅ Forecast future values
- ✅ Visualize original vs forecasted data

---

## 🚀 Step-by-Step Implementation

### 1️⃣ **Load the Dataset** 📂

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("Electric_Production.csv")

# Convert DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Set DATE as index
df.set_index('DATE', inplace=True)

# Display first few rows
df.head()
```

**💡 Remember**: Always set datetime as index for time series!

---

### 2️⃣ **Find Null Values** 🔍

```python
# Check for missing values
df.isnull().sum()
```

**💡 Remember**: Use `.isnull().sum()` to count missing values per column

---

### 3️⃣ **Visualize Seasonal/Cyclical Pattern** 📈

```python
# Plot the time series
plt.figure(figsize=(10,4))
plt.plot(df, label="Electric Production")
plt.title("Monthly Electric Production in the US")
plt.xlabel("Year")
plt.ylabel("Production")
plt.legend()
plt.show()
```

**💡 Remember**: Visual inspection helps identify trends and seasonality

---

### 4️⃣ **Check Stationarity** 🎲

```python
from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller test
result = adfuller(df['IPG2211A2N'])

print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

**💡 Remember**:

- **p-value < 0.05** → Stationary ✅
- **p-value > 0.05** → Non-stationary ❌

---

### 5️⃣ **Make Data Stationary** 🔄

```python
# Apply differencing
df_diff = df.diff().dropna()

# Check stationarity again
result_diff = adfuller(df_diff['IPG2211A2N'])

print('ADF Statistic (Differenced):', result_diff[0])
print('p-value (Differenced):', result_diff[1])
```

**💡 Remember**: `.diff()` removes trend by subtracting consecutive values

---

### 6️⃣ **Forecast Future Values** 🔮

```python
from statsmodels.tsa.arima.model import ARIMA

# Create ARIMA model (p=1, d=1, q=1)
model = ARIMA(df, order=(1,1,1))

# Fit the model
model_fit = model.fit()

# Forecast next 12 periods
forecast = model_fit.forecast(steps=12)

print(forecast)
```

**💡 Remember**: ARIMA(p,d,q)

- **p** = AR terms (lag observations)
- **d** = Differencing order
- **q** = MA terms (lag forecast errors)

---

### 7️⃣ **Plot Original vs Forecasted** 📊

```python
# Plot both original and forecasted data
plt.figure(figsize=(10,5))
plt.plot(df, label='Original Data')
plt.plot(forecast.index, forecast, label='Forecasted Data', color='red')
plt.title("Electric Production Forecast")
plt.xlabel("Year")
plt.ylabel("Production")
plt.legend()
plt.show()
```

**💡 Remember**: Red line = forecast, Blue line = original data

---

## 📦 Required Libraries

```python
pip install pandas matplotlib statsmodels
```

**Library Usage**:

- 🐼 **pandas** - Data manipulation
- 📊 **matplotlib** - Visualization
- 📈 **statsmodels** - Time series analysis & ARIMA

---

## 🧠 Key Concepts Cheat Sheet

| Concept          | What it Does                | Code                         |
| ---------------- | --------------------------- | ---------------------------- |
| **ADF Test**     | Tests if data is stationary | `adfuller(data)`             |
| **Differencing** | Removes trend               | `.diff()`                    |
| **ARIMA**        | Forecasting model           | `ARIMA(data, order=(p,d,q))` |
| **Forecast**     | Predict future values       | `.forecast(steps=n)`         |

---

## 🎨 Quick Commands Reference

```python
# Load data
df = pd.read_csv("file.csv")

# Check nulls
df.isnull().sum()

# Test stationarity
adfuller(df['column'])

# Make stationary
df_diff = df.diff().dropna()

# Build model
model = ARIMA(df, order=(1,1,1))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=12)

# Plot
plt.plot(df)
plt.show()
```

---

## 📌 Tips & Tricks

- 🔹 Always check for null values before analysis
- 🔹 Visualize data first to understand patterns
- 🔹 Non-stationary data needs differencing
- 🔹 Use ADF test to confirm stationarity
- 🔹 Start with simple ARIMA(1,1,1) model
- 🔹 Increase forecast steps for longer predictions

---

## 🎓 Understanding Results

**Stationarity Check**:

```
If p-value < 0.05 → Data is stationary 🎉
If p-value > 0.05 → Apply differencing 🔄
```

**ARIMA Model**:

```
ARIMA(1,1,1) = Simple model
- 1st order AR
- 1st order differencing
- 1st order MA
```

---

## 🏆 Results

✨ Successfully built a time series forecasting model
✨ Forecasted 12 future periods
✨ Visualized trends and predictions
✨ Achieved stationarity through differencing

---

## 🤝 Author

**Machine Learning Practical**  
EL 4152 - Data Science  
University Year 4.1
