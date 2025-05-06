import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Load example time series data (generate simple data here)
dates = pd.date_range('2023-01-01', periods=50)
data = pd.Series(range(50), index=dates)

# Plot the data
data.plot(title='Time Series Data')
plt.show()

# Check stationarity (p-value)
result = adfuller(data)
print('ADF p-value:', result[1])

# Fit simple ARIMA model (p=1, d=1, q=0)
model = ARIMA(data, order=(1,1,0))
model_fit = model.fit()

# Forecast next 5 points
forecast = model_fit.forecast(5)
print("Forecast:\n", forecast)

# Plot original and forecast
data.plot()
forecast.plot(style='--')
plt.show()
