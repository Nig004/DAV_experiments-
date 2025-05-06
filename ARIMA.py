import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load and check data
df = pd.read_csv("AirPassengers.csv")
print(df.columns)  # To check actual column names
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Fit ARIMA model
model = ARIMA(df['#Passengers'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 12 months
forecast = model_fit.forecast(steps=12)
forecast.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# Plot
plt.plot(df['#Passengers'], label='Original Data')
plt.plot(forecast, label='Forecast', color='red')
plt.title("ARIMA Forecast - AirPassengers")
plt.legend()
plt.show()
