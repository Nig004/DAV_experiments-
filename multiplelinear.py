import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create sample data (2 features)
X = np.array([
    [5, 1],  # [feature1, feature2]
    [10, 2],
    [15, 3],
    [20, 4],
    [25, 5]
])
y = np.array([15, 25, 35, 45, 55])  # Target values

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Print results
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Equation: y = {model.coef_[0]:.2f}*X1 + {model.coef_[1]:.2f}*X2 + {model.intercept_:.2f}")

# Make predictions
y_pred = model.predict(X)

# Calculate R-squared
r_squared = model.score(X, y)
print(f"R-squared: {r_squared:.4f}")

# Predict for new data
new_data = np.array([[12, 3], [18, 4]])
predictions = model.predict(new_data)
print(f"Predictions for new data: {predictions}")