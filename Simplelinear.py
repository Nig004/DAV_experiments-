import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 6])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Get the parameters
slope = model.coef_[0]
intercept = model.intercept_

# Print results
print(f"Equation: y = {slope:.2f}x + {intercept:.2f}")

# Make predictions
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y)
plt.plot(X, y_pred, color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.show()