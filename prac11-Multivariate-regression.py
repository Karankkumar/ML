#Multivariate regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating a synthetic dataset
np.random.seed(42)
X1 = 2 * np.random.rand(100, 1)  # First feature
X2 = 4 + np.random.rand(100, 1)  # Second feature
y = 3 + 4 * X1 + 5 * X2 + np.random.randn(100, 1)  # Target variable with noise

# Combining the features into a single matrix
X = np.hstack((X1, X2))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Multivariate Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print(f"Mean Squared Error (MSE): {mse}")

# Plotting (for visualization we can only plot against one variable at a time)
# Here we're visualizing the relationship with the first feature
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='red', marker='x', label='Predicted')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Multivariate Linear Regression Predictions')
plt.legend()
plt.show()
