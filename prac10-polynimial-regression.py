# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate a synthetic dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X**2 + np.random.randn(100, 1)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transforming the data to include polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Training the Polynomial Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_poly_train, y_train)

# Making predictions
X_new=np.linspace(0, 2, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

# Plotting the dataset and the model's predictions
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_new, y_new, color='red', label='Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Evaluating the model
y_pred = lin_reg.predict(X_poly_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
