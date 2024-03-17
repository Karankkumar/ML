# Linear Regression - Gradient Descent
import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic dataset: y = 3x + 4 + noise
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 instances, single feature
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + Gaussian noise

# Plot the generated dataset
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Dataset')
plt.show()

# Implementing Gradient Descent
X_b = np.c_[np.ones((100, 1)), X]  # Add x0 = 1 to each instance for the bias term

learning_rate = 0.01  # Step size
n_iterations = 1000  # Number of steps
m = 100  # Number of instances

theta = np.random.randn(2, 1)  # Random initialization of parameters (theta0, theta1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

print(f"Theta after Gradient Descent:\n{theta}")

# Predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add x0 = 1 to each instance
y_predict = X_new_b.dot(theta)

# Plotting the model's predictions
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Model Predictions')
plt.show()

# Evaluating the Model
from sklearn.metrics import mean_squared_error

y_train_predict = X_b.dot(theta)
mse = mean_squared_error(y, y_train_predict)
print(f"Mean Squared Error: {mse}")