#KNN Normalization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Generate a synthetic dataset for demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization using MinMaxScaler
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Lists to store performance metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
k_values = range(1, 26)  # Different values of k to test

# Loop over various values of k
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_normalized, y_train)  # Training using the normalized data
    predicted = model.predict(X_test_normalized)
    
    # Calculate performance metrics
    accuracy_list.append(accuracy_score(y_test, predicted))
    precision_list.append(precision_score(y_test, predicted, average='macro'))
    recall_list.append(recall_score(y_test, predicted, average='macro'))
    f1_list.append(f1_score(y_test, predicted, average='macro'))

plt.figure(figsize=(12, 8))
plt.plot(k_values, accuracy_list, label='Accuracy')
plt.plot(k_values, precision_list, label='Precision')
plt.plot(k_values, recall_list, label='Recall')
plt.plot(k_values, f1_list, label='F1 Score')
plt.xlabel('k')
plt.ylabel('Score')
plt.legend()
plt.title('Performance Metrics for Different k Values')

# Explicitly show the plot
plt.show()