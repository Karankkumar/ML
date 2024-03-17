# Prac - 8 : Weighted KNN
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Generate a synthetic dataset with adjusted parameters
X, y = make_classification(n_samples=1000, n_features=10, n_informative=4, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Arrays to store performance metrics
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Range of k values to try
k_values = range(1, 26)

# Apply Weighted k-NN for each value of k
for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)

    # Calculate and store performance metrics
    accuracies.append(accuracy_score(y_test, predictions))
    precisions.append(precision_score(y_test, predictions, average='macro'))
    recalls.append(recall_score(y_test, predictions, average='macro'))
    f1_scores.append(f1_score(y_test, predictions, average='macro'))

# Plotting the results
plt.figure(figsize=(10, 8))
plt.plot(k_values, accuracies, label='Accuracy')
plt.plot(k_values, precisions, label='Precision')
plt.plot(k_values, recalls, label='Recall')
plt.plot(k_values, f1_scores, label='F1 Score')
plt.xlabel('Number of Neighbors: k')
plt.ylabel('Performance Score')
plt.title('Weighted k-NN Performance for Different k Values')
plt.legend()
plt.show()