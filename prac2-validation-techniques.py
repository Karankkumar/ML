#Validation Techniques

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

# Load the Titanic dataset (adjust the file path accordingly)
titanic = pd.read_csv(r"titanic/train.csv")

# Prepare features and target variable
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# Holdout Method
X_train_holdout, X_test_holdout, y_train_holdout, y_test_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train_kfold, X_test_kfold = X.iloc[train_index], X.iloc[test_index]
    y_train_kfold, y_test_kfold = y.iloc[train_index], y.iloc[test_index]

# Stratified Random Sampling
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Bootstrap Sampling
n_samples = len(titanic)
indices = np.random.randint(0, n_samples, size=n_samples)
X_bootstrap = X.iloc[indices]
y_bootstrap = y.iloc[indices]

# Note: Saving data to CSV is commented out; uncomment to use
X_train_holdout.to_csv('X_train_holdout.csv', index=False)
y_train_holdout.to_csv('y_train_holdout.csv', index=False)
X_test_holdout.to_csv('X_test_holdout.csv', index=False)
y_test_holdout.to_csv('y_test_holdout.csv', index=False)

# Repeat saving for K-Fold, Stratified, and Bootstrap samples as needed

print("Data preparation methods implemented and data saved.")
