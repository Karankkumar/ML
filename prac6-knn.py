#KNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# Sample dataset: Let's assume 'data' is a DataFrame with mixed types
# data = pd.read_csv('your_dataset.csv')  # Assuming you have a dataset

# For demonstration, creating a sample DataFrame with mixed attributes
data = pd.DataFrame({
    'numerical_feature1': np.random.rand(100),
    'numerical_feature2': np.random.rand(100),
    'categorical_feature': np.random.choice(['Category1', 'Category2', 'Category3'], 100),
    'label': np.random.choice([0, 1], 100)
})

# Separating features and labels
X = data.drop('label', axis=1)
y = data['label']

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Encoding categorical features and scaling numerical features
categorical_features = ['categorical_feature']
numerical_features = ['numerical_feature1', 'numerical_feature2']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Performance metrics storage
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
k_values = range(1, 26)

# Iterating over different values of k
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    
    # Calculating and storing performance metrics
    accuracy_list.append(accuracy_score(y_test, predicted))
    precision_list.append(precision_score(y_test, predicted, average='macro'))
    recall_list.append(recall_score(y_test, predicted, average='macro'))
    f1_list.append(f1_score(y_test, predicted, average='macro'))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_list, label='Accuracy')
plt.plot(k_values, precision_list, label='Precision')
plt.plot(k_values, recall_list, label='Recall')
plt.plot(k_values, f1_list, label='F1 Score')
plt.xlabel('Number of Neighbors: k')
plt.ylabel('Performance')
plt.title('k-NN Performance for Different k Values')
plt.legend()
plt.show()