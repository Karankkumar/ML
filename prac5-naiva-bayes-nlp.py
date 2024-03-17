#Naive Bayesian in NLP Domain
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Example dataset
data = {
    'text': ['I love this product', 'Great product! Highly recommend', 'Bad experience, would not recommend', 'Worst service ever', 'Happy with my purchase', 'Terrible, would not buy again'],
    'sentiment': ['positive', 'positive', 'negative', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Vectorizing text data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

y = df['sentiment']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes classification
model = MultinomialNB()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

# Performance evaluation
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average='weighted', pos_label='positive')
recall = recall_score(y_test, predicted, average='weighted', pos_label='positive')
f1 = f1_score(y_test, predicted, average='weighted', pos_label='positive')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, predicted, labels=['positive', 'negative'])

# Plotting
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()