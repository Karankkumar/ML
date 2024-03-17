from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def calculate_performance_metrics(y_true, y_pred):
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Print the confusion matrix and performance metrics
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Example usage
# Actual and predicted values (example)
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]

calculate_performance_metrics(y_true, y_pred)
