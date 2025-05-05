from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve

def display_metrics(y_true = any, y_pred = any, y_scores = any):
    """
    Display classification metrics including accuracy, precision, recall, and AUC-ROC.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    """
    print("Classification Metrics:")
    
    # accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")