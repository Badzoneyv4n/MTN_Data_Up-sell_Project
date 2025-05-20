from sklearn.metrics import precision_score, auc, roc_curve , mean_absolute_error , r2_score, confusion_matrix, ConfusionMatrixDisplay , classification_report
import matplotlib.pyplot as plt

def display_metrics(y_true = any, y_pred = any, y_scores = any , mode = 'class'):
    """
    Display classification metrics including accuracy, precision, recall, and AUC-ROC.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - mode: represent the metrics type regr or class default = 'regr'
    """
    
    if( mode == 'class'):
        print("Classification Metrics:")
    
        precision = precision_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_scores )
        roc_auc = auc(fpr, tpr)

        print(f"Precision: {precision:.4f}")
        print(f"AUC-ROC: {roc_auc:.4f}")
    
    if( mode == 'multi' ):
        print('Multi Classification Metrics:')

        cr = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print("Classification Report:")
        print(cr)
        print("Confusion Matrix:")
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
        except Exception as e:
            print(f"Could not plot confusion matrix: {e}")

    if( mode == 'regr' ):
        
        print('Regression Metrics:')

        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)

        print(f'Mae : {mae:.4f}')
        print(f'R2_Score: {r2:.4f}')