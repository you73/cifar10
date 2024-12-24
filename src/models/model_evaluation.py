from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        if y_score.shape[1] != len(np.unique(y_test)):
            raise ValueError(
                "Number of classes in y_true not equal to the number of columns in 'y_score'"
            )
        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')

    return {
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'roc_auc': roc_auc
    }
