from sklearn.model_selection import train_test_split
from models.model_training import train_model
from models.model_evaluation import evaluate_model
import numpy as np

def test_evaluate_model():
    X = np.random.rand(100, 3072)
    y = np.tile(np.arange(10), 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model = train_model('sgd', X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics['accuracy'] > 0
    assert 'precision' in metrics
