from sklearn.model_selection import train_test_split
from models.model_training import train_model
from models.model_training import train_logistic_regression
import numpy as np

def test_train_model():
    X = np.random.rand(100, 3072)
    y = np.random.randint(0, 10, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_model('sgd', X_train, y_train)
    assert model is not None

def test_logistic_regression_training():
    X = np.random.rand(100, 3072)
    y = np.tile(np.arange(10), 10)
    model = train_logistic_regression(X, y)
    assert model is not None
    assert model.coef_.shape == (10, 3072)  # Vérifiez les dimensions des coefficients

test_logistic_regression_training()
