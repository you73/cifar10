from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier  # Importation du Decision Tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import numpy as np
import joblib  # Ajouter l'importation de joblib
import os  # Pour la gestion des chemins de fichiers

def save_model(model, model_name, preprocessor, feature_extractor):
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f"{models_dir}/{model_name}_{preprocessor}_{feature_extractor}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

def train_sgd_with_minibatch(X_train, y_train, batch_size, learning_rate, kernel, search='grid', **kwargs):
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'learning_rate': ['optimal', 'constant', 'invscaling'],
        'eta0': [learning_rate],
        'max_iter': [1000],
        'tol': [1e-3]
    }
    cv = StratifiedKFold(n_splits=3)
    clf = SGDClassifier(learning_rate='optimal', eta0=learning_rate, max_iter=1, warm_start=True)
    n_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(1000): 
        for batch in range(n_batches):
            X_batch = X_train[batch*batch_size:(batch+1)*batch_size]
            y_batch = y_train[batch*batch_size:(batch+1)*batch_size]
            clf.partial_fit(X_batch, y_batch, classes=np.unique(y_train))

    if search == 'grid':
        clf = GridSearchCV(clf, param_grid, cv=cv)
    else:
        clf = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=5, cv=cv)

    clf.fit(X_train, y_train)
    save_model(clf, "sgd", kwargs.get("preprocessor", "none"), kwargs.get("feature_extractor", "none"))
    return clf

def train_logistic_regression(X_train, y_train, search='grid', preprocessor='none', feature_extractor='none'):
    unique_classes = np.unique(y_train)
    if len(unique_classes) != 10:
        raise ValueError(f"Expected 10 unique classes, but got {len(unique_classes)}. Classes: {unique_classes}")
    clf = LogisticRegression(max_iter=150, solver='saga', penalty='l2', C=1.0, fit_intercept=True)
    param_grid = {'C': [0.1, 1, 10]}
    if search == 'grid':
        clf.fit(X_train, y_train)
        save_model(clf, "logistic_regression", preprocessor, feature_extractor)
        return clf
    else:
        search_cv = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=5, cv=StratifiedKFold(n_splits=3))
    search_cv.fit(X_train, y_train)
    best_model = search_cv.best_estimator_
    save_model(best_model, "logistic_regression", preprocessor, feature_extractor)
    return best_model

def train_knn(X_train, y_train, search='grid', **kwargs):
    param_grid = {'n_neighbors': [3, 5, 7]}
    cv = StratifiedKFold(n_splits=3)
    if search == 'grid':
        clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv)
    else:
        clf = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=param_grid, n_iter=5, cv=cv)
    clf.fit(X_train, y_train)
    save_model(clf, "knn", kwargs.get("preprocessor", "none"), kwargs.get("feature_extractor", "none"))
    return clf

def train_random_forest(X_train, y_train, search='grid', **kwargs):
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}
    cv = StratifiedKFold(n_splits=3)
    if search == 'grid':
        clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=cv)
    else:
        clf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid, n_iter=5, cv=cv)
    clf.fit(X_train, y_train)
    save_model(clf, "random_forest", kwargs.get("preprocessor", "none"), kwargs.get("feature_extractor", "none"))
    return clf

def train_svm(X_train, y_train, search='grid', **kwargs):
    param_grid = { 'kernel': ['linear', 'rbf']}
    cv = StratifiedKFold(n_splits=3)
    if search == 'grid':
        clf = GridSearchCV(SVC(probability=True), param_grid, cv=cv)
    else:
        clf = RandomizedSearchCV(SVC(probability=True), param_distributions=param_grid, n_iter=5, cv=cv)
    clf.fit(X_train, y_train)
    save_model(clf, "svm", kwargs.get("preprocessor", "none"), kwargs.get("feature_extractor", "none"))
    return clf

def train_decision_tree(X_train, y_train, search='grid', **kwargs):  # Fonction pour entraîner un Decision Tree
    param_grid = { 'min_samples_split': [2, 5]}
    cv = StratifiedKFold(n_splits=3)
    if search == 'grid':
        clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cv)
    else:
        clf = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=param_grid, n_iter=5, cv=cv)
    clf.fit(X_train, y_train)
    save_model(clf, "decision_tree", kwargs.get("preprocessor", "none"), kwargs.get("feature_extractor", "none"))
    return clf

def train_model(model_name, X_train, y_train, batch_size=32, learning_rate=0.01, kernel='linear', search='grid', **kwargs):
    common_kwargs = {'preprocessor': kwargs.get('preprocessor', 'none'), 'feature_extractor': kwargs.get('feature_extractor', 'none')}
    if model_name == 'sgd':
        return train_sgd_with_minibatch(X_train, y_train, batch_size, learning_rate, kernel, search, **common_kwargs)
    elif model_name == 'logistic_regression':
        return train_logistic_regression(X_train, y_train, **common_kwargs)
    elif model_name == 'knn':
        return train_knn(X_train, y_train, search, **common_kwargs)
    elif model_name == 'random_forest':
        return train_random_forest(X_train, y_train, search, **common_kwargs)
    elif model_name == 'svm':
        return train_svm(X_train, y_train, search, **common_kwargs)
    elif model_name == 'decision_tree':  # Ajout du Decision Tree
        return train_decision_tree(X_train, y_train, search, **common_kwargs)
    else:
        raise ValueError("Model name not recognized.")
